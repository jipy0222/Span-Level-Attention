# revised version of my_transformer.py for TTIC-setting(no padding for sentences in each batch), so src_len is removed
# origin version of my_transformer.py for triaffine-setting is in deep-span-transformer/model/my_transformer.py
# origin version also contains mymulti_head_attention_forward(simple loop, too slow) 
# and mymulti_head_attention_forward_padding(simple padding, too heavy)
# the current used is mymulti_head_attention_forward_grouppadding

import copy
import warnings
import sys
import math
from typing import Optional, Tuple, Union, Callable, List

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _mha_shape_check(query: Tensor, key: Tensor, value: Tensor, num_heads: int):
    # Verifies the expected shape for `query, `key`, `value`
    # and returns if the input is batched or not.
    # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

    # Shape check.
    if query.dim() == 3:
        # Batched Inputs
        is_batched = True
        assert key.dim() == 3 and value.dim() == 3, \
            ("For batched (3-D) `query`, expected `key` and `value` to be 3-D"
             f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")

    elif query.dim() == 2:
        # Unbatched Inputs
        is_batched = False
        assert key.dim() == 2 and value.dim() == 2, \
            ("For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
             f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")

    else:
        raise AssertionError(
            f"query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor")

    return is_batched


def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return F.linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.

    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`

        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`

    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.

    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.

    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.

        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    if attn_mask is not None:
        attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
    else:
        attn = torch.bmm(q, k.transpose(-2, -1))

    attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


def mymulti_head_attention_forward_grouppadding(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    need_weights: bool = True,
    attn_pattern: str = "insidetoken",
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True


    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """
    # print("entering mymulti_head_attention_forward()")

    is_batched = _mha_shape_check(query, key, value, num_heads)

    if not is_batched:
        raise RuntimeError("Will not allow non-batched input.")

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    source_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    # initial complete attention output
    seq = int(math.sqrt(tgt_len))
    complete_attn_output = torch.zeros([bsz, seq, seq, embed_dim], device=query.device)

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)
    # q, k, v: (seq*seq, bsz, embed_dim) batch_inside_order: lexico order. 

    # print("q", q.shape)
    # print("k", k.shape)
    # print("v", v.shape)

    if attn_pattern == "insidetoken":
        
        t = torch.arange(seq).repeat(seq).to(q.device)
        t1 = torch.broadcast_to(t[None, ...], (seq * seq, seq * seq))
        t2 = torch.broadcast_to(t[..., None], (seq * seq, seq * seq))
        h = torch.arange(seq).to(q.device)
        h = torch.broadcast_to(h[..., None], (seq, seq)).reshape(-1)
        h1 = torch.broadcast_to(h[None, ...], (seq * seq, seq * seq))
        h2 = torch.broadcast_to(h[..., None], (seq * seq, seq * seq))
        tempfilling = t - h + 1
        pattern_mask = ~((t1 <= t2) & (h1 >= h2))
        c = torch.ones(seq * seq).to(q.device)
        for i in range(0, seq * seq, seq + 1):
            c[i] = 0
        oric = torch.broadcast_to(c[None, ...], (seq * seq, seq * seq)) > 0
        pattern_mask = (oric | pattern_mask)
        pattern_mask = torch.broadcast_to(pattern_mask[None, ...], (bsz, seq * seq, seq * seq))

        # print("pattern_mask", pattern_mask)
        # print("pattern_mask_shape", pattern_mask.shape)

        q_repr = q.transpose(0, 1)
        k_repr = k.transpose(0, 1)
        v_repr = v.transpose(0, 1)

        templength = 1
        for i in range(1, seq + 1, 2):
            if i >= seq / 2:
                templength = i
                break
            
            tgt_len, bsz, hd = query.shape
            source_len, _, _ = key.shape
            seq = int(math.sqrt(tgt_len))
            lenlist = [i, i+1]
            # print("lenlist", lenlist)
            numofpattern = lenlist[-1]
            lenmask = torch.ones([bsz, seq, seq], device=q.device)
            total_span = 0
            for item in lenlist:
                total_span += (seq - item + 1)*bsz
                lenmask[:, range(seq - item + 1), range(item - 1, seq)] = 0
            lenmask = (lenmask > 0).reshape(bsz, seq * seq)
            pattern_mask_legal = pattern_mask[~lenmask].reshape(bsz, int(total_span / bsz), seq * seq)
            # print("pattern_mask_legal", pattern_mask_legal)

            temp = torch.broadcast_to(tempfilling[None, ..., None], (bsz, seq*seq, numofpattern))
            temp = temp[~lenmask].reshape(total_span, numofpattern)
            temp2 = torch.arange(start=1, end=numofpattern+1, step=1).to(q.device)
            temp2 = torch.broadcast_to(temp2[None, ...], (total_span, numofpattern))
            filling_mask = (temp-temp2) >= 0
            # print("filling_mask", filling_mask)

            # initial complete q, k, v
            complete_k = torch.zeros([total_span, numofpattern, embed_dim], device=q.device)
            complete_v = torch.zeros([total_span, numofpattern, embed_dim], device=q.device)
            complete_key_padding_mask = torch.ones([total_span, numofpattern], device=q.device) > 0

            # construct complete_q
            complete_q = q_repr[~lenmask].reshape(bsz, int(total_span/bsz), hd)
            # bsz(2), total_span/bsz, hd(3), batch_inside_order: lexico order
            complete_q = complete_q.reshape(total_span, 1, hd).transpose(0, 1)
            # tgt_len(1), total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_k
            new_k_repr = torch.broadcast_to(k_repr[:, None, ...], (bsz, int(total_span/bsz), seq*seq, hd))
            new_k_repr = new_k_repr[~pattern_mask_legal]
            complete_k[filling_mask] = new_k_repr
            complete_k = complete_k.transpose(0, 1)
            # numnofpattern, total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_v
            new_v_repr = torch.broadcast_to(v_repr[:, None, ...], (bsz, int(total_span/bsz), seq*seq, hd))
            new_v_repr = new_v_repr[~pattern_mask_legal]
            complete_v[filling_mask] = new_v_repr
            complete_v = complete_v.transpose(0, 1)
            # numnofpattern, total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_key_paddding_mask
            complete_key_padding_mask[filling_mask] = False

            # print("complete_q", complete_q)
            # print("complete_q_shape", complete_q.shape)
            # print("complete_k", complete_k)
            # print("complete_k_shape", complete_k.shape)
            # print("complete_v", complete_v)
            # print("complete_v_shape", complete_v.shape)
            # print("complete_key_padding_mask", complete_key_padding_mask)
            # print("complete_key_padding_mask_shape", complete_key_padding_mask.shape)

            ori_bsz = bsz
            tgt_len, bsz, embed_dim = complete_q.shape
            source_len, _, _ = complete_k.shape

            # prep key padding mask
            if complete_key_padding_mask is not None and complete_key_padding_mask.dtype == torch.uint8:
                warnings.warn("Byte tensor for key_padding_mask in myMultiheadAttention is deprecated. Use bool tensor instead.")
                complete_key_padding_mask = complete_key_padding_mask.to(torch.bool)

            # reshape q, k, v for multihead attention and make batch first
            complete_q = complete_q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
            complete_k = complete_k.contiguous().view(complete_k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
            complete_v = complete_v.contiguous().view(complete_v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

            # skip related content about bias_k, bias_v, add_zero_attention
            # update source sequence length after adjustments
            source_len = complete_k.size(1)

            # merge key padding and attention masks
            if complete_key_padding_mask is not None:
                assert complete_key_padding_mask.shape == (bsz, source_len), \
                    f"expecting key_padding_mask shape of {(bsz, source_len)}, but got {complete_key_padding_mask.shape}"
                complete_key_padding_mask = complete_key_padding_mask.view(bsz, 1, 1, source_len).   \
                    expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, source_len)
                attn_mask = complete_key_padding_mask

            # convert mask to float
            if attn_mask is not None and attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            # print("attn_mask", attn_mask)

            # (deep breath) calculate attention and out projection
            attn_output, attn_output_weights = _scaled_dot_product_attention(complete_q, complete_k, complete_v, attn_mask, dropout_p)
            # print("have nan in before contiguous attn_output: ", torch.isnan(new_k_repr).any())
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            # print("have nan in before linear attn_output: ", torch.isnan(new_k_repr).any())
            attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            # do reverse and fill attn_ouput to complete_attn_ouput
            attn_output = attn_output.transpose(0, 1).reshape(-1, hd)

            # print("have nan in attn_output: ", torch.isnan(attn_output).any())
            # if torch.isnan(attn_output).any():
            #     print("nan in q?", torch.isnan(complete_q).any())
            #     print("nan in k?", torch.isnan(new_k_repr).any())
            #     print("nan in v?", torch.isnan(new_v_repr).any())

            lenmask_attn = lenmask.reshape(ori_bsz, seq, seq)
            complete_attn_output[~lenmask_attn] = attn_output

            bsz = ori_bsz
        
        for i in range(templength, seq + 1, 4):
            if (i + 4) >= seq:
                lenlist = []
                for m in range(i, seq + 1):
                    lenlist.append(m)
            else:
                lenlist = [i, i + 1, i + 2, i + 3]
            # print("lenlist: ", lenlist)
            
            tgt_len, bsz, hd = query.shape
            source_len, _, _ = key.shape
            seq = int(math.sqrt(tgt_len))
            numofpattern = lenlist[-1]
            lenmask = torch.ones([bsz, seq, seq], device=q.device)
            total_span = 0
            for item in lenlist:
                total_span += (seq - item + 1)*bsz
                lenmask[:, range(seq - item + 1), range(item - 1, seq)] = 0
            lenmask = (lenmask > 0).reshape(bsz, seq * seq)
            pattern_mask_legal = pattern_mask[~lenmask].reshape(bsz, int(total_span / bsz), seq * seq)
            # print("pattern_mask_legal: ", pattern_mask_legal)
            # for b in range(bsz):
            #     for s in range(pattern_mask_legal.size()[1]):
            #         print("b, s:", b, s, pattern_mask_legal[b, s, :])

            temp = torch.broadcast_to(tempfilling[None, ..., None], (bsz, seq*seq, numofpattern))
            temp = temp[~lenmask].reshape(total_span, numofpattern)
            temp2 = torch.arange(start=1, end=numofpattern+1, step=1).to(q.device)
            temp2 = torch.broadcast_to(temp2[None, ...], (total_span, numofpattern))
            filling_mask = (temp-temp2) >= 0
            # print("filling_mask: ", filling_mask)

            # initial complete q, k, v
            complete_k = torch.zeros([total_span, numofpattern, embed_dim], device=q.device)
            complete_v = torch.zeros([total_span, numofpattern, embed_dim], device=q.device)
            complete_key_padding_mask = torch.ones([total_span, numofpattern], device=q.device) > 0

            # construct complete_q
            complete_q = q_repr[~lenmask].reshape(bsz, int(total_span/bsz), hd)
            # bsz(2), total_span/bsz, hd(3), batch_inside_order: lexico order
            complete_q = complete_q.reshape(total_span, 1, hd).transpose(0, 1)
            # tgt_len(1), total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_k
            new_k_repr = torch.broadcast_to(k_repr[:, None, ...], (bsz, int(total_span/bsz), seq*seq, hd))
            new_k_repr = new_k_repr[~pattern_mask_legal]
            complete_k[filling_mask] = new_k_repr
            complete_k = complete_k.transpose(0, 1)
            # numnofpattern, total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_v
            new_v_repr = torch.broadcast_to(v_repr[:, None, ...], (bsz, int(total_span/bsz), seq*seq, hd))
            new_v_repr = new_v_repr[~pattern_mask_legal]
            complete_v[filling_mask] = new_v_repr
            complete_v = complete_v.transpose(0, 1)
            # numnofpattern, total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_key_paddding_mask
            complete_key_padding_mask[filling_mask] = False

            # print("complete_q", complete_q)
            # print("complete_q_shape", complete_q.shape)
            # print("complete_k", complete_k)
            # print("complete_k_shape", complete_k.shape)
            # print("complete_v", complete_v)
            # print("complete_v_shape", complete_v.shape)
            # print("complete_key_padding_mask", complete_key_padding_mask)
            # print("complete_key_padding_mask_shape", complete_key_padding_mask.shape)

            ori_bsz = bsz
            tgt_len, bsz, embed_dim = complete_q.shape
            source_len, _, _ = complete_k.shape

            # prep key padding mask
            if complete_key_padding_mask is not None and complete_key_padding_mask.dtype == torch.uint8:
                warnings.warn("Byte tensor for key_padding_mask in myMultiheadAttention is deprecated. Use bool tensor instead.")
                complete_key_padding_mask = complete_key_padding_mask.to(torch.bool)

            # reshape q, k, v for multihead attention and make em batch first
            complete_q = complete_q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
            complete_k = complete_k.contiguous().view(complete_k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
            complete_v = complete_v.contiguous().view(complete_v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

            # skip related content about bias_k, bias_v, add_zero_attention
            # update source sequence length after adjustments
            source_len = complete_k.size(1)

            # merge key padding and attention masks
            if complete_key_padding_mask is not None:
                assert complete_key_padding_mask.shape == (bsz, source_len), \
                    f"expecting key_padding_mask shape of {(bsz, source_len)}, but got {complete_key_padding_mask.shape}"
                complete_key_padding_mask = complete_key_padding_mask.view(bsz, 1, 1, source_len).   \
                    expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, source_len)
                attn_mask = complete_key_padding_mask

            # convert mask to float
            if attn_mask is not None and attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            # print("attn_mask", attn_mask)

            # (deep breath) calculate attention and out projection
            attn_output, attn_output_weights = _scaled_dot_product_attention(complete_q, complete_k, complete_v, attn_mask, dropout_p)
            # print("have nan in before contiguous attn_output: ", torch.isnan(attn_output).any())
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            # print("have nan in before linear attn_output: ", torch.isnan(attn_output).any())
            attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            # do reverse and fill attn_ouput to complete_attn_ouput
            attn_output = attn_output.transpose(0, 1).reshape(-1, hd)

            # print("have nan in attn_ouput: ", torch.isnan(attn_output).any())
            # if torch.isnan(attn_output).any():
            #     print("nan in q?", torch.isnan(complete_q).any())
            #     print("nan in k?", torch.isnan(new_k_repr).any())
            #     print("nan in v?", torch.isnan(new_v_repr).any())

            lenmask_attn = lenmask.reshape(ori_bsz, seq, seq)
            complete_attn_output[~lenmask_attn] = attn_output

            bsz = ori_bsz

        complete_attn_output = complete_attn_output.reshape(bsz, seq*seq, embed_dim).transpose(0, 1)
        
        if need_weights:
            # optionally average attention weights over heads
            raise RuntimeError("Have not fill attention weights and do reverse on them.")
        else:
            if not is_batched:
                raise RuntimeError("Will not allow non-batched input.")
            return complete_attn_output, None

    elif attn_pattern == "samehandt":

        t = torch.arange(seq).repeat(seq).to(q.device)
        t1 = torch.broadcast_to(t[None, ...], (seq * seq, seq * seq))
        t2 = torch.broadcast_to(t[..., None], (seq * seq, seq * seq))
        h = torch.arange(seq).to(q.device)
        h = torch.broadcast_to(h[..., None], (seq, seq)).reshape(-1)
        h1 = torch.broadcast_to(h[None, ...], (seq * seq, seq * seq))
        h2 = torch.broadcast_to(h[..., None], (seq * seq, seq * seq))
        tempfilling = t - h + 1
        pattern_mask1 = (t1 == t2) & (h1 <= t2)
        pattern_mask2 = (h1 == h2) & (t1 >= h2)
        pattern_mask = ~(pattern_mask1 | pattern_mask2)
        pattern_mask = torch.broadcast_to(pattern_mask[None, ...], (bsz, seq * seq, seq * seq))

        # print("pattern_mask", pattern_mask)
        # print("pattern_mask_shape", pattern_mask.shape)

        q_repr = q.transpose(0, 1)
        k_repr = k.transpose(0, 1)
        v_repr = v.transpose(0, 1)

        templength = 1
        for i in range(1, seq + 1, 2):
            if i >= seq / 2:
                templength = i
                break
            
            tgt_len, bsz, hd = query.shape
            source_len, _, _ = key.shape
            seq = int(math.sqrt(tgt_len))
            lenlist = [i, i+1]
            # print("lenlist", lenlist)
            numofpattern = seq + lenlist[-1] - 1
            lenmask = torch.ones([bsz, seq, seq], device=q.device)
            total_span = 0
            for item in lenlist:
                total_span += (seq - item + 1)*bsz
                lenmask[:, range(seq - item + 1), range(item - 1, seq)] = 0
            lenmask = (lenmask > 0).reshape(bsz, seq * seq)
            pattern_mask_legal = pattern_mask[~lenmask].reshape(bsz, int(total_span / bsz), seq * seq)
            # print("pattern_mask_legal", pattern_mask_legal)

            temp = torch.broadcast_to(tempfilling[None, ..., None], (bsz, seq*seq, numofpattern))
            temp = temp[~lenmask].reshape(total_span, numofpattern)
            temp = seq + temp - 1
            temp2 = torch.arange(start=1, end=numofpattern+1, step=1).to(q.device)
            temp2 = torch.broadcast_to(temp2[None, ...], (total_span, numofpattern))
            filling_mask = (temp-temp2) >= 0
            # print("filling_mask", filling_mask)

            # initial complete q, k, v
            complete_k = torch.zeros([total_span, numofpattern, embed_dim], device=q.device)
            complete_v = torch.zeros([total_span, numofpattern, embed_dim], device=q.device).to(q.device)
            complete_key_padding_mask = torch.ones([total_span, numofpattern], device=q.device) > 0

            # construct complete_q
            complete_q = q_repr[~lenmask].reshape(bsz, int(total_span/bsz), hd)
            # bsz(2), total_span/bsz, hd(3), batch_inside_order: lexico order
            complete_q = complete_q.reshape(total_span, 1, hd).transpose(0, 1)
            # tgt_len(1), total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_k
            new_k_repr = torch.broadcast_to(k_repr[:, None, ...], (bsz, int(total_span/bsz), seq*seq, hd))
            new_k_repr = new_k_repr[~pattern_mask_legal]
            complete_k[filling_mask] = new_k_repr
            complete_k = complete_k.transpose(0, 1)
            # numnofpattern, total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_v
            new_v_repr = torch.broadcast_to(v_repr[:, None, ...], (bsz, int(total_span/bsz), seq*seq, hd))
            new_v_repr = new_v_repr[~pattern_mask_legal]
            complete_v[filling_mask] = new_v_repr
            complete_v = complete_v.transpose(0, 1)
            # numnofpattern, total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_key_paddding_mask
            complete_key_padding_mask[filling_mask] = False

            # print("complete_q", complete_q)
            # print("complete_q_shape", complete_q.shape)
            # print("complete_k", complete_k)
            # print("complete_k_shape", complete_k.shape)
            # print("complete_v", complete_v)
            # print("complete_v_shape", complete_v.shape)
            # print("complete_key_padding_mask", complete_key_padding_mask)
            # print("complete_key_padding_mask_shape", complete_key_padding_mask.shape)

            ori_bsz = bsz
            tgt_len, bsz, embed_dim = complete_q.shape
            source_len, _, _ = complete_k.shape

            # prep key padding mask
            if complete_key_padding_mask is not None and complete_key_padding_mask.dtype == torch.uint8:
                warnings.warn("Byte tensor for key_padding_mask in myMultiheadAttention is deprecated. Use bool tensor instead.")
                complete_key_padding_mask = complete_key_padding_mask.to(torch.bool)

            # reshape q, k, v for multihead attention and make em batch first
            complete_q = complete_q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
            complete_k = complete_k.contiguous().view(complete_k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
            complete_v = complete_v.contiguous().view(complete_v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

            # skip related content about bias_k, bias_v, add_zero_attention
            # update source sequence length after adjustments
            source_len = complete_k.size(1)

            # merge key padding and attention masks
            if complete_key_padding_mask is not None:
                assert complete_key_padding_mask.shape == (bsz, source_len), \
                    f"expecting key_padding_mask shape of {(bsz, source_len)}, but got {complete_key_padding_mask.shape}"
                complete_key_padding_mask = complete_key_padding_mask.view(bsz, 1, 1, source_len).   \
                    expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, source_len)
                attn_mask = complete_key_padding_mask

            # convert mask to float
            if attn_mask is not None and attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask

            # (deep breath) calculate attention and out projection
            attn_output, attn_output_weights = _scaled_dot_product_attention(complete_q, complete_k, complete_v, attn_mask, dropout_p)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            # do reverse and fill attn_ouput to complete_attn_ouput
            attn_output = attn_output.transpose(0, 1).reshape(-1, hd)
            lenmask_attn = lenmask.reshape(ori_bsz, seq, seq)
            complete_attn_output[~lenmask_attn] = attn_output

            bsz = ori_bsz
        
        for i in range(templength, seq + 1, 4):
            if (i + 4) >= seq:
                lenlist = []
                for m in range(i, seq + 1):
                    lenlist.append(m)
            else:
                lenlist = [i, i + 1, i + 2, i + 3]
            # print("lenlist", lenlist)
            
            tgt_len, bsz, hd = query.shape
            source_len, _, _ = key.shape
            seq = int(math.sqrt(tgt_len))
            numofpattern = seq + lenlist[-1] - 1
            lenmask = torch.ones([bsz, seq, seq], device=q.device)
            total_span = 0
            for item in lenlist:
                total_span += (seq - item + 1)*bsz
                lenmask[:, range(seq - item + 1), range(item - 1, seq)] = 0
            lenmask = (lenmask > 0).reshape(bsz, seq * seq)
            pattern_mask_legal = pattern_mask[~lenmask].reshape(bsz, int(total_span / bsz), seq * seq)
            # print("pattern_mask_legal", pattern_mask_legal)

            temp = torch.broadcast_to(tempfilling[None, ..., None], (bsz, seq*seq, numofpattern))
            temp = temp[~lenmask].reshape(total_span, numofpattern)
            temp = seq + temp - 1
            temp2 = torch.arange(start=1, end=numofpattern+1, step=1).to(q.device)
            temp2 = torch.broadcast_to(temp2[None, ...], (total_span, numofpattern))
            filling_mask = (temp-temp2) >= 0
            # print("filling_mask", filling_mask)

            # initial complete q, k, v
            complete_k = torch.zeros([total_span, numofpattern, embed_dim], device=q.device)
            complete_v = torch.zeros([total_span, numofpattern, embed_dim], device=q.device)
            complete_key_padding_mask = torch.ones([total_span, numofpattern], device=q.device) > 0

            # construct complete_q
            complete_q = q_repr[~lenmask].reshape(bsz, int(total_span/bsz), hd)
            # bsz(2), total_span/bsz, hd(3), batch_inside_order: lexico order
            complete_q = complete_q.reshape(total_span, 1, hd).transpose(0, 1)
            # tgt_len(1), total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_k
            new_k_repr = torch.broadcast_to(k_repr[:, None, ...], (bsz, int(total_span/bsz), seq*seq, hd))
            new_k_repr = new_k_repr[~pattern_mask_legal]
            complete_k[filling_mask] = new_k_repr
            complete_k = complete_k.transpose(0, 1)
            # numnofpattern, total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_v
            new_v_repr = torch.broadcast_to(v_repr[:, None, ...], (bsz, int(total_span/bsz), seq*seq, hd))
            new_v_repr = new_v_repr[~pattern_mask_legal]
            complete_v[filling_mask] = new_v_repr
            complete_v = complete_v.transpose(0, 1)
            # numnofpattern, total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_key_paddding_mask
            complete_key_padding_mask[filling_mask] = False

            # print("complete_q", complete_q)
            # print("complete_q_shape", complete_q.shape)
            # print("complete_k", complete_k)
            # print("complete_k_shape", complete_k.shape)
            # print("complete_v", complete_v)
            # print("complete_v_shape", complete_v.shape)
            # print("complete_key_padding_mask", complete_key_padding_mask)
            # print("complete_key_padding_mask_shape", complete_key_padding_mask.shape)

            ori_bsz = bsz
            tgt_len, bsz, embed_dim = complete_q.shape
            source_len, _, _ = complete_k.shape

            # prep key padding mask
            if complete_key_padding_mask is not None and complete_key_padding_mask.dtype == torch.uint8:
                warnings.warn("Byte tensor for key_padding_mask in myMultiheadAttention is deprecated. Use bool tensor instead.")
                complete_key_padding_mask = complete_key_padding_mask.to(torch.bool)

            # reshape q, k, v for multihead attention and make em batch first
            complete_q = complete_q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
            complete_k = complete_k.contiguous().view(complete_k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
            complete_v = complete_v.contiguous().view(complete_v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

            # skip related content about bias_k, bias_v, add_zero_attention
            # update source sequence length after adjustments
            source_len = complete_k.size(1)

            # merge key padding and attention masks
            if complete_key_padding_mask is not None:
                assert complete_key_padding_mask.shape == (bsz, source_len), \
                    f"expecting key_padding_mask shape of {(bsz, source_len)}, but got {complete_key_padding_mask.shape}"
                complete_key_padding_mask = complete_key_padding_mask.view(bsz, 1, 1, source_len).   \
                    expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, source_len)
                attn_mask = complete_key_padding_mask

            # convert mask to float
            if attn_mask is not None and attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            # print("attn_mask", attn_mask)

            # (deep breath) calculate attention and out projection
            attn_output, attn_output_weights = _scaled_dot_product_attention(complete_q, complete_k, complete_v, attn_mask, dropout_p)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            # do reverse and fill attn_ouput to complete_attn_ouput
            attn_output = attn_output.transpose(0, 1).reshape(-1, hd)
            lenmask_attn = lenmask.reshape(ori_bsz, seq, seq)
            complete_attn_output[~lenmask_attn] = attn_output

            bsz = ori_bsz

        complete_attn_output = complete_attn_output.reshape(bsz, seq*seq, embed_dim).transpose(0, 1)
        
        if need_weights:
            # optionally average attention weights over heads
            raise RuntimeError("Have not fill attention weights and do reverse on them.")
        else:
            if not is_batched:
                # squeeze the output if input was unbatched
                raise RuntimeError("Will not allow non-batched input.")
            return complete_attn_output, None

    elif attn_pattern == "subspan":

        t = torch.arange(seq).repeat(seq).to(q.device)
        t1 = torch.broadcast_to(t[None, ...], (seq * seq, seq * seq))
        t2 = torch.broadcast_to(t[..., None], (seq * seq, seq * seq))
        h = torch.arange(seq).to(q.device)
        h = torch.broadcast_to(h[..., None], (seq, seq)).reshape(-1)
        h1 = torch.broadcast_to(h[None, ...], (seq * seq, seq * seq))
        h2 = torch.broadcast_to(h[..., None], (seq * seq, seq * seq))
        tempfilling = t - h + 1
        tempfilling = (tempfilling+1)*tempfilling / 2
        pattern_mask = ~((t1 <= t2) & (h1 >= h2) & (h1 <= t1))
        pattern_mask = torch.broadcast_to(pattern_mask[None, ...], (bsz, seq * seq, seq * seq))

        # print("pattern_mask", pattern_mask)
        # print("pattern_mask_shape", pattern_mask.shape)

        q_repr = q.transpose(0, 1)
        k_repr = k.transpose(0, 1)
        v_repr = v.transpose(0, 1)

        templength = 1
        for i in range(1, seq + 1, 4):
            if i >= seq / 2:
                templength = i
                break
            
            tgt_len, bsz, hd = query.shape
            source_len, _, _ = key.shape
            seq = int(math.sqrt(tgt_len))
            lenlist = [i, i+1, i+2, i+3]
            # print(lenlist)
            maxlen = lenlist[-1]
            numofpattern = int((1+maxlen)*maxlen / 2)
            lenmask = torch.ones([bsz, seq, seq], device=q.device)
            total_span = 0
            for item in lenlist:
                total_span += (seq - item + 1)*bsz
                lenmask[:, range(seq - item + 1), range(item - 1, seq)] = 0
            lenmask = (lenmask > 0).reshape(bsz, seq * seq)
            pattern_mask_legal = pattern_mask[~lenmask].reshape(bsz, int(total_span / bsz), seq * seq)
            # print("pattern_mask_legal", pattern_mask_legal)

            temp = torch.broadcast_to(tempfilling[None, ..., None], (bsz, seq*seq, numofpattern))
            temp = temp[~lenmask].reshape(total_span, numofpattern)
            temp2 = torch.arange(start=1, end=numofpattern+1, step=1).to(q.device)
            temp2 = torch.broadcast_to(temp2[None, ...], (total_span, numofpattern))
            filling_mask = (temp-temp2) >= 0
            # print("filling_mask", filling_mask)

            # initial complete q, k, v
            complete_k = torch.zeros([total_span, numofpattern, embed_dim], device=q.device)
            complete_v = torch.zeros([total_span, numofpattern, embed_dim], device=q.device)
            complete_key_padding_mask = torch.ones([total_span, numofpattern], device=q.device) > 0
            
            # construct complete_q
            complete_q = q_repr[~lenmask].reshape(bsz, int(total_span/bsz), hd)
            # bsz(2), total_span/bsz, hd(3), batch_inside_order: lexico order
            complete_q = complete_q.reshape(total_span, 1, hd).transpose(0, 1)
            # tgt_len(1), total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_k
            new_k_repr = torch.broadcast_to(k_repr[:, None, ...], (bsz, int(total_span/bsz), seq*seq, hd))
            new_k_repr = new_k_repr[~pattern_mask_legal]
            complete_k[filling_mask] = new_k_repr
            complete_k = complete_k.transpose(0, 1)
            # numnofpattern, total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_v
            new_v_repr = torch.broadcast_to(v_repr[:, None, ...], (bsz, int(total_span/bsz), seq*seq, hd))
            new_v_repr = new_v_repr[~pattern_mask_legal]
            complete_v[filling_mask] = new_v_repr
            complete_v = complete_v.transpose(0, 1)
            # numnofpattern, total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_key_paddding_mask
            complete_key_padding_mask[filling_mask] = False

            # print("complete_q", complete_q)
            # print("complete_q_shape", complete_q.shape)
            # print("complete_k", complete_k)
            # print("complete_k_shape", complete_k.shape)
            # print("complete_v", complete_v)
            # print("complete_v_shape", complete_v.shape)
            # print("complete_key_padding_mask", complete_key_padding_mask)
            # print("complete_key_padding_mask_shape", complete_key_padding_mask.shape)

            ori_bsz = bsz
            tgt_len, bsz, embed_dim = complete_q.shape
            source_len, _, _ = complete_k.shape

            # prep key padding mask
            if complete_key_padding_mask is not None and complete_key_padding_mask.dtype == torch.uint8:
                warnings.warn("Byte tensor for key_padding_mask in myMultiheadAttention is deprecated. Use bool tensor instead.")
                complete_key_padding_mask = complete_key_padding_mask.to(torch.bool)

            # reshape q, k, v for multihead attention and make em batch first
            complete_q = complete_q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
            complete_k = complete_k.contiguous().view(complete_k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
            complete_v = complete_v.contiguous().view(complete_v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

            # skip related content about bias_k, bias_v, add_zero_attention
            # update source sequence length after adjustments
            source_len = complete_k.size(1)

            # merge key padding and attention masks
            if complete_key_padding_mask is not None:
                assert complete_key_padding_mask.shape == (bsz, source_len), \
                    f"expecting key_padding_mask shape of {(bsz, source_len)}, but got {complete_key_padding_mask.shape}"
                complete_key_padding_mask = complete_key_padding_mask.view(bsz, 1, 1, source_len).   \
                    expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, source_len)
                attn_mask = complete_key_padding_mask

            # convert mask to float
            if attn_mask is not None and attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            # print("attn_mask", attn_mask)

            # (deep breath) calculate attention and out projection
            attn_output, attn_output_weights = _scaled_dot_product_attention(complete_q, complete_k, complete_v, attn_mask, dropout_p)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            # do reverse and fill attn_ouput to complete_attn_ouput
            attn_output = attn_output.transpose(0, 1).reshape(-1, hd)
            lenmask_attn = lenmask.reshape(ori_bsz, seq, seq)
            complete_attn_output[~lenmask_attn] = attn_output

            bsz = ori_bsz
        
        for i in range(templength, seq + 1, 2):
            if (i + 2) >= seq:
                lenlist = []
                for m in range(i, seq + 1):
                    lenlist.append(m)
            else:
                lenlist = [i, i + 1]
            
            # print("lenlist", lenlist)
            tgt_len, bsz, hd = query.shape
            source_len, _, _ = key.shape
            seq = int(math.sqrt(tgt_len))
            maxlen = lenlist[-1]
            numofpattern = int((1+maxlen)*maxlen / 2)
            lenmask = torch.ones([bsz, seq, seq], device=q.device)
            total_span = 0
            for item in lenlist:
                total_span += (seq - item + 1)*bsz
                lenmask[:, range(seq - item + 1), range(item - 1, seq)] = 0
            lenmask = (lenmask > 0).reshape(bsz, seq * seq)
            pattern_mask_legal = pattern_mask[~lenmask].reshape(bsz, int(total_span / bsz), seq * seq)
            # print("pattern_mask_legal", pattern_mask_legal)

            temp = torch.broadcast_to(tempfilling[None, ..., None], (bsz, seq*seq, numofpattern))
            temp = temp[~lenmask].reshape(total_span, numofpattern)
            temp2 = torch.arange(start=1, end=numofpattern+1, step=1).to(q.device)
            temp2 = torch.broadcast_to(temp2[None, ...], (total_span, numofpattern))
            filling_mask = (temp-temp2) >= 0
            # print("filling_mask", filling_mask)

            # initial complete q, k, v
            complete_k = torch.zeros([total_span, numofpattern, embed_dim], device=q.device)
            complete_v = torch.zeros([total_span, numofpattern, embed_dim], device=q.device)
            complete_key_padding_mask = torch.ones([total_span, numofpattern], device=q.device) > 0
            
            # construct complete_q
            complete_q = q_repr[~lenmask].reshape(bsz, int(total_span/bsz), hd)
            # bsz(2), total_span/bsz, hd(3), batch_inside_order: lexico order
            complete_q = complete_q.reshape(total_span, 1, hd).transpose(0, 1)
            # tgt_len(1), total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_k
            new_k_repr = torch.broadcast_to(k_repr[:, None, ...], (bsz, int(total_span/bsz), seq*seq, hd))
            new_k_repr = new_k_repr[~pattern_mask_legal]
            complete_k[filling_mask] = new_k_repr
            complete_k = complete_k.transpose(0, 1)
            # numnofpattern, total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_v
            new_v_repr = torch.broadcast_to(v_repr[:, None, ...], (bsz, int(total_span/bsz), seq*seq, hd))
            new_v_repr = new_v_repr[~pattern_mask_legal]
            complete_v[filling_mask] = new_v_repr
            complete_v = complete_v.transpose(0, 1)
            # numnofpattern, total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_key_paddding_mask
            complete_key_padding_mask[filling_mask] = False

            # print("complete_q", complete_q)
            # print("complete_q_shape", complete_q.shape)
            # print("complete_k", complete_k)
            # print("complete_k_shape", complete_k.shape)
            # print("complete_v", complete_v)
            # print("complete_v_shape", complete_v.shape)
            # print("complete_key_padding_mask", complete_key_padding_mask)
            # print("complete_key_padding_mask_shape", complete_key_padding_mask.shape)

            ori_bsz = bsz
            tgt_len, bsz, embed_dim = complete_q.shape
            source_len, _, _ = complete_k.shape

            # prep key padding mask
            if complete_key_padding_mask is not None and complete_key_padding_mask.dtype == torch.uint8:
                warnings.warn("Byte tensor for key_padding_mask in myMultiheadAttention is deprecated. Use bool tensor instead.")
                complete_key_padding_mask = complete_key_padding_mask.to(torch.bool)

            # reshape q, k, v for multihead attention and make em batch first
            complete_q = complete_q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
            complete_k = complete_k.contiguous().view(complete_k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
            complete_v = complete_v.contiguous().view(complete_v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

            # skip related content about bias_k, bias_v, add_zero_attention
            # update source sequence length after adjustments
            source_len = complete_k.size(1)

            # merge key padding and attention masks
            if complete_key_padding_mask is not None:
                assert complete_key_padding_mask.shape == (bsz, source_len), \
                    f"expecting key_padding_mask shape of {(bsz, source_len)}, but got {complete_key_padding_mask.shape}"
                complete_key_padding_mask = complete_key_padding_mask.view(bsz, 1, 1, source_len).   \
                    expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, source_len)
                attn_mask = complete_key_padding_mask

            # convert mask to float
            if attn_mask is not None and attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            # print("attn_mask", attn_mask)

            # (deep breath) calculate attention and out projection
            attn_output, attn_output_weights = _scaled_dot_product_attention(complete_q, complete_k, complete_v, attn_mask, dropout_p)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            # do reverse and fill attn_ouput to complete_attn_ouput
            attn_output = attn_output.transpose(0, 1).reshape(-1, hd)
            lenmask_attn = lenmask.reshape(ori_bsz, seq, seq)
            complete_attn_output[~lenmask_attn] = attn_output

            bsz = ori_bsz

        complete_attn_output = complete_attn_output.reshape(bsz, seq*seq, embed_dim).transpose(0, 1)
        
        if need_weights:
            # optionally average attention weights over heads
            raise RuntimeError("Have not fill attention weights and do reverse on them.")
        else:
            if not is_batched:
                # squeeze the output if input was unbatched
                raise RuntimeError("Will not allow non-batched input.")
            return complete_attn_output, None

    elif attn_pattern == "sibling":

        t = torch.arange(seq).repeat(seq).to(q.device)
        t1 = torch.broadcast_to(t[None, ...], (seq * seq, seq * seq))
        t2 = torch.broadcast_to(t[..., None], (seq * seq, seq * seq))
        h = torch.arange(seq).to(q.device)
        h = torch.broadcast_to(h[..., None], (seq, seq)).reshape(-1)
        h1 = torch.broadcast_to(h[None, ...], (seq * seq, seq * seq))
        h2 = torch.broadcast_to(h[..., None], (seq * seq, seq * seq))
        tempfilling = t - h + 1
        tempfilling = seq - tempfilling
        pattern_mask = ~(((h1 == t2 + 1) | (t1 == h2 - 1)) & (h1 <= t1))
        pattern_mask = torch.broadcast_to(pattern_mask[None, ...], (bsz, seq * seq, seq * seq))
        # print("pattern_mask", pattern_mask)

        q_repr = q.transpose(0, 1)
        k_repr = k.transpose(0, 1)
        v_repr = v.transpose(0, 1)

        templength = 1
        for i in range(1, seq + 1, 2):
            if i >= seq / 2:
                templength = i
                break
            
            tgt_len, bsz, hd = query.shape
            source_len, _, _ = key.shape
            seq = int(math.sqrt(tgt_len))
            lenlist = [i, i+1]
            # print("lenlist", lenlist)
            numofpattern = seq - lenlist[0]
            lenmask = torch.ones([bsz, seq, seq], device=q.device)
            total_span = 0
            for item in lenlist:
                total_span += (seq - item + 1)*bsz
                lenmask[:, range(seq - item + 1), range(item - 1, seq)] = 0
            lenmask = (lenmask > 0).reshape(bsz, seq * seq)
            pattern_mask_legal = pattern_mask[~lenmask].reshape(bsz, int(total_span / bsz), seq * seq)
            # print("pattern_mask_legal", pattern_mask_legal)

            temp = torch.broadcast_to(tempfilling[None, ..., None], (bsz, seq*seq, numofpattern))
            temp = temp[~lenmask].reshape(total_span, numofpattern)
            temp2 = torch.arange(start=1, end=numofpattern+1, step=1).to(q.device)
            temp2 = torch.broadcast_to(temp2[None, ...], (total_span, numofpattern))
            filling_mask = (temp-temp2) >= 0
            # print("filling_mask", filling_mask)

            # initial complete q, k, v
            complete_k = torch.zeros([total_span, numofpattern, embed_dim], device=q.device)
            complete_v = torch.zeros([total_span, numofpattern, embed_dim], device=q.device)
            complete_key_padding_mask = torch.ones([total_span, numofpattern], device=q.device) > 0
            
            # construct complete_q
            complete_q = q_repr[~lenmask].reshape(bsz, int(total_span/bsz), hd)
            # bsz(2), total_span/bsz, hd(3), batch_inside_order: lexico order
            complete_q = complete_q.reshape(total_span, 1, hd).transpose(0, 1)
            # tgt_len(1), total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_k
            new_k_repr = torch.broadcast_to(k_repr[:, None, ...], (bsz, int(total_span/bsz), seq*seq, hd))
            new_k_repr = new_k_repr[~pattern_mask_legal]
            complete_k[filling_mask] = new_k_repr
            complete_k = complete_k.transpose(0, 1)
            # numnofpattern, total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_v
            new_v_repr = torch.broadcast_to(v_repr[:, None, ...], (bsz, int(total_span/bsz), seq*seq, hd))
            new_v_repr = new_v_repr[~pattern_mask_legal]
            complete_v[filling_mask] = new_v_repr
            complete_v = complete_v.transpose(0, 1)
            # numnofpattern, total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_key_paddding_mask
            complete_key_padding_mask[filling_mask] = False

            # print("complete_q", complete_q)
            # print("complete_q_shape", complete_q.shape)
            # print("complete_k", complete_k)
            # print("complete_k_shape", complete_k.shape)
            # print("complete_v", complete_v)
            # print("complete_v_shape", complete_v.shape)
            # print("complete_key_padding_mask", complete_key_padding_mask)
            # print("complete_key_padding_mask_shape", complete_key_padding_mask.shape)

            ori_bsz = bsz
            tgt_len, bsz, embed_dim = complete_q.shape
            source_len, _, _ = complete_k.shape

            # prep key padding mask
            if complete_key_padding_mask is not None and complete_key_padding_mask.dtype == torch.uint8:
                warnings.warn("Byte tensor for key_padding_mask in myMultiheadAttention is deprecated. Use bool tensor instead.")
                complete_key_padding_mask = complete_key_padding_mask.to(torch.bool)

            # reshape q, k, v for multihead attention and make em batch first
            complete_q = complete_q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
            complete_k = complete_k.contiguous().view(complete_k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
            complete_v = complete_v.contiguous().view(complete_v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

            # skip related content about bias_k, bias_v, add_zero_attention
            # update source sequence length after adjustments
            source_len = complete_k.size(1)

            # merge key padding and attention masks
            if complete_key_padding_mask is not None:
                assert complete_key_padding_mask.shape == (bsz, source_len), \
                    f"expecting key_padding_mask shape of {(bsz, source_len)}, but got {complete_key_padding_mask.shape}"
                complete_key_padding_mask = complete_key_padding_mask.view(bsz, 1, 1, source_len).   \
                    expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, source_len)
                attn_mask = complete_key_padding_mask

            # convert mask to float
            if attn_mask is not None and attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            # print("attn_mask", attn_mask)

            # (deep breath) calculate attention and out projection
            attn_output, attn_output_weights = _scaled_dot_product_attention(complete_q, complete_k, complete_v, attn_mask, dropout_p)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            # do reverse and fill attn_ouput to complete_attn_ouput
            attn_output = attn_output.transpose(0, 1).reshape(-1, hd)
            lenmask_attn = lenmask.reshape(ori_bsz, seq, seq)
            complete_attn_output[~lenmask_attn] = attn_output

            bsz = ori_bsz
        
        for i in range(templength, seq, 4):
            if (i + 4) >= seq:
                lenlist = []
                for m in range(i, seq):
                    lenlist.append(m)
            else:
                lenlist = [i, i + 1, i + 2, i + 3]
            
            # print("lenlist", lenlist)
            tgt_len, bsz, hd = query.shape
            source_len, _, _ = key.shape
            seq = int(math.sqrt(tgt_len))
            numofpattern = seq - lenlist[0]
            lenmask = torch.ones([bsz, seq, seq], device=q.device)
            total_span = 0
            for item in lenlist:
                total_span += (seq - item + 1)*bsz
                lenmask[:, range(seq - item + 1), range(item - 1, seq)] = 0
            lenmask = (lenmask > 0).reshape(bsz, seq * seq)
            pattern_mask_legal = pattern_mask[~lenmask].reshape(bsz, int(total_span / bsz), seq * seq)
            # print("pattern_mask_legal", pattern_mask_legal)

            temp = torch.broadcast_to(tempfilling[None, ..., None], (bsz, seq*seq, numofpattern))
            temp = temp[~lenmask].reshape(total_span, numofpattern)
            temp2 = torch.arange(start=1, end=numofpattern+1, step=1).to(q.device)
            temp2 = torch.broadcast_to(temp2[None, ...], (total_span, numofpattern))
            filling_mask = (temp-temp2) >= 0
            # print("filling_mask", filling_mask)

            # initial complete q, k, v
            complete_k = torch.zeros([total_span, numofpattern, embed_dim], device=q.device)
            complete_v = torch.zeros([total_span, numofpattern, embed_dim], device=q.device)
            complete_key_padding_mask = torch.ones([total_span, numofpattern], device=q.device) > 0

            # construct complete_q
            complete_q = q_repr[~lenmask].reshape(bsz, int(total_span/bsz), hd)
            # bsz(2), total_span/bsz, hd(3), batch_inside_order: lexico order
            complete_q = complete_q.reshape(total_span, 1, hd).transpose(0, 1)
            # tgt_len(1), total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_k
            new_k_repr = torch.broadcast_to(k_repr[:, None, ...], (bsz, int(total_span/bsz), seq*seq, hd))
            new_k_repr = new_k_repr[~pattern_mask_legal]
            complete_k[filling_mask] = new_k_repr
            complete_k = complete_k.transpose(0, 1)
            # numnofpattern, total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_v
            new_v_repr = torch.broadcast_to(v_repr[:, None, ...], (bsz, int(total_span/bsz), seq*seq, hd))
            new_v_repr = new_v_repr[~pattern_mask_legal]
            complete_v[filling_mask] = new_v_repr
            complete_v = complete_v.transpose(0, 1)
            # numnofpattern, total_span, hd(3), total_span_inside_order: batch first, then lexico order

            # construct complete_key_paddding_mask
            complete_key_padding_mask[filling_mask] = False

            # print("complete_q", complete_q)
            # print("complete_q_shape", complete_q.shape)
            # print("complete_k", complete_k)
            # print("complete_k_shape", complete_k.shape)
            # print("complete_v", complete_v)
            # print("complete_v_shape", complete_v.shape)
            # print("complete_key_padding_mask", complete_key_padding_mask)
            # print("complete_key_padding_mask_shape", complete_key_padding_mask.shape)

            ori_bsz = bsz
            tgt_len, bsz, embed_dim = complete_q.shape
            source_len, _, _ = complete_k.shape

            # prep key padding mask
            if complete_key_padding_mask is not None and complete_key_padding_mask.dtype == torch.uint8:
                warnings.warn("Byte tensor for key_padding_mask in myMultiheadAttention is deprecated. Use bool tensor instead.")
                complete_key_padding_mask = complete_key_padding_mask.to(torch.bool)

            # reshape q, k, v for multihead attention and make em batch first
            complete_q = complete_q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
            complete_k = complete_k.contiguous().view(complete_k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
            complete_v = complete_v.contiguous().view(complete_v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

            # skip related content about bias_k, bias_v, add_zero_attention
            # update source sequence length after adjustments
            source_len = complete_k.size(1)

            # merge key padding and attention masks
            if complete_key_padding_mask is not None:
                assert complete_key_padding_mask.shape == (bsz, source_len), \
                    f"expecting key_padding_mask shape of {(bsz, source_len)}, but got {complete_key_padding_mask.shape}"
                complete_key_padding_mask = complete_key_padding_mask.view(bsz, 1, 1, source_len).   \
                    expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, source_len)
                attn_mask = complete_key_padding_mask

            # convert mask to float
            if attn_mask is not None and attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            # print("attn_mask", attn_mask)

            # (deep breath) calculate attention and out projection
            attn_output, attn_output_weights = _scaled_dot_product_attention(complete_q, complete_k, complete_v, attn_mask, dropout_p)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            # do reverse and fill attn_ouput to complete_attn_ouput
            attn_output = attn_output.transpose(0, 1).reshape(-1, hd)
            lenmask_attn = lenmask.reshape(ori_bsz, seq, seq)
            complete_attn_output[~lenmask_attn] = attn_output

            bsz = ori_bsz

        lenlist = [seq]
        # print("lenlist", lenlist)
        tgt_len, bsz, hd = query.shape
        source_len, _, _ = key.shape
        seq = int(math.sqrt(tgt_len))
        numofpattern = seq*seq
        lenmask = torch.ones([bsz, seq, seq], device=q.device)
        total_span = 0
        for item in lenlist:
            total_span += (seq - item + 1)*bsz
            lenmask[:, range(seq - item + 1), range(item - 1, seq)] = 0
        lenmask = (lenmask > 0).reshape(bsz, seq * seq)
        # pattern_mask_legal = pattern_mask[~lenmask].reshape(bsz, int(total_span / bsz), seq * seq)
        pattern_mask_legal = torch.zeros((bsz, int(total_span / bsz), seq*seq), device=pattern_mask.device) > 0
        # print("pattern_mask_legal", pattern_mask_legal)

        filling_mask = torch.zeros([total_span, numofpattern], device=tempfilling.device) == 0
        # print("filling_mask", filling_mask)
        # all true in filling_mask

        # initial complete q, k, v
        complete_k = torch.zeros([total_span, numofpattern, embed_dim], device=q.device)
        complete_v = torch.zeros([total_span, numofpattern, embed_dim], device=q.device)
        complete_key_padding_mask = torch.ones([total_span, numofpattern], device=q.device) > 0

        # construct complete_q
        complete_q = q_repr[~lenmask].reshape(bsz, int(total_span/bsz), hd)
        # bsz(2), total_span/bsz, hd(3), batch_inside_order: lexico order
        complete_q = complete_q.reshape(total_span, 1, hd).transpose(0, 1)
        # tgt_len(1), total_span, hd(3), total_span_inside_order: batch first, then lexico order

        # construct complete_k
        new_k_repr = torch.broadcast_to(k_repr[:, None, ...], (bsz, int(total_span/bsz), seq*seq, hd))
        new_k_repr = new_k_repr[~pattern_mask_legal]
        complete_k[filling_mask] = new_k_repr
        complete_k = complete_k.transpose(0, 1)
        # numnofpattern, total_span, hd(3), total_span_inside_order: batch first, then lexico order

        # construct complete_v
        new_v_repr = torch.broadcast_to(v_repr[:, None, ...], (bsz, int(total_span/bsz), seq*seq, hd))
        new_v_repr = new_v_repr[~pattern_mask_legal]
        complete_v[filling_mask] = new_v_repr
        complete_v = complete_v.transpose(0, 1)
        # numnofpattern, total_span, hd(3), total_span_inside_order: batch first, then lexico order

        # construct complete_key_paddding_mask
        complete_key_padding_mask[filling_mask] = False

        # print("complete_q", complete_q)
        # print("complete_q_shape", complete_q.shape)
        # print("complete_k", complete_k)
        # print("complete_k_shape", complete_k.shape)
        # print("complete_v", complete_v)
        # print("complete_v_shape", complete_v.shape)
        # print("complete_key_padding_mask", complete_key_padding_mask)
        # print("complete_key_padding_mask_shape", complete_key_padding_mask.shape)

        ori_bsz = bsz
        tgt_len, bsz, embed_dim = complete_q.shape
        source_len, _, _ = complete_k.shape

        # prep key padding mask
        if complete_key_padding_mask is not None and complete_key_padding_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for key_padding_mask in myMultiheadAttention is deprecated. Use bool tensor instead.")
            complete_key_padding_mask = complete_key_padding_mask.to(torch.bool)

        # reshape q, k, v for multihead attention and make em batch first
        complete_q = complete_q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        complete_k = complete_k.contiguous().view(complete_k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        complete_v = complete_v.contiguous().view(complete_v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

        # skip related content about bias_k, bias_v, add_zero_attention
        # update source sequence length after adjustments
        source_len = complete_k.size(1)

        # merge key padding and attention masks
        if complete_key_padding_mask is not None:
            assert complete_key_padding_mask.shape == (bsz, source_len), \
                f"expecting key_padding_mask shape of {(bsz, source_len)}, but got {complete_key_padding_mask.shape}"
            complete_key_padding_mask = complete_key_padding_mask.view(bsz, 1, 1, source_len).   \
                expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, source_len)
            attn_mask = complete_key_padding_mask

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
        # print("attn_mask", attn_mask)

        # (deep breath) calculate attention and out projection
        attn_output, attn_output_weights = _scaled_dot_product_attention(complete_q, complete_k, complete_v, attn_mask, dropout_p)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        # do reverse and fill attn_ouput to complete_attn_ouput
        attn_output = attn_output.transpose(0, 1).reshape(-1, hd)
        lenmask_attn = lenmask.reshape(ori_bsz, seq, seq)
        complete_attn_output[~lenmask_attn] = attn_output
        
        bsz = ori_bsz

        complete_attn_output = complete_attn_output.reshape(bsz, seq*seq, embed_dim).transpose(0, 1)
        
        if need_weights:
            # optionally average attention weights over heads
            raise RuntimeError("Have not fill attention weights and do reverse on them.")
        else:
            if not is_batched:
                # squeeze the output if input was unbatched
                raise RuntimeError("Will not allow non-batched input.")
            return complete_attn_output, None

    else:
        raise RuntimeError("attention pattern not defined.")


class myMultiheadAttention(Module):
    r"""Allows the model to jointly attend to information

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> multihead_attn = myMultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    """
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(myMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(myMultiheadAttention, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                attn_pattern: str = "insidetoken", need_weights: bool = True, 
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        attn_pattern: the attention pattern (optional, default: "insidetoken"--attend to inside token).
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
          :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
          where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
          embedding dimension ``embed_dim``.
        - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
          returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_weights=False``, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

        .. note::
            `batch_first` argument is ignored for unbatched inputs.
        
        """
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        
        # print("query: ", query)
        # print("key: ", key)
        # print("value: ", value)
        # print("query_size: ", query.size())
        # print("key_size: ", key.size())
        # print("value_size: ", value.size())

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = mymulti_head_attention_forward_grouppadding(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                need_weights=need_weights,
                attn_pattern=attn_pattern, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights)
        else:
            attn_output, attn_output_weights = mymulti_head_attention_forward_grouppadding(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                need_weights=need_weights,
                attn_pattern=attn_pattern, average_attn_weights=average_attn_weights)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class myTransformerEncoderLayer(Module):
    r"""myTransformerEncoderLayer is made up of self-attn and feedforward network.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=1024).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = myTransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = myTransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 1024, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(myTransformerEncoderLayer, self).__init__()
        self.self_attn = myMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        self.activation = activation

    def __setstate__(self, state):
        super(myTransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(self, src: Tensor, attn_pattern: str = "insidetoken") -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder (required).
            attn_pattern: the attention pattern (optional, default: "insidetoken"--attend to inside token).
        
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_pattern)
            x = x + self._ff_block(self.norm2(x))
        else:
            # print("orig_x: ", x)
            x = self.norm1(x + self._sa_block(x, attn_pattern))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_pattern: str = "insidetoken") -> Tensor:
        # print("enter _sa_block()")
        # print("x: ", x)
        x = self.self_attn(x, x, x,
                           attn_pattern=attn_pattern,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class myTransformerEncoder(Module):
    r"""myTransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the myTransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = myTransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = myTransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(myTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, attn_pattern: str = "insidetoken") -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            attn_pattern: the attention pattern (optional, default: "insidetoken"--attend to inside token).
        
        """
        output = src

        for mod in self.layers:
            output = mod(output, attn_pattern=attn_pattern)

        if self.norm is not None:
            output = self.norm(output)

        return output

if __name__ == "__main__":
    # test
    # src = torch.arange(72).reshape(9,2,4).float()
    src = torch.arange(392).reshape(49,2,4).float()
    print("self-defined_x: ", src)
    encoder_layer = myTransformerEncoderLayer(d_model=4, nhead=4)
    transformer_encoder = myTransformerEncoder(encoder_layer, num_layers=1)
    out = transformer_encoder(src, attn_pattern="sibling")
    print("out: ", out)