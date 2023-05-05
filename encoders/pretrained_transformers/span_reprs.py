"""Different batched non-parametric span representations."""
import sys
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from encoders.pretrained_transformers.utils import get_span_mask
from encoders.pretrained_transformers.new_transformer import myTransformerEncoderLayer, myTransformerEncoder


class SpanRepr(ABC, nn.Module):
    """Abstract class describing span representation."""

    def __init__(self, input_dim, use_proj=False, proj_dim=256):
        super(SpanRepr, self).__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.use_proj = use_proj
        if use_proj:
            self.proj = nn.Linear(input_dim, proj_dim)

    @abstractmethod
    def forward(self, flag, encoded_input, start_ids_1, end_ids_1, query_batch_idx, start_ids_2, end_ids_2):
        raise NotImplementedError

    def get_input_dim(self):
        return self.input_dim

    @abstractmethod
    def get_output_dim(self):
        raise NotImplementedError


class ComplexSpanRepr(ABC, nn.Module):
    """Abstract class describing span representation."""

    def __init__(self, method, input_dim, use_proj=False, proj_dim=256, attn_schema=['none'], nhead=2, nlayer=2):
        super(ComplexSpanRepr, self).__init__()
        self.method = method
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.use_proj = use_proj
        if use_proj:
            self.proj = nn.Linear(input_dim, proj_dim)
        self.attn_schema = attn_schema
        self.nhead = nhead
        self.nlayer = nlayer

    @abstractmethod
    def forward(self, flag, encoded_input, start_ids_1, end_ids_1, query_batch_idx, start_ids_2, end_ids_2):
        raise NotImplementedError

    def get_input_dim(self):
        return self.input_dim

    @abstractmethod
    def get_output_dim(self):
        raise NotImplementedError


class MeanSpanRepr(SpanRepr, nn.Module):
    """Class implementing the mean span representation."""

    def forward(self, flag, encoded_input, start_ids_1, end_ids_1, query_batch_idx, start_ids_2, end_ids_2):
        if self.use_proj:
            encoded_input = self.proj(encoded_input)

        tmp_encoded_input = encoded_input
        bsz, seq, hd = encoded_input.size()
        span_repr = torch.zeros([bsz, seq, seq, hd], device=encoded_input.device)
        for i in range(seq):
            tmp_encoded_input = ((tmp_encoded_input[:, 0:seq - i, :] * i + encoded_input[:, i:, :]) / (i + 1)).float()
            span_repr[:, range(seq - i), range(i, seq), :] = tmp_encoded_input
        
        if start_ids_2 == None:
            res = span_repr[query_batch_idx, start_ids_1, end_ids_1, :]
            return res, None
        else:
            res1 = span_repr[query_batch_idx, start_ids_1, end_ids_1, :]
            res2 = span_repr[query_batch_idx, start_ids_2, end_ids_2, :]
            return res1, res2

    def get_output_dim(self):
        if self.use_proj:
            return self.proj_dim
        else:
            return self.input_dim


class EndPointRepr(SpanRepr, nn.Module):
    """Class implementing the diff span representation - [h_j; h_i]"""

    def forward(self, flag, encoded_input, start_ids_1, end_ids_1, query_batch_idx, start_ids_2, end_ids_2):
        if self.use_proj:
            encoded_input = self.proj(encoded_input)

        bsz, seq, hd = encoded_input.size()
        hd = hd*2
        span_repr = torch.zeros([bsz, seq, seq, hd], device=encoded_input.device)
        for i in range(seq):
            span_repr[:, range(seq - i), range(i, seq), :] = torch.cat((encoded_input[:, 0:seq-i, :], encoded_input[:, i:, :]), dim=2).float()
        
        if start_ids_2 == None:
            res = span_repr[query_batch_idx, start_ids_1, end_ids_1, :]
            return res, None
        else:
            res1 = span_repr[query_batch_idx, start_ids_1, end_ids_1, :]
            res2 = span_repr[query_batch_idx, start_ids_2, end_ids_2, :]
            return res1, res2

    def get_output_dim(self):
        if self.use_proj:
            return 2 * self.proj_dim
        else:
            return 2 * self.input_dim


class DiffSumSpanRepr(SpanRepr, nn.Module):
    """Class implementing the diff_sum span representation - [h_j - h_i; h_j + h_i]"""

    def forward(self, flag, encoded_input, start_ids_1, end_ids_1, query_batch_idx, start_ids_2, end_ids_2):
        if self.use_proj:
            encoded_input = self.proj(encoded_input)
        
        bsz, seq, hd = encoded_input.size()
        hd = hd*2
        span_repr = torch.zeros([bsz, seq, seq, hd], device=encoded_input.device)
        for i in range(seq):
            span_repr[:, range(seq - i), range(i, seq), :] = torch.cat((encoded_input[:, i:, :]-encoded_input[:, 0:seq-i, :], encoded_input[:, i:, :]+encoded_input[:, 0:seq-i, :]), dim=2).float()
        
        if start_ids_2 == None:
            res = span_repr[query_batch_idx, start_ids_1, end_ids_1, :]
            return res, None
        else:
            res1 = span_repr[query_batch_idx, start_ids_1, end_ids_1, :]
            res2 = span_repr[query_batch_idx, start_ids_2, end_ids_2, :]
            return res1, res2

    def get_output_dim(self):
        if self.use_proj:
            return 2 * self.proj_dim
        else:
            return 2 * self.input_dim


class MaxSpanRepr(SpanRepr, nn.Module):
    """Class implementing the max-pool span representation."""

    def forward(self, flag, encoded_input, start_ids_1, end_ids_1, query_batch_idx, start_ids_2, end_ids_2):
        if self.use_proj:
            encoded_input = self.proj(encoded_input)

        tmp_encoded_input = encoded_input
        bsz, seq, hd = encoded_input.size()
        span_repr = torch.zeros([bsz, seq, seq, hd], device=encoded_input.device)
        for i in range(seq):
            tmp_encoded_input = (torch.maximum(encoded_input[:, i:, :], tmp_encoded_input[:, 0:seq - i, :])).float()
            span_repr[:, range(seq - i), range(i, seq), :] = tmp_encoded_input
        
        print("nan in max-pooling: ", torch.isnan(span_repr).any())
        if torch.isnan(span_repr).any():
            print("input: ", encoded_input)
            print("span_repr: ", span_repr)
        
        if start_ids_2 == None:
            res = span_repr[query_batch_idx, start_ids_1, end_ids_1, :]
            return res, None
        else:
            res1 = span_repr[query_batch_idx, start_ids_1, end_ids_1, :]
            res2 = span_repr[query_batch_idx, start_ids_2, end_ids_2, :]
            return res1, res2

    def get_output_dim(self):
        if self.use_proj:
            return self.proj_dim
        else:
            return self.input_dim


class orig_AttnSpanRepr(SpanRepr, nn.Module):
    """Class implementing the attention-based span representation."""

    def __init__(self, input_dim, use_proj=False, proj_dim=256):
        """just return the attention pooled term."""
        
        super(orig_AttnSpanRepr, self).__init__(input_dim, use_proj=use_proj, proj_dim=proj_dim)
        if use_proj:
            input_dim = proj_dim
        self.attention_params = nn.Linear(input_dim, 1)
        # Initialize weight to zero weight
        # self.attention_params.weight.data.fill_(0)
        # self.attention_params.bias.data.fill_(0)

    def forward(self, encoded_input, start_ids, end_ids, query_batch_idx):
        if self.use_proj:
            encoded_input = self.proj(encoded_input)

        span_mask = get_span_mask(start_ids, end_ids, encoded_input.shape[1])
        attn_mask = (1 - span_mask) * (-1e10)
        to_masked_encoded_input = encoded_input[query_batch_idx, :, :]
        attn_logits = self.attention_params(to_masked_encoded_input) + attn_mask
        attention_wts = nn.functional.softmax(attn_logits, dim=1)
        attention_term = torch.sum(attention_wts * to_masked_encoded_input, dim=1)
        return attention_term

    def get_output_dim(self):
        if self.use_proj:
            return self.proj_dim
        else:
            return self.input_dim


class AttnSpanRepr(SpanRepr, nn.Module):
    """Class implementing the attention-based span representation."""

    def __init__(self, input_dim, use_proj=False, proj_dim=256):
        """just return the attention pooled term."""
        
        super(AttnSpanRepr, self).__init__(input_dim, use_proj=use_proj, proj_dim=proj_dim)
        if use_proj:
            input_dim = proj_dim
        self.attention_params = nn.Linear(input_dim, 1)
        # Initialize weight to zero weight
        # self.attention_params.weight.data.fill_(0)
        # self.attention_params.bias.data.fill_(0)

    def forward(self, flag, encoded_input, start_ids_1, end_ids_1, query_batch_idx, start_ids_2, end_ids_2):
        if self.use_proj:
            encoded_input = self.proj(encoded_input)

        bsz, seq, hd = encoded_input.size()
        span_repr = torch.zeros([bsz, seq, seq, hd], device=encoded_input.device)
        for start in range(seq):
            tmp_start_ids = torch.tensor([start]).repeat(seq-start).repeat(bsz)
            tmp_end_ids = torch.arange(start, seq).repeat(bsz)
            tmp_query_batch_idx = torch.arange(bsz).repeat_interleave(seq-start).tolist()

            span_mask = get_span_mask(tmp_start_ids, tmp_end_ids, encoded_input.shape[1])

            attn_mask = (1 - span_mask) * (-1e10)
            to_masked_encoded_input = encoded_input[tmp_query_batch_idx, :, :]
            attn_logits = self.attention_params(to_masked_encoded_input) + attn_mask
            attention_wts = nn.functional.softmax(attn_logits, dim=1)
            attention_term = torch.sum(attention_wts * to_masked_encoded_input, dim=1)
            span_repr[tmp_query_batch_idx, tmp_start_ids, tmp_end_ids, :] = attention_term

        if start_ids_2 == None:
            res = span_repr[query_batch_idx, start_ids_1, end_ids_1, :]
            return res, None
        else:
            res1 = span_repr[query_batch_idx, start_ids_1, end_ids_1, :]
            res2 = span_repr[query_batch_idx, start_ids_2, end_ids_2, :]
            return res1, res2

    def get_output_dim(self):
        if self.use_proj:
            return self.proj_dim
        else:
            return self.input_dim


class FullyConnectSpanRepr(ComplexSpanRepr, nn.Module):
    
    def __init__(self, method, input_dim, use_proj=False, proj_dim=256, attn_schema=['none'], nhead=2, nlayer=2):
        
        super(FullyConnectSpanRepr, self).__init__(method, input_dim, use_proj=use_proj, proj_dim=proj_dim, 
                                                   attn_schema=attn_schema, nhead=nhead, nlayer=nlayer)
        if use_proj:
            input_dim = proj_dim
        if method == 'attn':
            self.attention_params = nn.Linear(input_dim, 1)
        if self.method in ['max', 'mean', 'attn']:
            self.output_dim = input_dim
        elif self.method in ['endpoint', 'diff_sum']:
            self.output_dim = input_dim * 2

        trans_encoder_layer = nn.TransformerEncoderLayer(d_model=self.output_dim, nhead=self.nhead)
        trans_layernorm = nn.LayerNorm(self.output_dim)
        self.trans = nn.TransformerEncoder(trans_encoder_layer, num_layers=self.nlayer, norm=trans_layernorm)

    def forward(self, flag, encoded_input, start_ids_1, end_ids_1, query_batch_idx, start_ids_2, end_ids_2):
        if self.use_proj:
            encoded_input = self.proj(encoded_input)
        
        if self.method == 'max':
            tmp_encoded_input = encoded_input
            bsz, seq, hd = encoded_input.size()
            span_repr = torch.zeros([bsz, seq, seq, hd], device=encoded_input.device)
            for i in range(seq):
                tmp_encoded_input = (torch.maximum(encoded_input[:, i:, :], tmp_encoded_input[:, 0:seq - i, :])).float()
                span_repr[:, range(seq - i), range(i, seq), :] = tmp_encoded_input
        elif self.method == 'mean':
            tmp_encoded_input = encoded_input
            bsz, seq, hd = encoded_input.size()
            span_repr = torch.zeros([bsz, seq, seq, hd], device=encoded_input.device)
            for i in range(seq):
                tmp_encoded_input = ((tmp_encoded_input[:, 0:seq - i, :] * i + encoded_input[:, i:, :]) / (i + 1)).float()
                span_repr[:, range(seq - i), range(i, seq), :] = tmp_encoded_input
        elif self.method == 'endpoint':
            bsz, seq, hd = encoded_input.size()
            hd = hd*2
            span_repr = torch.zeros([bsz, seq, seq, hd], device=encoded_input.device)
            for i in range(seq):
                span_repr[:, range(seq - i), range(i, seq), :] = torch.cat((encoded_input[:, 0:seq-i, :], encoded_input[:, i:, :]), dim=2).float()
        elif self.method == 'diff_sum':
            bsz, seq, hd = encoded_input.size()
            hd = hd*2
            span_repr = torch.zeros([bsz, seq, seq, hd], device=encoded_input.device)
            for i in range(seq):
                span_repr[:, range(seq - i), range(i, seq), :] = torch.cat((encoded_input[:, i:, :]-encoded_input[:, 0:seq-i, :], encoded_input[:, i:, :]+encoded_input[:, 0:seq-i, :]), dim=2).float()
        elif self.method == 'attn':
            bsz, seq, hd = encoded_input.size()
            span_repr = torch.zeros([bsz, seq, seq, hd], device=encoded_input.device)
            for start in range(seq):
                tmp_start_ids = torch.tensor([start]).repeat(seq-start).repeat(bsz)
                tmp_end_ids = torch.arange(start, seq).repeat(bsz)
                tmp_query_batch_idx = torch.arange(bsz).repeat_interleave(seq-start).tolist()

                span_mask = get_span_mask(tmp_start_ids, tmp_end_ids, encoded_input.shape[1])

                attn_mask = (1 - span_mask) * (-1e10)
                to_masked_encoded_input = encoded_input[tmp_query_batch_idx, :, :]
                attn_logits = self.attention_params(to_masked_encoded_input) + attn_mask
                attention_wts = nn.functional.softmax(attn_logits, dim=1)
                attention_term = torch.sum(attention_wts * to_masked_encoded_input, dim=1)
                span_repr[tmp_query_batch_idx, tmp_start_ids, tmp_end_ids, :] = attention_term
        
        bsz, seq, _, hd = span_repr.size()
        seq_t = torch.arange(seq).to(span_repr.device)
        seq_x = torch.broadcast_to(seq_t[None, None, ...], (bsz, seq, seq))
        seq_y = torch.broadcast_to(seq_t[None, ..., None], (bsz, seq, seq))
        mask = seq_x < seq_y
        src_key_mask = mask.reshape(bsz, seq*seq)

        trans_repr = span_repr.reshape(bsz, seq * seq, hd)
        span_repr = self.trans(trans_repr.permute(1, 0, 2), src_key_padding_mask=src_key_mask).permute(1, 0, 2)
        span_repr = span_repr.reshape(bsz, seq, seq, hd)

        if start_ids_2 == None:
            res = span_repr[query_batch_idx, start_ids_1, end_ids_1, :]
            return res, None
        else:
            res1 = span_repr[query_batch_idx, start_ids_1, end_ids_1, :]
            res2 = span_repr[query_batch_idx, start_ids_2, end_ids_2, :]
            return res1, res2

    def get_output_dim(self):
        return self.output_dim


class AttnSchemaSpanRepr(ComplexSpanRepr, nn.Module):
    
    def __init__(self, method, input_dim, use_proj=False, proj_dim=256, attn_schema=['none'], nhead=2, nlayer=2):
        
        super(AttnSchemaSpanRepr, self).__init__(method, input_dim, use_proj=use_proj, proj_dim=proj_dim, 
                                                   attn_schema=attn_schema, nhead=nhead, nlayer=nlayer)
        if use_proj:
            input_dim = proj_dim
        if method == 'attn':
            self.attention_params = nn.Linear(input_dim, 1)
        if self.method in ['max', 'mean', 'attn']:
            self.output_dim = input_dim
        elif self.method in ['endpoint', 'diff_sum']:
            self.output_dim = input_dim * 2

        trans_encoder_layer = myTransformerEncoderLayer(d_model=self.output_dim, nhead=self.nhead)
        trans_layernorm = nn.LayerNorm(self.output_dim)
        self.trans = myTransformerEncoder(trans_encoder_layer, num_layers=self.nlayer, norm=trans_layernorm)

    def forward(self, flag, encoded_input, start_ids_1, end_ids_1, query_batch_idx, start_ids_2, end_ids_2):
        if self.use_proj:
            encoded_input = self.proj(encoded_input)
        
        if flag:
            print("encoded_input: ", encoded_input)
        
        if self.method == 'max':
            tmp_encoded_input = encoded_input
            bsz, seq, hd = encoded_input.size()
            span_repr = torch.zeros([bsz, seq, seq, hd], device=encoded_input.device)
            for i in range(seq):
                tmp_encoded_input = (torch.maximum(encoded_input[:, i:, :], tmp_encoded_input[:, 0:seq - i, :])).float()
                span_repr[:, range(seq - i), range(i, seq), :] = tmp_encoded_input
        
            if flag:
                print("span_repr: ", span_repr)

        elif self.method == 'mean':
            tmp_encoded_input = encoded_input
            bsz, seq, hd = encoded_input.size()
            span_repr = torch.zeros([bsz, seq, seq, hd], device=encoded_input.device)
            for i in range(seq):
                tmp_encoded_input = ((tmp_encoded_input[:, 0:seq - i, :] * i + encoded_input[:, i:, :]) / (i + 1)).float()
                span_repr[:, range(seq - i), range(i, seq), :] = tmp_encoded_input
        elif self.method == 'endpoint':
            bsz, seq, hd = encoded_input.size()
            hd = hd*2
            span_repr = torch.zeros([bsz, seq, seq, hd], device=encoded_input.device)
            for i in range(seq):
                span_repr[:, range(seq - i), range(i, seq), :] = torch.cat((encoded_input[:, 0:seq-i, :], encoded_input[:, i:, :]), dim=2).float()
        elif self.method == 'diff_sum':
            bsz, seq, hd = encoded_input.size()
            hd = hd*2
            span_repr = torch.zeros([bsz, seq, seq, hd], device=encoded_input.device)
            for i in range(seq):
                span_repr[:, range(seq - i), range(i, seq), :] = torch.cat((encoded_input[:, i:, :]-encoded_input[:, 0:seq-i, :], encoded_input[:, i:, :]+encoded_input[:, 0:seq-i, :]), dim=2).float()
        elif self.method == 'attn':
            bsz, seq, hd = encoded_input.size()
            span_repr = torch.zeros([bsz, seq, seq, hd], device=encoded_input.device)
            for start in range(seq):
                tmp_start_ids = torch.tensor([start]).repeat(seq-start).repeat(bsz)
                tmp_end_ids = torch.arange(start, seq).repeat(bsz)
                tmp_query_batch_idx = torch.arange(bsz).repeat_interleave(seq-start).tolist()

                span_mask = get_span_mask(tmp_start_ids, tmp_end_ids, encoded_input.shape[1])

                attn_mask = (1 - span_mask) * (-1e10)
                to_masked_encoded_input = encoded_input[tmp_query_batch_idx, :, :]
                attn_logits = self.attention_params(to_masked_encoded_input) + attn_mask
                attention_wts = nn.functional.softmax(attn_logits, dim=1)
                attention_term = torch.sum(attention_wts * to_masked_encoded_input, dim=1)
                span_repr[tmp_query_batch_idx, tmp_start_ids, tmp_end_ids, :] = attention_term
        
        bsz, seq, _, hd = span_repr.size()

        # print("nan in max-pooling: ", torch.isnan(span_repr).any())
        # if torch.isnan(span_repr).any():
        #     print("input: ", encoded_input)
        #     print("span_repr: ", span_repr)

        trans_repr = span_repr.reshape(bsz, seq * seq, hd)
        span_repr = self.trans(trans_repr.permute(1, 0, 2), attn_pattern=self.attn_schema).permute(1, 0, 2)
        span_repr = span_repr.reshape(bsz, seq, seq, hd)

        if start_ids_2 == None:
            res = span_repr[query_batch_idx, start_ids_1, end_ids_1, :]
            if torch.isnan(res).any():
                print("nan in transformer")
            return res, None
        else:
            res1 = span_repr[query_batch_idx, start_ids_1, end_ids_1, :]
            res2 = span_repr[query_batch_idx, start_ids_2, end_ids_2, :]
            if torch.isnan(res1).any() or torch.isnan(res2).any():
                print("nan in transformer")
            return res1, res2

    def get_output_dim(self):
        return self.output_dim


def get_span_module(input_dim, method="max", use_proj=False, proj_dim=256, attn_schema=['none'], nhead=2, nlayer=2):
    """Initializes the appropriate span representation class and returns the object.
    """
    if attn_schema == ['none']:
        if method == "mean":
            return MeanSpanRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim)
        elif method == "max":
            return MaxSpanRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim)
        elif method == "diff_sum":
            return DiffSumSpanRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim)
        elif method == "endpoint":
            return EndPointRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim)
        elif method == "attn":
            return AttnSpanRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim)
        else:
            raise NotImplementedError
    elif attn_schema == ['fullyconnect']:
        return FullyConnectSpanRepr(method, input_dim, use_proj=use_proj, proj_dim=proj_dim, nhead=nhead, nlayer=nlayer)
    else:
        for item in attn_schema:
            if item not in ['insidetoken', 'samehandt', 'sibling', 'alltoken']:
                raise NotImplementedError
        return AttnSchemaSpanRepr(method, input_dim, use_proj=use_proj, proj_dim=proj_dim, attn_schema=attn_schema, nhead=nhead, nlayer=nlayer)


if __name__ == '__main__':
    span_model = AttnSpanRepr(768, use_endpoints=True, use_proj=True)
    print(span_model.get_output_dim())
    print(span_model.use_proj)
    # import encoder
    # import numpy as np
    # mymodel = encoder.Encoder(model='bert', model_size='base', use_proj=False).cuda()
    # tokenized_input = mymodel.tokenize_sentence(
    #     "Hello beautiful world!", get_subword_indices=False)
    # tokenized_input2 = mymodel.tokenize_batch(
    #     ["Hello beautiful world!", "Chomsky says hello."], get_subword_indices=False)
    # print("tokenized_input: ", tokenized_input)
    # print("tokenized_input2: ", tokenized_input2)
    # output = mymodel(tokenized_input)
    # print("output_size:", output.size())
    # print("output:", output)

    # span_model = get_span_module(768, method='mean', use_proj=False).cuda()
    # print(span_model.get_output_dim())
    # print(span_model.use_proj)
    # start = np.array([0,1])
    # end = np.array([3,2])
    # print("start: ", start, "end: ", end)
    # span_output = span_model(output, torch.from_numpy(start).cuda(), torch.from_numpy(end).cuda(), [0, 1])
    # print(span_output)
    # print(span_output.size())
