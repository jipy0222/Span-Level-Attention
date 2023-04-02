from functools import reduce

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn import ModuleDict
from encoders.pretrained_transformers.span_reprs import get_span_module
from util.io_net_loader import load_net


class SpanModel(nn.Module):
    def __init__(self, encoder_dict, span_dim=300, pool_methods=None, use_proj=False, label_itos=None, num_spans=1,
                 **kwargs):
        super().__init__()
        self.label_itos = label_itos  # a list saving the mapping from index to label, to output predictions
        self.set_encoder(encoder_dict)
        self.pool_methods = pool_methods
        self.span_nets = ModuleDict()
        self.pooled_dim_map = {}
        self.io_zero_init = kwargs.pop('io_zero_init')
        io_pool_methods = [pm for pm in pool_methods if ("iornn" in pm or "diora" in pm)]
        assert len(io_pool_methods) <= 1  # only support one type ior pooling method.
        self.p2e_map = _get_p2e_map(self.encoder_key_lst, self.pool_methods, io_pool_methods)

        for pool_method in self.pool_methods:
            encoder_key = self.p2e_map[pool_method]
            if "iornn" in pool_method or "diora" in pool_method:
                self.io_type, self.io_rep_type = pool_method.split("_")

                io_model_init = kwargs.pop('io_model_init')
                io_model_path = kwargs.pop('io_model_path')
                io_flag_path = kwargs.pop('io_flag_path')
                fine_tune_io_model = kwargs.pop('fine_tune_io_model')

                if io_model_init == 'load':
                    self.io_net, input_dim = load_net(io_model_path, io_flag_path, self.encoder_dict[encoder_key].hidden_size,
                                                      self.io_type)
                else:
                    self.io_net, input_dim = load_net(None, io_flag_path, self.encoder_dict[encoder_key].hidden_size, self.io_type)

                if not fine_tune_io_model:
                    for param in self.io_net.parameters():
                        param.requires_grad = False
            else:
                input_dim = self.encoder_dict[encoder_key].hidden_size

            self.span_nets[pool_method] = get_span_module(method=pool_method,
                                                          input_dim=input_dim,
                                                          use_proj=use_proj,
                                                          proj_dim=span_dim)
            self.pooled_dim_map[pool_method] = self.span_nets[pool_method].get_output_dim()

        num_labels = len(self.label_itos)
        self.num_spans = num_spans
        pooled_dim = sum(self.pooled_dim_map.values())
        self.pooled_dim = pooled_dim
        self.label_net = self.create_net(pooled_dim, span_dim, num_labels, self.num_spans)

        self.training_criterion = nn.BCELoss()

    def set_encoder(self, encoder_dict):
        self.encoder_dict = {}
        self.encoder_key_lst = list(encoder_dict.keys())
        self.encoder1 = encoder_dict[self.encoder_key_lst[0]]
        self.encoder_dict[self.encoder_key_lst[0]] = self.encoder1
        if len(self.encoder_key_lst) == 2:
            self.encoder2 = encoder_dict[self.encoder_key_lst[1]]
            self.encoder_dict[self.encoder_key_lst[1]] = self.encoder2

    def io_init(self, linear):
        # ValueError: "can't optimize a non-leaf Tensor" if directly change the value of weight
        if self.io_zero_init == True:
            pool_method = None
            for pm in self.pool_methods:
                if not ("iornn" in pm or "diora" in pm):
                    pool_method = pm
            pool_method_dim = self.pooled_dim_map[pool_method]
            new_weight = linear.weight
            if self.num_spans == 1:
                new_weight[:, pool_method_dim:] = 0.
            else:
                new_weight[:, 2 * pool_method_dim:] = 0.
            linear.weight = nn.Parameter(new_weight)
        return linear

    # @staticmethod
    def create_net(self, pooled_dim, span_dim, num_labels, num_spans):
        return nn.Sequential(
            self.io_init(nn.Linear(num_spans * pooled_dim, span_dim)),
            nn.Tanh(),
            nn.LayerNorm(span_dim),
            nn.Dropout(0.2),
            nn.Linear(span_dim, num_labels),
            nn.Sigmoid()
        )

    def io_net_forward(self, batch, encoded_input, rep_type, encoder_key):
        def _get_first_subword_mask(subword_to_word_idx):
            # calculate mask
            input_mask = (subword_to_word_idx != -1)
            is_first_subword = ((subword_to_word_idx[:, 1:] - subword_to_word_idx[:, :-1]) > 0).long()
            first_subword_mask = torch.zeros_like(input_mask).long()
            first_subword_mask[:, 0] = input_mask[:, 0]
            first_subword_mask[:, 1:] = is_first_subword * input_mask[:, 1:]

            return first_subword_mask

        device = encoded_input.device
        H = encoded_input.shape[2]
        seq_len = batch['seq_len']

        subword_to_word_idx = batch['subword_to_word_idx'][encoder_key]
        first_subword_mask = _get_first_subword_mask(subword_to_word_idx)
        first_subword_mask = first_subword_mask.unsqueeze(-1)

        selected_encoded_input = torch.masked_select(encoded_input, first_subword_mask.bool().to(device)).view(-1, H)
        split_selected_encoded_input = torch.split(selected_encoded_input, seq_len.tolist(), dim=0)
        new_encoded_input = pad_sequence(split_selected_encoded_input, batch_first=True)

        ret = None
        if self.io_net:
            input = {
                'encoded_input': new_encoded_input.to(device),
                'seq_len': batch['seq_len']
            }
            if 'parse_child_rel' in batch:
                input['tree'] = batch['parse_child_rel']
            ret = self.io_net(input, rep_type)
        return ret

    def calc_span_repr(self, span_net, encoded_input, span_indices, query_batch_idx):
        span_start, span_end = span_indices[:, 0], span_indices[:, 1]
        span_repr = span_net(encoded_input, span_start, span_end, query_batch_idx)
        return span_repr

    def forward(self, batch):
        subwords = batch['subwords']
        spans_1 = batch['spans1']
        spans_2 = batch['spans2']
        token_encoder_output_dict = {}
        for encoder_key in self.encoder_key_lst:
            token_encoder_output_dict[encoder_key] = self.encoder_dict[encoder_key](subwords[encoder_key])
        B = token_encoder_output_dict[self.encoder_key_lst[0]].shape[0]

        query_batch_idx = [i
                           for i in range(B)
                           for _ in range(len(spans_1[self.encoder_key_lst[0]][i]))]
        final_repr_list = []
        # Collect pooled span repr.
        for pool_method in self.pool_methods:
            encoder_key = self.p2e_map[pool_method]
            span_net = self.span_nets[pool_method]
            if "iornn" in pool_method or "diora" in pool_method:
                span_encoder_input = self.io_net_forward(batch, token_encoder_output_dict[encoder_key], self.io_rep_type, encoder_key)
            else:
                span_encoder_input = token_encoder_output_dict[encoder_key]

            spans_idx_1 = _process_spans(spans_1[encoder_key])
            s1_repr = self.calc_span_repr(span_net, span_encoder_input, spans_idx_1, query_batch_idx)
            s_repr = s1_repr
            if self.num_spans == 2:
                spans_idx_2 = _process_spans(spans_2[encoder_key])
                s2_repr = self.calc_span_repr(span_net, span_encoder_input, spans_idx_2, query_batch_idx)
                s_repr = torch.cat((s1_repr, s2_repr), dim=1)
            final_repr_list.append(s_repr)
        final_repr = torch.cat(final_repr_list, dim=1)
        pred = self.label_net(final_repr)
        return pred


def _process_spans(spans):
    """
    spans = [
        [[1,2], [2,3], [3,4]],
        [[1,2], [3,4]]
    ]

    return torch.tensor[
        [1,2],[2,3],[3,4],[1,2],[3,4]
    ]
    """
    span_list = reduce(lambda xs, x: xs + x, spans, [])
    spans_idx = torch.tensor(span_list).long()
    if torch.cuda.is_available():
        spans_idx = spans_idx.cuda()
    return spans_idx

def _get_p2e_map(encoder_key_lst, pool_methods, io_pool_methods):
    p2e_map = {}
    if len(encoder_key_lst) == 1:
        for pool_method in pool_methods:
            p2e_map[pool_method] = encoder_key_lst[0]
    else:# glove and bert
        if len(io_pool_methods) == 0: # no io
            for idx, pool_method in enumerate(pool_methods):
                p2e_map[pool_method] = encoder_key_lst[idx]
        else: # one io methods(glove) and one pooling methods(bert)
            io_pool_method = io_pool_methods[0]
            pool_method = None
            for pm in pool_methods:
                if not ("iornn" in pm or "diora" in pm):
                    pool_method = pm
            p2e_map[io_pool_method] = "glove"
            p2e_map[pool_method] = "bert"
    return p2e_map




