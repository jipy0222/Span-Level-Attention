from functools import reduce

import sys
import torch
import torch.nn as nn
from torch.nn import ModuleDict
from encoders.pretrained_transformers.span_reprs import get_span_module


class SpanModel(nn.Module):
    def __init__(self, encoder_dict, span_dim=256, pool_methods=None, use_proj=False, label_itos=None, num_spans=1,
                 **kwargs):
        super().__init__()
        self.label_itos = label_itos  # a list saving the mapping from index to label, to output predictions
        self.set_encoder(encoder_dict)
        self.pool_methods = pool_methods
        self.span_nets = ModuleDict()
        if num_spans == 2:
            self.span_nets2 = ModuleDict()
        self.pooled_dim_map = {}

        self.p2e_map = _get_p2e_map(self.encoder_key_lst, self.pool_methods)

        for pool_method in self.pool_methods:
            encoder_key = self.p2e_map[pool_method]
            input_dim = self.encoder_dict[encoder_key].hidden_size
            # the pool method will be different
            self.span_nets[pool_method] = get_span_module(method=pool_method,
                                                            input_dim=input_dim,
                                                            use_proj=use_proj,
                                                            proj_dim=span_dim)
            if num_spans == 2:
                self.span_nets2[pool_method] = get_span_module(method=pool_method,
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

    def create_net(self, pooled_dim, span_dim, num_labels, num_spans):
        return nn.Sequential(
            nn.Linear(num_spans * pooled_dim, span_dim),
            nn.Tanh(),
            nn.LayerNorm(span_dim),
            nn.Dropout(0.2),
            nn.Linear(span_dim, num_labels),
            nn.Sigmoid()
        )

    # calc_span_repr: encoded_input: [batch_size, seq_len, hidden_size]
    #                span_indices: [number_of_predicted_labels, 2]
    #                query_batch_idx: [number_of_predicted_labels]
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
        # query_batch_idx = [0,0,0,1,1,2,2,2,2,3,4,4,4]

        final_repr_list = []
        # Collect pooled span repr.
        for pool_method in self.pool_methods:
            encoder_key = self.p2e_map[pool_method]
            span_net = self.span_nets[pool_method]
            if self.num_spans == 2:
                span_net2 = self.span_nets2[pool_method]
            span_encoder_input = token_encoder_output_dict[encoder_key]

            spans_idx_1 = _process_spans(spans_1[encoder_key])
            s1_repr = self.calc_span_repr(span_net, span_encoder_input, spans_idx_1, query_batch_idx)
            s_repr = s1_repr
            if self.num_spans == 2:
                spans_idx_2 = _process_spans(spans_2[encoder_key])
                s2_repr = self.calc_span_repr(span_net2, span_encoder_input, spans_idx_2, query_batch_idx)
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

def _get_p2e_map(encoder_key_lst, pool_methods):
    p2e_map = {}
    if len(encoder_key_lst) == 1:
        for pool_method in pool_methods:
            p2e_map[pool_method] = encoder_key_lst[0]
    else:  # glove and bert, no io
        for idx, pool_method in enumerate(pool_methods):
            p2e_map[pool_method] = encoder_key_lst[idx]
    return p2e_map


if __name__ == '__main__':
    from encoders.pretrained_transformers import Encoder

    def calc_span_repr(span_net, encoded_input, span_indices, query_batch_idx):
        span_start, span_end = span_indices[:, 0], span_indices[:, 1]
        span_repr = span_net(encoded_input, span_start, span_end, query_batch_idx)
        return span_repr

    batch = {'subwords': {'bert': torch.tensor([[  101,  1130,  1924,   118,   118,   102],
        [  101, 14153, 14153,   120,   119,   102],
        [  101,  1448,  1160,   120,   118,   102],
        [  101,  8790, 21205,   120,   119,   102],
        [  101,  8790, 14153,   120,   119,   102],
        [  101,  5749, 21205,   120,   119,   102],
        [  101,   157,  4538,   120,   119,   102],
        [  101,  1252,  4317,   120,   119,   102],
        [  101,  1109, 14742,  4537,   119,   102],
        [  101,  4428,  6586,  2069,   119,   102],
        [  101,  1512,  6627,  3667,   119,   102],
        [  101,  5749,   117,  3579,   119,   102],
        [  101,  1542,  1127,  1841,   119,   102],
        [  101, 19536,  3970,  6752, 14512,   102],
        [  101,   185,   119,  4925,  1604,   102],
        [  101,   185,   119,  4925,  1580,   102],
        [  101,  1753,  1142,  1214,   119,   102],
        [  101,  1188,  1110,  1999,   136,   102],
        [  101,  1203,  1365,  1392,   131,   102],
        [  101,  2748,   157, 16996, 10399,   102],
        [  101,  1960,   118,  4714,  1715,   102],
        [  101,  9294, 26835, 19162,   136,   102],
        [  101, 26835,  1204, 14001,   119,   102],
        [  101,  1119,  3024, 10898,  5035,   102],
        [  101,  1103,  5291,  1398,   118,   102],
        [  101, 15109,   118, 19753,  2240,   102],
        [  101,  8158, 12120,  7609,  1161,   102],
        [  101,  8158, 12120,  7609,  1161,   102],
        [  101,  3840,  7641,  9379,  1358,   102],
        [  101,   118, 10408,  8189,   118,   102],
        [  101,  1448,  1314,  1159,   131,   102],
        [  101,   123,   131,  3882, 14123,   102]], device='cuda:0')}, 
        'subword_to_word_idx': {'bert': torch.tensor([[-1,  0,  1,  2,  2, -1],
        [-1,  0,  1,  2,  2, -1],
        [-1,  0,  1,  2,  2, -1],
        [-1,  0,  1,  2,  2, -1],
        [-1,  0,  1,  2,  2, -1],
        [-1,  0,  1,  2,  2, -1],
        [-1,  0,  0,  1,  1, -1],
        [-1,  0,  1,  2,  2, -1],
        [-1,  0,  1,  2,  3, -1],
        [-1,  0,  1,  1,  2, -1],
        [-1,  0,  1,  2,  3, -1],
        [-1,  0,  1,  2,  3, -1],
        [-1,  0,  1,  2,  3, -1],
        [-1,  0,  0,  1,  2, -1],
        [-1,  0,  0,  1,  1, -1],
        [-1,  0,  0,  1,  1, -1],
        [-1,  0,  1,  2,  3, -1],
        [-1,  0,  1,  2,  3, -1],
        [-1,  0,  1,  2,  3, -1],
        [-1,  0,  1,  1,  1, -1],
        [-1,  0,  1,  2,  3, -1],
        [-1,  0,  1,  1,  2, -1],
        [-1,  0,  0,  0,  1, -1],
        [-1,  0,  1,  2,  3, -1],
        [-1,  0,  1,  2,  2, -1],
        [-1,  0,  1,  2,  2, -1],
        [-1,  0,  1,  1,  1, -1],
        [-1,  0,  1,  1,  1, -1],
        [-1,  0,  0,  0,  0, -1],
        [-1,  0,  1,  2,  3, -1],
        [-1,  0,  1,  2,  3, -1],
        [-1,  0,  0,  0,  1, -1]], device='cuda:0')}, 
        'spans1': {'bert': [[[2, 2]], [[2, 2], [1, 1]], [[2, 2], [1, 1]], [[2, 2]], [[2, 2]], [[2, 2]], [[1, 2]], [[2, 2]], [[1, 4]], [[2, 3]], [[1, 1]], [[3, 3]], [[1, 1]], [[4, 4]], [[3, 4]], [[3, 4]], [[2, 3]], [[3, 3]], [[1, 4]], [[1, 4]], [[1, 1]], [[2, 3]], [[1, 3]], [[3, 3]], [[2, 2]], [[3, 4], [1, 1]], [[1, 4]], [[1, 4]], [[1, 4]], [[1, 4]], [[1, 1]], [[1, 4]]]}, 
        'spans2': {}, 
        'labels': [[[7]], [[8], [8]], [[3], [3]], [[8]], [[8]], [[8]], [[8]], [[8]], [[1]], [[0]], [[3]], [[8]], [[3]], [[6]], [[3]], [[3]], [[7]], [[6]], [[6]], [[8]], [[3]], [[6]], [[6]], [[0]], [[8]], [[0], [6]], [[8]], [[8]], [[8]], [[8]], [[3]], [[12]]], 
        'seq_len': torch.tensor([3, 3, 3, 3, 3, 3, 2, 3, 4, 3, 4, 4, 4, 3, 2, 2, 4, 4, 4, 2, 4, 3, 2, 4,
        3, 3, 2, 2, 1, 4, 4, 2], device='cuda:0')}

# torch.Size([32, 6])

    subwords = batch['subwords']
    spans_1 = batch['spans1']
    spans_2 = batch['spans2']
    token_encoder_output_dict = {}

    encoder_key_lst = ['bert']
    encoder_dict = {}
    encoder_dict['bert'] = Encoder('bert', 'base', True, fine_tune=False).cuda()
    token_encoder_output_dict = {}
    for encoder_key in encoder_key_lst:
        token_encoder_output_dict[encoder_key] = encoder_dict[encoder_key](subwords[encoder_key])
    B = token_encoder_output_dict[encoder_key_lst[0]].shape[0]
    query_batch_idx = [i
                   for i in range(B)
                   for _ in range(len(spans_1[encoder_key_lst[0]][i]))]

    final_repr_list = []
    # Collect pooled span repr.
    encoder_key = 'bert'
    span_net = get_span_module(method='attn',
                            input_dim=768,
                            use_proj=False,
                            proj_dim=256).cuda()
    print("orig_weight: ", span_net.attention_params)
    my_span_net = get_span_module(method='myattn',
                            input_dim=768,
                            use_proj=False,
                            proj_dim=256).cuda()
    my_span_net.attention_params = span_net.attention_params
    print("new_weight: ", my_span_net.attention_params)
    print(span_net.attention_params == my_span_net.attention_params)

    span_encoder_input = token_encoder_output_dict[encoder_key]

    spans_idx_1 = _process_spans(spans_1[encoder_key])
    print("span_encoder_input: ", span_encoder_input)
    print("span_encoder_input_size: ", span_encoder_input.size())
    print("span_idx_1: ", spans_idx_1)
    print("span_idx_1_len: ", len(spans_idx_1))
    print("query_batch_idx: ", query_batch_idx)
    print("query_batch_idx_len: ", len(query_batch_idx))

    s1_repr = calc_span_repr(span_net, span_encoder_input, spans_idx_1, query_batch_idx)
    s1_repr_new = calc_span_repr(my_span_net, span_encoder_input, spans_idx_1, query_batch_idx)

    print("s1_repr: ", s1_repr)
    print("s1_repr_size: ", s1_repr.size())
    print("s1_repr_new: ", s1_repr_new)
    print("s1_repr_new_size: ", s1_repr_new.size())

    print(s1_repr == s1_repr_new)
        
    s_repr = s1_repr
    final_repr_list.append(s_repr)
    
    final_repr = torch.cat(final_repr_list, dim=1)

    