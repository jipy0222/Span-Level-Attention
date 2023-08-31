import json
import torch
from torch.utils.data import Dataset
import random

'''
The SingleSpanDataset and DoubleSpanDataset are different only in that DoubleSpanDataset has one more span2 to load;
it is possible to rewrite this part to use a shared parent class to provide more simplicity 

Different from the original code, here spans and labels are list, so one sentence can at most be converted to one instance
'''


class SpanDataset(Dataset):
    label_dict = dict()
    encoder = None
    def __init__(self, path, encoder_dict, train_frac=1.0, length_filter=None, **kwargs):
        super().__init__()
        word_level_span_idx = kwargs.pop('word_level_span_idx', None)
        encoder_key_list = list(encoder_dict.keys())
        self.encoder_type = encoder_key_list[0]
        with open(path, 'r') as f:
            raw_data = f.readlines()

        if train_frac < 1.0:
            red_num_lines = int(len(raw_data) * train_frac)
            raw_data = raw_data[:red_num_lines]

        # preprocess
        self.data = list()
        filter_by_length_cnt = 0
        filter_by_empty_label_cnt = 0

        for data in raw_data:
            instance = json.loads(data)
            words = instance['text'].split()

            if length_filter is not None and length_filter > 0:
                if len(words) > length_filter:
                    filter_by_length_cnt += 1
                    continue
            subwords_dict = {}
            subword_to_word_idx_dict = {}
            for encoder_key in encoder_key_list:
                subwords, subword_to_word_idx = encoder_dict[encoder_key].tokenize(words, get_subword_indices=True)
                subwords_dict[encoder_key] = subwords
                subword_to_word_idx_dict[encoder_key] = subword_to_word_idx
            span_label_pair = {}
            for item in instance['targets']:
                spans = []
                for span_key in ('span1', 'span2'):
                    if span_key in item:
                        span = tuple(item[span_key])
                        spans.append(span)

                spans = tuple(spans)
                # in case of multiple labels for one span in one sentence
                if spans not in span_label_pair:
                    span_label_pair[spans] = set()

                label = item['label']
                self.add_label(label)
                span_label_pair[spans].add(self.label_dict[label])
            
            # span_label_pair contains all the spans and labels need to be predicted in current sentence
            # form: {(span1, span2[option]): {label1, label2, ...}, ...}
            
            # Process
            def _process_span_idx(span_idx, encoder_key):
                w2w_idx = subword_to_word_idx_dict[encoder_key]
                span_idx = self.get_tokenized_span_indices(w2w_idx, span_idx)
                return span_idx
            # spans : {
            #            'span1': {'glove':[[st1, ed1], [st2, ed2], ...]
            #                      'bert':[[st1, ed1], [st2, ed2], ...]},
            #            'span2': {...}
            #            'label': [{labels for first span in this sentence}, {labels for second span}, ...]
            #                                                                                                  }
            spans = {'span1': {}, 'span2': {}, 'label': []}
            for span in span_label_pair:
                for encoder_key in encoder_key_list:
                    if encoder_key not in spans['span1']:
                        spans['span1'][encoder_key] = []
                    spans['span1'][encoder_key].append(_process_span_idx(span[0], encoder_key))
                    if len(span) > 1:
                        if encoder_key not in spans['span2']:
                            spans['span2'][encoder_key] = []
                        spans['span2'][encoder_key].append(_process_span_idx(span[1], encoder_key))
                spans['label'].append(span_label_pair[span])

            labels = [list(x) for x in spans['label']]
            if len(labels) != 0:
                for encoder_key in encoder_key_list:
                    subwords_dict[encoder_key] = torch.tensor(subwords_dict[encoder_key]).long()
                    subword_to_word_idx_dict[encoder_key] = torch.tensor(subword_to_word_idx_dict[encoder_key]).long()
                instance_dict = {
                    'subwords': subwords_dict,
                    'subword_to_word_idx': subword_to_word_idx_dict,
                    'spans1': spans['span1'],
                    'spans2': spans['span2'] if len(spans['span2']) > 0 else None,
                    'labels': labels,
                    'seq_len': len(words)
                }
                # if len(instance_dict['subwords']['bert']) >= 60:
                #     filter_by_length_cnt += 1
                # else:
                self.data.append(
                    instance_dict
                )
            else:
                filter_by_empty_label_cnt += 1

        self.data.sort(key=self.instance_length_getter)

        self.length_map = {}
        for idx, rec in enumerate(self.data):
            self.length_map.setdefault(self.instance_length_getter(rec), 0)
            self.length_map[self.instance_length_getter(rec)] += 1

        self.info = {
            'size': len(self),
            f'filter_by_length_{length_filter}': filter_by_length_cnt,
            'filter_by_empty_labels': filter_by_empty_label_cnt,
        }

    def __len__(self):
        return len(self.data)

    def instance_length_getter(self, rec):
        return len(rec['subwords'][self.encoder_type])

    def __getitem__(self, index):
        return self.data[index]
    
    def reorder(self):
        map = {}
        maxlen = -1
        for item in self.data:
            l = self.instance_length_getter(item)
            if l not in map:
                map[l] = []
            map[l].append(item)
            if l > maxlen:
                maxlen = l
        order = []
        for l in range(maxlen+1):
            if l not in map:
                continue
            order.append(l)
        random.shuffle(order)
        res = []
        for item in order:
            res.extend(map[item])
        self.data = res

    @staticmethod
    def get_tokenized_span_indices(subword_to_word_idx, orig_span_indices):
        orig_start_idx, orig_end_idx = orig_span_indices
        start_idx = subword_to_word_idx.index(orig_start_idx)
        # Search for the index of the last subword
        end_idx = len(subword_to_word_idx) - 1 - subword_to_word_idx[::-1].index(orig_end_idx - 1)
        return [start_idx, end_idx]

    @classmethod
    def add_label(cls, label):
        if label not in cls.label_dict:
            cls.label_dict[label] = len(cls.label_dict)


if __name__ == '__main__':
    from encoders.pretrained_transformers import Encoder
    data_path = '/public/home/jipy/data/ontonotes/ner/train.json'
    encoder_dict = {}
    encoder_dict['bert'] = Encoder('bert', 'base', True, fine_tune = False)
    dataset = SpanDataset(data_path, encoder_dict, train_frac=1.0, length_filter=None)
    for idx in range(5):
        print(dataset[idx])

# exapmle: 
# 
# {'subwords': 
# {'bert': tensor([  101,  1284,  4161,  5834, 13967,  1128,  1106,  2824,   170,  1957, 2596,  1104, 14754,  1975,   119,   102])}, 
# 'subword_to_word_idx': 
# {'bert': tensor([-1,  0,  1,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, -1])},
# 'spans1': {'bert': [[12, 13]]}, 'spans2': None, 
# 'labels': [[0]], 'seq_len': 13}

# {'subwords': {'bert': tensor([  101,   160,  2924,  1563, 18405,  1116,  1113,  1103,  2038,  2746,
#          1104,  1975,   131, 21342, 19917,  1104, 16191, 17204,  3757,   102])}, 'subword_to_word_idx': {'bert': tensor([-1,  0,  0,  1,  2,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 13,
#         14, -1])}, 'spans1': {'bert': [[1, 18]]}, 'spans2': None, 'labels': [[1]], 'seq_len': 15}
# {'subwords': {'bert': tensor([  101,  9996,  3543,  1113, 16191, 17204,  3757,  1110,  1103, 12267,
#          1106,  1103, 15090,  3391,  1116, 17354,   119,   102])}, 'subword_to_word_idx': {'bert': tensor([-1,  0,  1,  2,  3,  3,  4,  5,  6,  7,  8,  9, 10, 11, 11, 12, 13, -1])}, 'spans1': {'bert': [[8, 15], [4, 6]]}, 'spans2': None, 'labels': [[1], [2]], 'seq_len': 14}
# {'subwords': {'bert': tensor([  101,  1135,  1110,  2766,  1104,   170,  2425,   188,  7854,  1162,
#           117,  3718,   188,  7854,  1279,   117,   170,  3321,  1668,  7115,
#          1105, 25973,  3590,   117,  1105,  1103,  2038,  6250,   117,  1621,
#          1168,  1614,   119,   102])}, 'subword_to_word_idx': {'bert': tensor([-1,  0,  1,  2,  3,  4,  5,  6,  6,  6,  7,  8,  9,  9,  9, 10, 11, 12,
#         13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, -1])}, 'spans1': {'bert': [[25, 27]]}, 'spans2': None, 'labels': [[1]], 'seq_len': 28}
# {'subwords': {'bert': tensor([  101,   138,  2425,   188,  7854,  1162,   117,  1210,  3718,   188,
#          7854,  1279,   117,  1105,  1160, 15505,   188,  7854,  1279,   119,
#           102])}, 'subword_to_word_idx': {'bert': tensor([-1,  0,  1,  2,  2,  2,  3,  4,  5,  6,  6,  6,  7,  8,  9, 10, 11, 11,
#         11, 12, -1])}, 'spans1': {'bert': [[14, 14], [7, 7]]}, 'spans2': None, 'labels': [[3], [3]], 'seq_len': 13}
