import json
import torch
from torch.utils.data import Dataset

'''
The SingleSpanDataset and DoubleSpanDataset are different only in that DoubleSpanDataset has one more span2 to load;
it is possible to rewrite this part to use a shared parent class to provide more simplicity 

Different from the original code, here spans and labels are list, so one sentence can at most be converted to one instance
'''


def _process_child_rel(child_rel_list):
    return {tuple(p): (tuple(l), tuple(r)) for (p, (l, r)) in child_rel_list}


class SpanDataset(Dataset):
    label_dict = dict()
    encoder = None
    def __init__(self, path, encoder_dict, train_frac=1.0, length_filter=None, **kwargs):
        super().__init__()
        word_level_span_idx = kwargs.pop('word_level_span_idx')
        has_parse_child_rel = kwargs.pop('has_parse_child_rel')
        encoder_key_list = list(encoder_dict.keys())
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
                if spans not in span_label_pair:
                    span_label_pair[spans] = set()

                label = item['label']
                self.add_label(label)
                span_label_pair[spans].add(self.label_dict[label])
            # Process
            def _process_span_idx(span_idx, encoder_key):
                w2w_idx = subword_to_word_idx_dict[encoder_key]
                span_idx = self.get_tokenized_span_indices(w2w_idx, span_idx)
                return span_idx
            # spans : {
            #            'span1': {'glove':[]
            #                       'bert':[]},
            #            'span2': {...}
            #                             }
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
                if has_parse_child_rel:
                    parse_child_rel = _process_child_rel(instance['parse_child_rel'])
                    instance_dict['parse_child_rel'] = parse_child_rel
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
        if 'glove' in rec['subwords']:
            return len(rec['subwords']['glove'])
        else:
            return len(rec['subwords']['bert'])

    def __getitem__(self, index):
        return self.data[index]

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
