from torchtext.data import Iterator
from torch.utils.data.sampler import SequentialSampler
from torch.nn.utils.rnn import pad_sequence
import torch

pad_token_id = 0
class FixLengthLoader(object):
    def __init__(self, dataset, batch_size, shuffle, has_parse_child_rel=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.bs = batch_size
        self.sampler = SequentialSampler(self.dataset)
        self.item_dict_key = ['subwords','subword_to_word_idx','spans1','spans2']
        self.item_lst_key = ['labels','seq_len']
        if has_parse_child_rel:
            self.item_lst_key.append('parse_child_rel')

    def __iter__(self):
        batch = []
        length = 0
        for rec in self.dataset:
            length_getter = lambda x: self.dataset.instance_length_getter(x)
            if len(batch) > 0 and (len(batch) >= self.bs or length_getter(rec) != length):
                yield self.make_batch(batch)
                batch = []
            batch.append(rec)
            length = length_getter(rec)
        if len(batch) > 0:
            yield self.make_batch(batch)

    def __len__(self):
        lm = self.dataset.length_map
        l = 0
        for i in lm:
            l += lm[i] // self.bs
            if lm[i] % self.bs > 0:
                l += 1
        return l

    def make_batch(self, batch):
        encoder_key_lst = list(batch[0]['subwords'].keys())
        batch_dict = {}
        for key in batch[0].keys():
            batch_dict[key] = {} if key in self.item_dict_key else []
        for rec in batch:
            for k in rec:
                if k in self.item_dict_key:
                    if rec[k] != None: # 'ctl', 'nel' dont have spans2  ('spans2': spans['span2'] if len(spans['span2']) > 0 else None in data.py)
                        for encoder_key in encoder_key_lst:
                            if encoder_key not in batch_dict[k]:
                                batch_dict[k][encoder_key] = []
                            batch_dict[k][encoder_key].append(rec[k][encoder_key])
                else:
                    batch_dict[k].append(rec[k])
        for encoder_key in encoder_key_lst:
            if encoder_key == "glove":
                batch_dict['subwords'][encoder_key] = torch.stack(batch_dict['subwords'][encoder_key]).long()
            else:
                batch_dict['subwords'][encoder_key] = pad_sequence(batch_dict['subwords'][encoder_key], batch_first=True, padding_value=pad_token_id).long()
            if torch.cuda.is_available():
                batch_dict['subwords'][encoder_key] = batch_dict['subwords'][encoder_key].cuda()
        batch_dict['seq_len'] = torch.tensor(batch_dict['seq_len'])

        for encoder_key in encoder_key_lst:
            subword_to_word_idx_list = batch_dict['subword_to_word_idx'][encoder_key]
            subword_to_word_idx = pad_sequence(subword_to_word_idx_list, batch_first=True, padding_value=-1)
            batch_dict['subword_to_word_idx'][encoder_key] = subword_to_word_idx
        return batch_dict
