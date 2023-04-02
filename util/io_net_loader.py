import json
from abc import ABC

from util.logger import get_logger
from diora.net.diora import DioraTreeLSTM, DioraMLP, DioraMLPShared
from iornn.net.iornn import IORnnTreeLSTM, IORnnMLP, IORnnMLPShared
import torch.nn as nn
import torch


class IoSpanNet(nn.Module, ABC):
    def __init__(self, io_model, token_encoder_dim, model_path=None, proj_has_bias=True):
        super().__init__()
        self.size = io_model.size
        self.io_model = io_model
        self.proj_layer = torch.nn.Linear(token_encoder_dim, self.size, bias=proj_has_bias)

        logger = get_logger()
        if model_path is None or model_path == 'random':
            logger.info(f'Initializing the {self.__class__.__name__} with random parameters')
            self.reset_parameters()
        else:
            logger.info(f'Loading model {self.__class__.__name__} from {model_path}')
            self.load_model(model_path)

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def load_model(self, model_path):
        raise NotImplementedError

    def io_model_forward(self, input):
        raise NotImplementedError

    def forward(self, input, method):
        res = self.io_model_forward(input)

        if method == "inside":
            return (res.inside_h,)
        elif method == "outside":
            return (res.outside_h,)
        elif method == "both":
            return (res.inside_h, res.outside_h)


class DioraSpanNet(IoSpanNet):
    def __init__(self, diora, token_encoder_dim, model_path=None):
        super().__init__(diora, token_encoder_dim, model_path, proj_has_bias=False)

    def load_model(self, model_path):
        prefix_map = {
            'diora.': 'io_model.',
            'module.diora.': 'io_model.',
            'embed.mat': 'proj_layer.weight',
            'module.embed.mat': 'proj_layer.weight',
        }
        state_dict_toload = torch.load(model_path)['state_dict']
        mapped_state_dict = {}

        for k in state_dict_toload.keys():
            for p in prefix_map.keys():
                if k.startswith(p):
                    new_k = f'{prefix_map[p]}{k[len(p):]}'
                    mapped_state_dict[new_k] = state_dict_toload[k]
        self.load_state_dict(mapped_state_dict)

    def io_model_forward(self, input):
        encoded_input = input['encoded_input']
        proj_input = self.proj_layer(encoded_input)
        self.io_model(proj_input)
        return self.io_model


class IornnSpanNet(IoSpanNet):
    def __init__(self, iornn, token_encoder_dim, model_path=None):
        super().__init__(iornn, token_encoder_dim, model_path, proj_has_bias=True)

    def load_model(self, model_path):
        prefix_map = {
            'iornn.': 'io_model.',
            'embedding_layer.proj_layer.': 'proj_layer.',
        }

        state_dict_toload = torch.load(model_path)['state_dict']
        mapped_state_dict = {}

        for k in state_dict_toload.keys():
            for p in prefix_map.keys():
                if k.startswith(p):
                    new_k = f'{prefix_map[p]}{k[len(p):]}'
                    mapped_state_dict[new_k] = state_dict_toload[k]
        self.load_state_dict(mapped_state_dict)

    def io_model_forward(self, input):
        encoded_input = input['encoded_input']
        tree = input['tree']
        seq_len = input['seq_len']

        proj_input = self.proj_layer(encoded_input)
        self.io_model(proj_input, tree, seq_len)

        return self.io_model


def load_net(model_path, model_flag, token_encoder_dim, io_type):
    # Load model flags
    with open(model_flag) as f:
        flags = json.loads(f.read())
    arch = flags['arch']
    span_dim = flags['hidden_dim']
    compress = flags['compress']
    normalize = flags['normalize']
    cuda = flags['cuda']

    cls_map = {
        'diora': {
            'treelstm': DioraTreeLSTM,
            'mlp': DioraMLP,
            'mlp-shared': DioraMLPShared,
        },
        'iornn': {
            'treelstm': IORnnTreeLSTM,
            'mlp': IORnnMLP,
            'mlp-shared': IORnnMLPShared,
        }
    }

    cls = cls_map[io_type][arch]
    io_model = cls(span_dim, outside=True, normalize=normalize, compress=compress)
    if io_type == 'diora':
        net = DioraSpanNet(io_model, token_encoder_dim, model_path)
    elif io_type == 'iornn':
        net = IornnSpanNet(io_model, token_encoder_dim, model_path)
    else:
        net = None

    if cuda:
        net.cuda()
    return net, span_dim


if __name__ == "__main__":
    d = DioraTreeLSTM(300, outside=True, normalize='unit', compress=False)
    a = DioraSpanNet(d, "/root/alanyevs/span-rep/encoders/diora/pretrained/diora-glove-lstm/model.step_250000.pt")
