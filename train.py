import argparse
import sys
from functools import reduce

import numpy as np
import os
import time
import torch

from encoders.pretrained_transformers import Encoder
from model import SpanModel
from data import SpanDataset
from utils import instance_f1_info, f1_score, print_example

from util.iterator import FixLengthLoader
from util.logger import configure_logger, get_logger
from util.name import get_model_path


class LearningRateController(object):
    # Learning rate controller copied form constituent/train.py
    def __init__(self, weight_decay_range=5, terminate_range=20):
        self.data = list()
        self.not_improved = 0
        self.weight_decay_range = weight_decay_range
        self.terminate_range = terminate_range
        self.best_performance = -1e10

    def add_value(self, val):
        # add value
        if len(self.data) == 0 or val > self.best_performance:
            self.not_improved = 0
            self.best_performance = val
        else:
            self.not_improved += 1
        self.data.append(val)
        return self.not_improved


def forward_batch(model, batch, mode='loss'):
    labels_3d = batch['labels']

    preds = model(batch)

    num_pred = preds.shape[0]
    num_label = len(model.label_itos)
    one_hot_labels = torch.zeros(num_pred, num_label).long()

    def flatten_list(input_list):
        return reduce(lambda xs, x: xs + x, input_list, [])

    labels_2d = flatten_list(labels_3d)
    labels_1d = flatten_list(labels_2d)

    span_idx = reduce(lambda xs, i: xs + [i] * len(labels_2d[i]), range(num_pred), [])
    one_hot_labels[span_idx, labels_1d] = 1

    if torch.cuda.is_available():
        one_hot_labels = one_hot_labels.cuda()

    '''
    there are two ways of generating answers
    one is to pick the label value > 0.5 
    one is to pick the most possible label
    in some tasks like ctl, there might be multiple labels for one span
    '''

    # p = torch.argmax(preds, dim=1).cuda()
    # pred_labels = torch.zeros_like(preds)
    # pred_labels.scatter_(1, p.unsqueeze(dim=1), 1)
    # pred_labels = pred_labels.long()
    pred_labels = (preds > 0.5).long()

    if mode == 'pred':  # for validation
        return pred_labels, one_hot_labels
    elif mode == 'loss':  # for training
        loss = model.training_criterion(preds, one_hot_labels.float())
        return loss


def validate(loader, model, output_example=False):
    # save the random state for recovery
    rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.random.get_rng_state()
    numerator = denom_p = denom_r = 0

    for batch_dict in loader:
        preds, ans = forward_batch(model, batch_dict, mode='pred')
        num, dp, dr = instance_f1_info(ans, preds)
        numerator += num
        denom_p += dp
        denom_r += dr

    # recover the random state for reproduction
    torch.random.set_rng_state(rng_state)
    torch.cuda.random.set_rng_state(cuda_rng_state)
    return f1_score(numerator, denom_p, denom_r)


def log_arguments(args):
    # log the parameters
    logger = get_logger()
    hp_dict = vars(args)
    for key, value in hp_dict.items():
        logger.info(f"{key}\t{value}")


def set_seed(seed):
    # initialize random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_parser():
    # arguments from snippets
    parser = argparse.ArgumentParser()
    # data path
    parser.add_argument('-data_path', type=str, default='~/data/ontonotes/ner')
    parser.add_argument('-exp_path', type=str, default='exp')
    parser.add_argument('-has_parse_child_rel', action='store_true', default=False)
    # shortcuts
    # experiment type
    parser.add_argument('-task', type=str, default='nel', choices=('nel', 'ctl', 'coref', 'srl'))

    # training setting
    parser.add_argument('-batch_size', type=int, default=10)
    parser.add_argument('-real_batch_size', type=int, default=60)
    parser.add_argument('-eval_batch_size', type=int, default=10)
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-optimizer', type=str, default='Adam')
    parser.add_argument('-learning_rate', type=float, default=5e-4)
    parser.add_argument('-log_step', type=int, default=50)
    parser.add_argument('-eval_step', type=int, default=500)
    parser.add_argument('-seed', type=int, default=1111)
    parser.add_argument('-train_length_filter', default=20, type=int)
    parser.add_argument('-eval_length_filter', default=40, type=int)

    # customized arguments
    parser.add_argument('-span_dim', type=int, default=300)
    parser.add_argument('-hidden_dim', type=int, default=400)
    parser.add_argument('-use_proj', action='store_true', default=False)

    # encoder arguments
    parser.add_argument('-model_type', type=str, default='bert', choices=('glove','bert'))
    parser.add_argument('-model_size', type=str, default='base')
    parser.add_argument('-uncased', action='store_false', dest='cased')

    # pool_method
    parser.add_argument('-pool_methods', type=str, nargs="*", default='avg',
                        choices=('avg', 'max', 'diff_sum', 'endpoint', 'coherent', 'attn',
                                 'diff', 'coherent_original', 'attn_coref',
                                 'diora_inside', 'diora_outside', 'diora_both',
                                 'iornn_inside', 'iornn_outside', 'iornn_both')
                        )

    parser.add_argument('-fine_tune_all', action='store_true', default=False)
    parser.add_argument('-fine_tune_io_model', action='store_true', default=False)

    # IO model parameters
    parser.add_argument('-io_model_init', choices=('load', 'init'), default='load')
    parser.add_argument('-io_model_path', default='encoders/diora/pretrained/caijiong/model.pt') # /root/Diora-span-embedding/pretrained_io_model/init/model.step_650000.pt
    parser.add_argument('-io_flag_path', default='encoders/diora/pretrained/caijiong/flags.json') # /root/Diora-span-embedding/pretrained_io_model/init/flags.json
    parser.add_argument('-emb_path', default='~/data/glove/')
    # IO model initial
    parser.add_argument('-io_zero_init', action='store_true', default = False)

    # args for test
    parser.add_argument('-train_frac', default=1.0, type=float)
    # to activate eval mode, there I conduct tests
    parser.add_argument('-eval', action='store_true', default=False)
    parser.add_argument('-disable_loading', default=False, action='store_true',
                        help='Not to load from existing checkpoints')
    parser.add_argument('-output_example', default=False, action='store_true',
                        help='Output the incorrect results')
    parser.add_argument('-use_argmax', default=False, action='store_true',
                        help='Use argmax instead of requiring the softmax score to be > 0.5')
    parser.add_argument('-output_rp', default=False, action='store_true',
                        help='Output recall and precision')
    parser.add_argument('-test_mode', default=False, action='store_true')
    parser.add_argument('-time_limit', type=float, default=288000, help='Default time limit: 80 hours')

    # args for few shot (not considered when computing exp name)
    parser.add_argument('-k_shot', default=100, type=int)
    parser.add_argument('-n_repeat', default=10, type=int)

    return parser


def process_args(args):
    # For convenience of setting path args.
    for k, v in args.__dict__.items():
        if type(v) == str and v.startswith('~'):
            args.__dict__[k] = os.path.expanduser(v)
    return args


def main():
    parser = create_parser()
    args = parser.parse_args()
    args = process_args(args)

    set_seed(args.seed)
    if args.task in ('ctl', 'nel'):
        num_spans = 1
    elif args.task in ('coref', 'srl'):
        num_spans = 2
    else:
        raise NotImplementedError()
    # save arguments
    model_path = get_model_path(args.exp_path, args)
    log_path = os.path.join(model_path, "log")
    if not args.eval:
        configure_logger(log_path)
        log_arguments(args)
    logger = get_logger()

    args.start_time = time.time()
    logger.info(f"Model path: {model_path}")
    #####################
    # create data sets, tokenizers, and data loaders
    #####################
    # Set whether fine tune token encoder.
    encoder_dict = {}
    if type(args.pool_methods) == str:
        args.pool_methods = [args.pool_methods]
    if len(args.pool_methods) == 1:
        encoder_dict[args.model_type] = Encoder(args.model_type, args.model_size, args.cased,
                                                fine_tune=args.fine_tune_all,
                                                glove_path=args.emb_path)
    else:
        # first: other pool method  second: io pool method
        other_pool_methods = [pm for pm in args.pool_methods if not ("iornn" in pm or "diora" in pm)]
        io_pool_methods = [pm for pm in args.pool_methods if ("iornn" in pm or "diora" in pm)]
        args.pool_methods = [other_pool_methods[0], io_pool_methods[0]]
        encoder_dict["glove"] = Encoder("glove", args.model_size, args.cased, fine_tune=args.fine_tune_all,
                          glove_path=args.emb_path)
        if args.model_type == 'bert':
            encoder_dict[args.model_type] = Encoder(args.model_type, args.model_size, args.cased, fine_tune=args.fine_tune_all,
                          glove_path=args.emb_path)
    data_loader_path = os.path.join(model_path, 'dataloader.pt')
    # TODO:
    use_word_level_span_idx = any([("iornn" in pm) or ("diora" in pm) for pm in args.pool_methods]) # not used

    if args.eval:
        logger.info('Creating datasets in eval mode.')
        try:
            data_info = torch.load(data_loader_path)
            SpanDataset.label_dict = data_info['label_dict']
        except:  # dataloader do not exist or dataloader is outdated
            s = SpanDataset(
                os.path.join(args.data_path, 'train.json'),
                encoder_dict=encoder_dict,
                train_frac=args.train_frac,
                length_filter=args.train_length_filter,
                word_level_span_idx=use_word_level_span_idx,
                has_parse_child_rel = args.has_parse_child_rel
            )
        data_set = SpanDataset(
            os.path.join(args.data_path, 'test.json'),
            encoder_dict=encoder_dict,
            length_filter=args.eval_length_filter,
            word_level_span_idx=use_word_level_span_idx,
            has_parse_child_rel = args.has_parse_child_rel
        )
        data_loader = FixLengthLoader(data_set, args.eval_batch_size, shuffle=False, has_parse_child_rel=args.has_parse_child_rel)

    elif os.path.exists(data_loader_path) and not args.disable_loading:
        logger.info('Loading datasets.')
        data_info = torch.load(data_loader_path)
        data_loader = data_info['data_loader']

        for split in ['train', 'development', 'test']:
            is_train = (split == 'train')
            bs = args.batch_size if is_train else args.eval_batch_size
            data_loader[split] = FixLengthLoader(data_loader[split].dataset, bs, shuffle=is_train, has_parse_child_rel=args.has_parse_child_rel)
        SpanDataset.label_dict = data_info['label_dict']
    else:
        logger.info("Creating datasets from: %s" % args.data_path)
        data_set = dict()
        data_loader = dict()
        for split in ['train', 'development', 'test']:
            is_train = (split == 'train')
            frac = args.train_frac if is_train else 1.0
            len_filter = args.train_length_filter if is_train else args.eval_length_filter
            bs = args.batch_size if is_train else args.eval_batch_size
            data_set[split] = SpanDataset(
                os.path.join(args.data_path, f'{split}.json'),
                encoder_dict=encoder_dict,
                train_frac=frac,
                length_filter=len_filter,
                word_level_span_idx=use_word_level_span_idx,
                has_parse_child_rel=args.has_parse_child_rel
            )
            data_loader[split] = FixLengthLoader(data_set[split], bs, shuffle=is_train, has_parse_child_rel=args.has_parse_child_rel)

        torch.save(
            {
                'data_loader': data_loader,
                'label_dict': SpanDataset.label_dict
            },
            data_loader_path
        )

    logger.info("Dataset info:")
    logger.info('-' * 80)
    for split in ('train', 'development', 'test'):
        logger.info(split)
        dataset = data_loader[split].dataset
        for k in dataset.info:
            logger.info(f'{k}:{dataset.info[k]}')
        logger.info('-' * 80)

    # initialize model
    logger.info('Initializing models.')
    model = SpanModel(
        encoder_dict, span_dim=args.span_dim, pool_methods=args.pool_methods, use_proj=args.use_proj,
        label_itos={value: key for key, value in SpanDataset.label_dict.items()},
        num_spans=num_spans,
        fine_tune_io_model=args.fine_tune_io_model or args.fine_tune_all,
        io_model_init=args.io_model_init,
        io_model_path=args.io_model_path,
        io_flag_path=args.io_flag_path,
        io_zero_init=args.io_zero_init
    )

    if torch.cuda.is_available():
        model = model.cuda()

    # initialize optimizer
    if not args.eval:
        logger.info('Initializing optimizer.')

        logger.info('Fine tune information: ')
        if args.fine_tune_all:
            logger.info('Fine tuning all the parameters in Encoder and IoModel')
        elif args.fine_tune_io_model:
            logger.info('Fine tuning all the parameters in IoModel')

        logger.info('Trainable parameters: ')
        params = list()
        names = list()
        for name, param in list(model.named_parameters()):
            if param.requires_grad:
                params.append(param)
                names.append(name)
                logger.info(f"{name}: {param.data.size()}")
        optimizer = getattr(torch.optim, args.optimizer)(params, lr=args.learning_rate)
    # initialize best model info, and lr controller
    best_f1 = 0
    best_model = None
    lr_controller = LearningRateController()

    # load checkpoint, if exists
    args.start_epoch = 0
    args.epoch_step = -1
    ckpt_path = os.path.join(model_path, 'ckpt')

    if args.eval:
        checkpoint = torch.load(ckpt_path)
        best_model = checkpoint['best_model']
        assert best_model is not None
        model.load_state_dict(best_model)
        model.eval()
        with torch.no_grad():
            test_f1 = validate(data_loader, model, args.output_example)
            logger.info(f'Test F1 {test_f1 * 100:6.2f}%')
        return 0

    if os.path.exists(ckpt_path) and not args.disable_loading:
        logger.info(f'Loading checkpoint from {ckpt_path}.')
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model'])
        best_model = checkpoint['best_model']
        best_f1 = checkpoint['best_f1']
        if not args.eval:
            optimizer.load_state_dict(checkpoint['optimizer'])
        lr_controller = checkpoint['lr_controller']
        torch.cuda.random.set_rng_state(checkpoint['cuda_rng_state'])
        args.start_epoch = checkpoint['epoch']
        args.epoch_step = checkpoint['step']
        # encoder.weighing_params.data = checkpoint['weighing_params'].data
        # if lr_controller.not_improved >= lr_controller.terminate_range:
        #     logger.info('No more optimization, testing and exiting.')
        #     assert best_model is not None
        #     model.load_state_dict(best_model)
        #     encoder.load_state_dict(best_encoder)
        #     encoder.weighing_params.data = best_weighing_params.data
        #     model.eval()
        #     with torch.no_grad():
        #         test_f1 = validate(data_loader['test'], model, encoder)
        #     logger.info(f'Test F1 {test_f1 * 100:6.2f}%')
        #     exit(0)

    logger.info('Initial model:')
    model.eval()
    logger.info('-' * 80)
    with torch.no_grad():
        curr_f1 = validate(data_loader['development'], model)
    logger.info(f'Validation F1 {curr_f1 * 100:6.2f}%')

    # training
    terminate = False
    for epoch in range(args.epochs):
        if terminate:
            break
        model.train()
        cumulated_loss = cumulated_num = 0
        for step, batch in enumerate(data_loader['train']):
            if terminate:
                break
            # ignore batches to recover the same data loader state of checkpoint
            if (epoch < args.start_epoch) or (epoch == args.start_epoch and step <= args.epoch_step):
                continue

            loss = forward_batch(model, batch, mode='loss')
            actual_step = len(data_loader['train']) * epoch + step + 1
            # optimize model
            if (actual_step - 1) % (args.real_batch_size // args.batch_size) == 0:
                optimizer.zero_grad()

            loss.backward()

            if actual_step % (args.real_batch_size // args.batch_size) == 0:
                optimizer.step()
            # update metadata
            num_instances = len(batch['labels'])
            cumulated_loss += loss.item() * num_instances
            cumulated_num += num_instances
            # log
            if (actual_step % (args.real_batch_size // args.batch_size) == 0) and (
                    actual_step // (args.real_batch_size // args.batch_size)) % args.log_step == 0:
                logger.info(
                    f'Train '
                    f'Epoch #{epoch} | Step {actual_step // (args.real_batch_size // args.batch_size)} | '
                    f'loss {cumulated_loss / cumulated_num:8.4f}'
                )
            # validate
            if (actual_step % (args.real_batch_size // args.batch_size) == 0) and (
                    actual_step // (args.real_batch_size // args.batch_size)) % args.eval_step == 0:
                model.eval()
                logger.info('-' * 80)
                with torch.no_grad():
                    curr_f1 = validate(data_loader['development'], model)
                logger.info(f'Validation F1 {curr_f1 * 100:6.2f}%')
                # update when there is a new best model
                if curr_f1 > best_f1:
                    best_f1 = curr_f1
                    best_model = model.state_dict()
                    logger.info('New best model!')
                logger.info('-' * 80)
                model.train()
                # update validation result
                not_improved_epoch = lr_controller.add_value(curr_f1)
                if not_improved_epoch == 0:
                    pass
                elif not_improved_epoch >= lr_controller.terminate_range:
                    logger.info(
                        'Terminating due to lack of validation improvement.')
                    terminate = True
                elif not_improved_epoch % lr_controller.weight_decay_range == 0:
                    logger.info(
                        f'Re-initialize learning rate to '
                        f'{optimizer.param_groups[0]["lr"] / 2.0:.8f}'
                    )
                    optimizer = getattr(torch.optim, args.optimizer)(
                        params,
                        lr=optimizer.param_groups[0]['lr'] / 2.0
                    )
                # save checkpoint
                torch.save({
                    'model': model.state_dict(),
                    'best_model': best_model,
                    'best_f1': best_f1,
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'lr_controller': lr_controller,
                    'cuda_rng_state': torch.cuda.random.get_rng_state(),
                }, ckpt_path)
                # pre-terminate to avoid saving problem
                if (time.time() - args.start_time) >= args.time_limit:
                    logger.info('Training time is almost up -- terminating.')
                    exit(0)

    # finished training, testing
    assert best_model is not None
    model.load_state_dict(best_model)
    # encoder.load_state_dict(best_encoder)
    # encoder.weighing_params.data = best_weighing_params.data
    model.eval()
    with torch.no_grad():
        test_f1 = validate(data_loader['test'], model)
    logger.info(f'Test F1 {test_f1 * 100:6.2f}%')


if __name__ == '__main__':
    main()
