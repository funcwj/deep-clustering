#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import os
import time
import warnings

import torch as th
from torch.nn.utils.rnn import PackedSequence

from dcnet import l2_loss
from dataset import logger

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")


def create_optimizer(optimizer, params, **kwargs):
    supported_optimizer = {
        'sgd': th.optim.SGD,  # momentum, weight_decay, lr
        'rmsprop': th.optim.RMSprop,  # momentum, weight_decay, lr
        'adam': th.optim.Adam  # weight_decay, lr
        # ...
    }
    if optimizer not in supported_optimizer:
        raise ValueError('Unsupported optimizer {}'.format(optimizer))
    if optimizer == 'adam':
        del kwargs['momentum']
    opt = supported_optimizer[optimizer](params, **kwargs)
    logger.info('Create optimizer {}({})'.format(optimizer, kwargs))
    return opt


class PerUttTrainer(object):
    def __init__(self,
                 dcnet,
                 checkpoint="checkpoint",
                 optimizer="adam",
                 lr=1e-5,
                 momentum=0.9,
                 weight_decay=0,
                 clip_norm=None,
                 num_spks=2):
        self.nnet = dcnet
        logger.info("DCNet:\n{}".format(self.nnet))
        if type(lr) is str:
            lr = float(lr)
            logger.info("Transfrom lr from str to float => {}".format(lr))
        self.optimizer = create_optimizer(
            optimizer,
            self.nnet.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay)
        self.nnet.to(device)
        self.checkpoint = checkpoint
        self.num_spks = num_spks
        self.clip_norm = clip_norm
        if self.clip_norm:
            logger.info("Clip gradient by 2-norm {}".format(clip_norm))
        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint)

    def train(self, dataloader):
        self.nnet.train()
        logger.info("Training...")
        tot_loss = 0
        num_batches = len(dataloader)
        for mix_spect, tgt_index, vad_masks in dataloader:
            self.optimizer.zero_grad()
            mix_spect = mix_spect.cuda() if isinstance(
                mix_spect, PackedSequence) else mix_spect.to(device)
            tgt_index = tgt_index.to(device)
            vad_masks = vad_masks.to(device)
            # mix_spect = mix_spect * vad_masks
            net_embed = self.nnet(mix_spect)
            cur_loss = self.loss(net_embed, tgt_index, vad_masks)
            tot_loss += cur_loss.item()
            cur_loss.backward()
            if self.clip_norm:
                th.nn.utils.clip_grad_norm_(self.nnet.parameters(),
                                            self.clip_norm)
            self.optimizer.step()
        return tot_loss / num_batches, num_batches

    def validate(self, dataloader):
        self.nnet.eval()
        logger.info("Evaluating...")
        tot_loss = 0
        num_batches = len(dataloader)
        # do not need to keep gradient
        with th.no_grad():
            for mix_spect, tgt_index, vad_masks in dataloader:
                mix_spect = mix_spect.cuda() if isinstance(
                    mix_spect, PackedSequence) else mix_spect.to(device)
                tgt_index = tgt_index.to(device)
                vad_masks = vad_masks.to(device)
                # mix_spect = mix_spect * vad_masks
                net_embed = self.nnet(mix_spect)
                cur_loss = self.loss(net_embed, tgt_index, vad_masks)
                tot_loss += cur_loss.item()
        return tot_loss / num_batches, num_batches

    def run(self, train_set, dev_set, num_epoches=20):
        init_loss, _ = self.validate(dev_set)
        logger.info("Start training for {} epoches".format(num_epoches))
        logger.info("Epoch {:2d}: dev = {:.4e}".format(0, init_loss))
        th.save(self.nnet.state_dict(),
                os.path.join(self.checkpoint, 'dcnet.0.pkl'))
        for epoch in range(1, num_epoches + 1):
            on_train_start = time.time()
            train_loss, train_num_batch = self.train(train_set)
            on_valid_start = time.time()
            valid_loss, valid_num_batch = self.validate(dev_set)
            on_valid_end = time.time()
            logger.info(
                "Loss(time/num-utts) - Epoch {:2d}: train = {:.4e}({:.2f}s/{:d}) |"
                " dev = {:.4e}({:.2f}s/{:d})".format(
                    epoch, train_loss, on_valid_start - on_train_start,
                    train_num_batch, valid_loss, on_valid_end - on_valid_start,
                    valid_num_batch))
            save_path = os.path.join(self.checkpoint,
                                     'dcnet.{:d}.pkl'.format(epoch))
            th.save(self.nnet.state_dict(), save_path)
        logger.info("Training for {} epoches done!".format(num_epoches))

    def loss(self, net_embed, tgt_index, binary_mask):
        """
        Arguments:
            net_embed N x TF x D
            tgt_embed N x T x F
            binary_mask N x T x F
        """
        if tgt_index.shape != binary_mask.shape:
            raise ValueError("Dimension mismatch {} vs \
                             {}".format(tgt_index.shape, binary_mask.shape))
        if th.max(tgt_index) != self.num_spks - 1:
            warnings.warn(
                "Maybe something wrong with target embeddings computing")

        if tgt_index.dim() == 2:
            tgt_index = th.unsqueeze(tgt_index, 0)
            binary_mask = th.unsqueeze(binary_mask, 0)

        N, T, F = tgt_index.shape
        # shape binary_mask: N x TF x 1
        binary_mask = binary_mask.view(N, T * F, 1)

        # encode one-hot
        tgt_embed = th.zeros([N, T * F, self.num_spks], device=device)
        tgt_embed.scatter_(2, tgt_index.view(N, T * F, 1), 1)

        # net_embed: N x TF x D
        # tgt_embed: N x TF x S
        net_embed = net_embed * binary_mask
        tgt_embed = tgt_embed * binary_mask

        loss = l2_loss(th.bmm(th.transpose(net_embed, 1, 2), net_embed)) + \
            l2_loss(th.bmm(th.transpose(tgt_embed, 1, 2), tgt_embed)) - \
            l2_loss(th.bmm(th.transpose(net_embed, 1, 2), tgt_embed)) * 2

        return loss / th.sum(binary_mask)
