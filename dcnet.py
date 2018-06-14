#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import torch as th
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


def l2_loss(x):
    norm = th.norm(x, 2)
    return norm**2


def l2_normalize(x, dim=0, eps=1e-12):
    assert (dim < x.dim())
    norm = th.norm(x, 2, dim, keepdim=True)
    return x / (norm + eps)


class DCNet(th.nn.Module):
    def __init__(self,
                 num_bins,
                 rnn="lstm",
                 embedding_dim=20,
                 num_layers=2,
                 hidden_size=600,
                 dropout=0.0,
                 non_linear="tanh",
                 bidirectional=True):
        super(DCNet, self).__init__()
        if non_linear not in ['tanh', 'sigmoid']:
            raise ValueError(
                "Unsupported non-linear type: {}".format(non_linear))
        rnn = rnn.upper()
        if rnn not in ['RNN', 'LSTM', 'GRU']:
            raise ValueError("Unsupported rnn type: {}".format(rnn))
        self.rnn = getattr(th.nn, rnn)(
            num_bins,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional)
        self.drops = th.nn.Dropout(p=dropout)
        self.embed = th.nn.Linear(
            hidden_size * 2
            if bidirectional else hidden_size, num_bins * embedding_dim)
        self.non_linear = {
            "tanh": th.nn.functional.tanh,
            "sigmoid": th.nn.functional.sigmoid
        }[non_linear]
        self.embedding_dim = embedding_dim

    def forward(self, x, train=True):
        is_packed = isinstance(x, PackedSequence)
        if not is_packed and x.dim() != 3:
            x = th.unsqueeze(x, 0)
        x, _ = self.rnn(x)
        if is_packed:
            x, _ = pad_packed_sequence(x, batch_first=True)
        N = x.size(0)
        # N x T x H
        x = self.drops(x)
        # N x T x FD
        x = self.embed(x)
        x = self.non_linear(x)

        if train:
            # N x T x FD => N x TF x D
            x = x.view(N, -1, self.embedding_dim)
        else:
            # for inference
            # N x T x FD => NTF x D
            x = x.view(-1, self.embedding_dim)
        x = l2_normalize(x, -1)
        return x
