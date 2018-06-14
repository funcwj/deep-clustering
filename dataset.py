#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import os
import random
import logging
import pickle

import numpy as np
import torch as th

from torch.nn.utils.rnn import pack_sequence, pad_sequence

from utils import parse_scps, stft, compute_vad_mask, apply_cmvn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class SpectrogramReader(object):
    """
        Wrapper for short-time fourier transform of dataset
    """

    def __init__(self, wave_scp, **kwargs):
        if not os.path.exists(wave_scp):
            raise FileNotFoundError("Could not find file {}".format(wave_scp))
        self.stft_kwargs = kwargs
        self.wave_dict = parse_scps(wave_scp)
        self.wave_keys = [key for key in self.wave_dict.keys()]
        logger.info(
            "Create SpectrogramReader for {} with {} utterances".format(
                wave_scp, len(self.wave_dict)))

    def __len__(self):
        return len(self.wave_dict)

    def __contains__(self, key):
        return key in self.wave_dict

    # stft
    def _load(self, key):
        return stft(self.wave_dict[key], **self.stft_kwargs)

    # sequential index
    def __iter__(self):
        for key in self.wave_dict:
            yield key, self._load(key)

    # random index
    def __getitem__(self, key):
        if key not in self.wave_dict:
            raise KeyError("Could not find utterance {}".format(key))
        return self._load(key)


class Dataset(object):
    def __init__(self, mixture_reader, targets_reader_list):
        self.mixture_reader = mixture_reader
        self.keys_list = mixture_reader.wave_keys
        self.targets_reader_list = targets_reader_list

    def __len__(self):
        return len(self.keys_list)

    def _has_target(self, key):
        for targets_reader in self.targets_reader_list:
            if key not in targets_reader:
                return False
        return True

    def _index_by_key(self, key):
        """
            Return a tuple like (matrix, [matrix, ...])
        """
        if key not in self.mixture_reader or not self._has_target(key):
            raise KeyError("Missing targets or mixture")
        target_list = [reader[key] for reader in self.targets_reader_list]
        return (self.mixture_reader[key], target_list)

    def _index_by_num(self, num):
        """
            Return a tuple like (matrix, [matrix, ...])
        """
        if num >= len(self.keys_list):
            raise IndexError("Index out of dataset, {} vs {}".format(
                num, len(self.keys_list)))
        key = self.keys_list[num]
        return self._index_by_key(key)

    def _index_by_list(self, list_idx):
        """
            Returns a list of tuple like [
                (matrix, [matrix, ...]),
                (matrix, [matrix, ...]),
                ...
            ]
        """
        if max(list_idx) >= len(self.keys_list):
            raise IndexError("Index list contains index out of dataset")
        return [self._index_by_num(index) for index in list_idx]

    def __getitem__(self, index):
        if type(index) == int:
            return self._index_by_num(index)
        elif type(index) == str:
            return self._index_by_key(index)
        elif type(index) == list:
            return self._index_by_list(index)
        else:
            raise KeyError("Unsupported index type(int/str/list)")


class BatchSampler(object):
    def __init__(self,
                 sampler_size,
                 batch_size=16,
                 shuffle=True,
                 drop_last=False):
        if batch_size <= 0:
            raise ValueError(
                "Illegal batch_size(= {}) detected".format(batch_size))
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler_index = list(range(sampler_size))
        self.sampler_size = sampler_size
        if shuffle:
            random.shuffle(self.sampler_index)

    def __len__(self):
        return self.sampler_size

    def __iter__(self):
        base = 0
        step = self.batch_size
        while True:
            if base + step > self.sampler_size:
                break
            yield (self.sampler_index[base:base + step]
                   if step != 1 else self.sampler_index[base])
            base += step
        if not self.drop_last and base < self.sampler_size:
            yield self.sampler_index[base:]


class DataLoader(object):
    """
        Multi/Per utterance loader for DCNet training
    """

    def __init__(self,
                 dataset,
                 shuffle=True,
                 batch_size=16,
                 drop_last=False,
                 vad_threshold=40,
                 mvn_dict=None):
        self.dataset = dataset
        self.vad_threshold = vad_threshold
        self.mvn_dict = mvn_dict
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        if mvn_dict:
            logger.info("Using cmvn dictionary from {}".format(mvn_dict))
            with open(mvn_dict, "rb") as f:
                self.mvn_dict = pickle.load(f)

    def __len__(self):
        remain = len(self.dataset) % self.batch_size
        if self.drop_last or not remain:
            return len(self.dataset) // self.batch_size
        else:
            return len(self.dataset) // self.batch_size + 1

    def _transform(self, mixture_specs, targets_specs_list):
        """
            Transform from numpy/list to torch types
        """
        # compute vad mask before cmvn
        vad_mask = compute_vad_mask(
            mixture_specs, self.vad_threshold, apply_exp=True)
        # apply cmvn
        if self.mvn_dict:
            mixture_specs = apply_cmvn(mixture_specs, self.mvn_dict)
        # compute target embedding index
        target_attr = np.argmax(np.array(targets_specs_list), 0)
        return {
            "num_frames": mixture_specs.shape[0],
            "spectrogram": th.tensor(mixture_specs, dtype=th.float32),
            "target_attr": th.tensor(target_attr, dtype=th.int64),
            "silent_mask": th.tensor(vad_mask, dtype=th.float32)
        }

    def _process(self, index):
        if type(index) is list:
            dict_list = sorted(
                [self._transform(s, t) for s, t in self.dataset[index]],
                key=lambda x: x["num_frames"],
                reverse=True)
            spectrogram = pack_sequence([d["spectrogram"] for d in dict_list])
            target_attr = pad_sequence(
                [d["target_attr"] for d in dict_list], batch_first=True)
            silent_mask = pad_sequence(
                [d["silent_mask"] for d in dict_list], batch_first=True)
            return spectrogram, target_attr, silent_mask
        elif type(index) is int:
            s, t = self.dataset[index]
            data_dict = self._transform(s, t)
            return data_dict["spectrogram"], \
                   data_dict["target_attr"], \
                   data_dict["silent_mask"]
        else:
            raise ValueError("Unsupported index type({})".format(type(index)))

    def __iter__(self):
        sampler = BatchSampler(
            len(self.dataset),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last)
        num_utts = 0
        for e, index in enumerate(sampler):
            num_utts += (len(index) if type(index) is list else 1)
            if not (e + 1) % 100:
                logger.info("Processed {} batches, {} utterances".format(
                    e + 1, num_utts))
            yield self._process(index)
        logger.info("Processed {} utterances in total".format(num_utts))
