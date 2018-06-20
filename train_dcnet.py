#!/usr/bin/env python
# coding=utf-8

# wujian@2018

import argparse
import os

from trainer import Trainer
from dataset import SpectrogramReader, Dataset, DataLoader, logger
from dcnet import DCNet
from utils import nfft, parse_yaml


def uttloader(scp_config, reader_kwargs, loader_kwargs, train=True):
    mix_reader = SpectrogramReader(scp_config['mixture'], **reader_kwargs)
    target_reader = [
        SpectrogramReader(scp_config[spk_key], **reader_kwargs)
        for spk_key in scp_config if spk_key[:3] == 'spk'
    ]
    dataset = Dataset(mix_reader, target_reader)
    # modify shuffle status
    loader_kwargs["shuffle"] = train
    # validate perutt if needed
    # if not train:
    #     loader_kwargs["batch_size"] = 1
    # if validate, do not shuffle
    utt_loader = DataLoader(dataset, **loader_kwargs)
    return utt_loader


def train(args):
    debug = args.debug
    logger.info(
        "Start training in {} model".format('debug' if debug else 'normal'))
    num_bins, config_dict = parse_yaml(args.config)
    reader_conf = config_dict["spectrogram_reader"]
    loader_conf = config_dict["dataloader"]
    dcnnet_conf = config_dict["dcnet"]

    batch_size = loader_conf["batch_size"]
    logger.info(
        "Training in {}".format("per utterance" if batch_size == 1 else
                                '{} utterance per batch'.format(batch_size)))

    train_loader = uttloader(
        config_dict["train_scp_conf"]
        if not debug else config_dict["debug_scp_conf"],
        reader_conf,
        loader_conf,
        train=True)
    valid_loader = uttloader(
        config_dict["valid_scp_conf"]
        if not debug else config_dict["debug_scp_conf"],
        reader_conf,
        loader_conf,
        train=False)
    checkpoint = config_dict["trainer"]["checkpoint"]
    logger.info("Training for {} epoches -> {}...".format(
        args.num_epoches, "default checkpoint"
        if checkpoint is None else checkpoint))

    dcnet = DCNet(num_bins, **dcnnet_conf)
    trainer = Trainer(dcnet, **config_dict["trainer"])
    trainer.run(train_loader, valid_loader, num_epoches=args.num_epoches)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Command to train DCNet, configured by .yaml files")
    parser.add_argument(
        "--config",
        type=str,
        default="train.yaml",
        dest="config",
        help="Location of .yaml configure files for training")
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        dest="debug",
        help="If true, start training in debug data")
    parser.add_argument(
        "--num-epoches",
        type=int,
        default=20,
        dest="num_epoches",
        help="Number of epoches to train dcnet")
    args = parser.parse_args()
    train(args)
