#!/usr/bin/env python
# coding=utf-8

# wujian@2018

import argparse
import pickle
import tqdm
import numpy as np

from dataset import SpectrogramReader
from utils import parse_yaml

def run(args):
    num_bins, conf_dict = parse_yaml(args.train_conf)
    reader = SpectrogramReader(args.wave_scp, **conf_dict["spectrogram_reader"])
    mean = np.zeros(num_bins)
    std = np.zeros(num_bins)
    num_frames = 0
    # D(X) = E(X^2) - E(X)^2
    for _, spectrogram in tqdm.tqdm(reader):
        num_frames += spectrogram.shape[0]
        mean += np.sum(spectrogram, 0)
        std += np.sum(spectrogram**2, 0)
    mean = mean / num_frames
    std = np.sqrt(std / num_frames - mean**2)
    with open(args.cmvn_dst, "wb") as f:
        cmvn_dict = {"mean": mean, "std": std}
        pickle.dump(cmvn_dict, f)
    print("Totally processed {} frames".format(num_frames))
    print("Global mean: {}".format(mean))
    print("Global std: {}".format(std))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Command to compute global cmvn stats")
    parser.add_argument(
        "wave_scp", type=str, help="Location of mixture wave scripts")
    parser.add_argument(
        "train_conf", type=str, help="Location of training configure files")
    parser.add_argument(
        "cmvn_dst", type=str, help="Location to dump cmvn stats")
    args = parser.parse_args()
    run(args)