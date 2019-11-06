#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import argparse
import os
import pickle
import sklearn

import numpy as np
import torch as th
import scipy.io as sio

from utils import stft, istft, parse_scps, compute_vad_mask, apply_cmvn, parse_yaml, EPSILON
from dcnet import DCNet

class DeepCluster(object):
    def __init__(self, dcnet, dcnet_state, num_spks, pca=False, cuda=False):
        if not os.path.exists(dcnet_state):
            raise RuntimeError(
                "Could not find state file {}".format(dcnet_state))
        self.dcnet = dcnet

        self.location = "cuda" if args.cuda else "cpu"
        self.dcnet.load_state_dict(
            th.load(dcnet_state, map_location='cpu'))
        self.dcnet.to(self.location)
        self.dcnet.eval()
        self.kmeans = sklearn.cluster.KMeans(n_clusters=num_spks)
        self.pca = sklearn.decomposition.PCA(n_components=3) if pca else None
        self.num_spks = num_spks

    def _cluster(self, spectra, vad_mask):
        """
        Arguments
            spectra:    log-magnitude spectrogram(real numbers)
            vad_mask:   binary mask for non-silence bins(if non-sil: 1)
            return
                pca_embed: PCA embedding vector(dim 3)
                spk_masks: binary masks for each speaker
        """
        # TF x D
        net_embed = self.dcnet(
            th.tensor(spectra, dtype=th.float32, device=self.location),
            train=False).cpu().data.numpy()
        # filter silence embeddings: TF x D => N x D
        active_embed = net_embed[vad_mask.reshape(-1)]
        # classes: N x D
        # pca_mat: N x 3
        classes = self.kmeans.fit_predict(active_embed)

        pca_mat = None
        if self.pca:
            pca_mat = self.pca.fit_transform(active_embed)

        def form_mask(classes, spkid, vad_mask):
            mask = ~vad_mask
            # mask = np.zeros_like(vad_mask)
            mask[vad_mask] = (classes == spkid)
            return mask

        return pca_mat, [
            form_mask(classes, spk, vad_mask) for spk in range(self.num_spks)
        ]

    def seperate(self, spectra, cmvn=None):
        """
            spectra: stft complex results T x F
            cmvn: python dict contains global mean/std
        """
        if not np.iscomplexobj(spectra):
            raise ValueError("Input must be matrix in complex value")
        # compute log-magnitude spectrogram
        log_spectra = np.log(np.maximum(np.abs(spectra), EPSILON))
        # compute vad mask before do mvn
        vad_mask = compute_vad_mask(
            log_spectra, threshold_db=40).astype(np.bool)

        # print("Keep {} bins out of {}".format(np.sum(vad_mask), vad_mask.size))
        pca_mat, spk_masks = self._cluster(
            apply_cmvn(log_spectra, cmvn) if cmvn else log_spectra, vad_mask)

        return pca_mat, spk_masks, [
            spectra * spk_mask for spk_mask in spk_masks
        ]


def run(args):
    num_bins, config_dict = parse_yaml(args.config)
    # Load cmvn
    dict_mvn = config_dict["dataloader"]["mvn_dict"]
    if dict_mvn:
        if not os.path.exists(dict_mvn):
            raise FileNotFoundError("Could not find mvn files")
        with open(dict_mvn, "rb") as f:
            dict_mvn = pickle.load(f)

    dcnet = DCNet(num_bins, **config_dict["dcnet"])

    frame_length = config_dict["spectrogram_reader"]["frame_length"]
    frame_shift = config_dict["spectrogram_reader"]["frame_shift"]
    window = config_dict["spectrogram_reader"]["window"]

    cluster = DeepCluster(
        dcnet,
        args.dcnet_state,
        args.num_spks,
        pca=args.dump_pca,
        cuda=args.cuda)

    utt_dict = parse_scps(args.wave_scp)
    num_utts = 0
    for key, utt in utt_dict.items():
        try:
            samps, stft_mat = stft(
                utt,
                frame_length=frame_length,
                frame_shift=frame_shift,
                window=window,
                center=True,
                return_samps=True)
        except FileNotFoundError:
            print("Skip utterance {}... not found".format(key))
            continue
        print("Processing utterance {}".format(key))
        num_utts += 1
        norm = np.linalg.norm(samps, np.inf)
        pca_mat, spk_mask, spk_spectrogram = cluster.seperate(
            stft_mat, cmvn=dict_mvn)

        for index, stft_mat in enumerate(spk_spectrogram):
            istft(
                os.path.join(args.dump_dir, '{}.spk{}.wav'.format(
                    key, index + 1)),
                stft_mat,
                frame_length=frame_length,
                frame_shift=frame_shift,
                window=window,
                center=True,
                norm=norm,
                fs=8000,
                nsamps=samps.size)
            if args.dump_mask:
                sio.savemat(
                    os.path.join(args.dump_dir, '{}.spk{}.mat'.format(
                        key, index + 1)), {"mask": spk_mask[index]})
        if args.dump_pca:
            sio.savemat(
                os.path.join(args.dump_dir, '{}.mat'.format(key)),
                {"pca_matrix": pca_mat})
    print("Processed {} utterance!".format(num_utts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "Command to seperate single-channel speech using masks clustered on embeddings of DCNet"
    )
    parser.add_argument(
        "config", type=str, help="Location of training configure files")
    parser.add_argument(
        "dcnet_state", type=str, help="Location of networks state file")
    parser.add_argument(
        "wave_scp",
        type=str,
        help="Location of input wave scripts in kaldi format")
    parser.add_argument(
        "--cuda",
        default=False,
        action="store_true",
        dest="cuda",
        help="If true, inference on GPUs")
    parser.add_argument(
        "--num-spks",
        type=int,
        default=2,
        dest="num_spks",
        help="Number of speakers to be seperated")
    parser.add_argument(
        "--dump-dir",
        type=str,
        default="cache",
        dest="dump_dir",
        help="Location to dump seperated speakers")
    parser.add_argument(
        "--dump-pca",
        default=False,
        action="store_true",
        dest="dump_pca",
        help="If true, dump pca matrix")
    parser.add_argument(
        "--dump-mask",
        default=False,
        action="store_true",
        dest="dump_mask",
        help="If true, dump binary mask matrix")
    args = parser.parse_args()
    run(args)
