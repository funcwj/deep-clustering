import os
import warnings
import yaml

import librosa as audio_lib
import numpy as np

MAX_INT16 = np.iinfo(np.int16).max


config_keys = [
    "trainer", "dcnet", "spectrogram_reader", "dataloader", "train_scp_conf",
    "valid_scp_conf", "debug_scp_conf"
]

def nfft(window_size):
    return int(2**np.ceil(int(np.log2(window_size))))


# return F x T or T x F
def stft(file,
         frame_length=1024,
         frame_shift=256,
         window="hann",
         return_samps=False,
         apply_abs=False,
         apply_log=False,
         apply_pow=False,
         transpose=True):
    if not os.path.exists(file):
        raise FileNotFoundError("Input file {} do not exists!".format(file))
    if apply_log and not apply_abs:
        apply_abs = True
        warnings.warn(
            "Ignore apply_abs=False cause function return real values")
    samps, _ = audio_lib.load(file, sr=None)
    stft_mat = audio_lib.stft(
        samps,
        nfft(frame_length),
        frame_shift,
        frame_length,
        window=window,
        center=False)
    if apply_abs:
        stft_mat = np.abs(stft_mat)
    if apply_pow:
        stft_mat = np.power(stft_mat, 2)
    if apply_log:
        stft_mat = np.log(stft_mat)
    if transpose:
        stft_mat = np.transpose(stft_mat)
    return stft_mat if not return_samps else (samps, stft_mat)


def istft(file,
          stft_mat,
          frame_length=1024,
          frame_shift=256,
          window="hanning",
          transpose=True,
          norm=None,
          fs=16000):
    if transpose:
        stft_mat = np.transpose(stft_mat)
    samps = audio_lib.istft(stft_mat, frame_shift, frame_length, window=window)
    samps_norm = np.linalg.norm(samps, np.inf)
    # renorm if needed
    if not norm:
        samps = samps * norm / samps_norm
    samps_int16 = (samps * MAX_INT16).astype(np.int16)
    fdir = os.path.dirname(file)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    audio_lib.output.write_wav(file, samps_int16, fs)


def compute_vad_mask(spectra, threshold_db=40, apply_exp=True):
    # to linear first if needed
    if apply_exp:
        spectra = np.exp(spectra)
    # to dB
    spectra_db = 20 * np.log10(spectra)
    max_magnitude_db = np.max(spectra_db)
    threshold = 10**((max_magnitude_db - threshold_db) / 20)
    mask = np.array(spectra > threshold, dtype=np.float32)
    return mask

def apply_cmvn(feats, cmvn_dict):
    if type(cmvn_dict) != dict:
        raise TypeError("Input must be a python dictionary")
    if 'mean' in cmvn_dict:
        feats = feats - cmvn_dict['mean']
    if 'std' in cmvn_dict:
        feats = feats / cmvn_dict['std']
    return feats

def parse_scps(scp_path):
    assert os.path.exists(scp_path)
    scp_dict = dict()
    with open(scp_path, 'r') as f:
        for scp in f:
            scp_tokens = scp.strip().split()
            if len(scp_tokens) != 2:
                raise RuntimeError(
                    "Error format of context \'{}\'".format(scp))
            key, addr = scp_tokens
            if key in scp_dict:
                raise ValueError("Duplicate key \'{}\' exists!".format(key))
            scp_dict[key] = addr
    return scp_dict

def filekey(path):
    fname = os.path.basename(path)
    if not fname:
        raise ValueError("{}(Is directory path?)".format(path))
    token = fname.split(".")
    if len(token) == 1:
        return token[0]
    else:
        return '.'.join(token[:-1])

def parse_yaml(yaml_conf):
    if not os.path.exists(yaml_conf):
        raise FileNotFoundError(
            "Could not find configure files...{}".format(yaml_conf))
    with open(yaml_conf, 'r') as f:
        config_dict = yaml.load(f)

    for key in config_keys:
        if key not in config_dict:
            raise KeyError("Missing {} configs in yaml".format(key))
    num_frames = config_dict["spectrogram_reader"]["frame_length"]
    num_bins = nfft(num_frames) // 2 + 1
    return num_bins, config_dict
