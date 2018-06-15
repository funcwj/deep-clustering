#!/usr/bin/env bash
# wujian@2018

mix_scp=./data/tune/mix.scp
mdl_dir=./tune/2spk_dcnet_a

set -eu

[ -d ./cache ] && rm -rf cache

mkdir cache

shuf $mix_scp | head -n30 > egs.scp

./separate.py --dump-pca --num-spks 2 $mdl_dir/train.yaml $mdl_dir/final.pkl egs.scp

rm -f egs.scp
