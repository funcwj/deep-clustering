#!/usr/bin/env bash

mix_scp=./data/tune/mix.scp
mdl_dir=./tune/2spk_dcnet_a

set -eu

[ -d ./cache ] && rm -rf cache

mkdir cache

shuf $mix_scp | head -n30 > test.scp

./separate.py --dump-pca --num-spks 2 $mdl_dir/train.yaml $mdl_dir/final.pkl test.scp

rm -f test.scp
