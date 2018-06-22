## Deep clustering for single-channel speech separation

Implement of "Deep Clustering Discriminative Embeddings for Segmentation and Separation"

### Requirements

see [requirements.txt](requirements.txt)

### Usage

1. Configure experiments in .yaml files, for example: `train.yaml`

2. Training:

    ```shell
    python ./train_dcnet.py --config conf/train.yaml --num-epoches 20 > train.log 2>&1 &
    ```

3. Inference:
    ```
    python ./separate.py --num-spks 2 $mdl_dir/train.yaml $mdl_dir/final.pkl egs.scp
    ```

### Experiments

| Configure | Epoch |  FM   |  FF  |  MM  | FF/MM | AVG  |
| :-------: | :---: | :---: | :--: | :--: | :---: | :--: |
| [config-1](conf/1.config.yaml) |  25   | 11.42 | 6.85 | 7.88 | 7.36  | 9.54 |

### Reference

1. Hershey J R, Chen Z, Le Roux J, et al. Deep clustering: Discriminative embeddings for segmentation and separation[C]//Acoustics, Speech and Signal Processing (ICASSP), 2016 IEEE International Conference on. IEEE, 2016: 31-35.
2. Isik Y, Roux J L, Chen Z, et al. Single-channel multi-speaker separation using deep clustering[J]. arXiv preprint arXiv:1607.02173, 2016.