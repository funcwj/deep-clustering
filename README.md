## DCNet for single-channel speech separation

Implement of "Deep Clustering Discriminative Embeddings for Segmentation and Separation"

Updating...

### Requirements

see requirements.txt

### Usage

1. Configure experiments in .yaml files, for example: `train.yaml`

2. Use command:

```shell
python ./train_dcnet.py --config conf/train.yaml --num-epoches 20 > train.log 2>&1 &
```

### Experiments

| Configure | Epoch |  FM   |  FF  |  MM  | FF/MM | AVG  |
| :-------: | :---: | :---: | :--: | :--: | :---: | :--: |
| [config-1](conf/1.config.yaml) |  26   | 11.34 | 6.41 | 7.89 | 7.15  | 9.44 |