# HDCPD-ReID
![Python >=3.6](https://img.shields.io/badge/Python->=3.6-blue.svg)
![PyTorch >=1.6](https://img.shields.io/badge/PyTorch->=1.6-yellow.svg)

# Hybrid Dynamic Contrast and Probability Distillation for Unsupervised Person Re-Id (HDCPD)

The *official* repository for [*Hybrid Dynamic Contrast and Probability Distillation for Unsupervised Person Re-Id*](https://ieeeplore.ieee.org/document/9765363). 
`HDCPD` achieves state-of-the-art performances on both **unsupervised learning** tasks and **unsupervised domain adaptation** tasks for person re-ID.
For now, we could only release the test code and model already trained. We will gradually release our training code as the paper delivered.

## Prepare Datasets

We use 4 datasets in our train and test, including Market1501, DukeMTMC-ReID, MSMT17, PersonX.
Please unzip the datasets under the diretory like 
```
HDCPD-ReID/data
├── market1501
│   └── Market-1501-v15.09.15
├── msmt17
│   └── MSMT17_V1
├── personx
│   └── PersonX
└── duke
    └── DukeMTMC-reID
```
## Training
We utilize 2 tesla v100 GPU for training

To recover UDA results, run:
```shell
sh uda.sh
```

To recover USL results, run:
```shell
sh usl.sh
```

## Evaluation

We utilize 1 tesla v100 GPU for testing.

### Download Models

You can download model for market1501, duke and personx from [Here](https://pan.baidu.com/s/1XnynYhopFchqa4w9qW8LIg) with access code estr

### Unsupervised Learning

To evaluate the model released, run:
```shell
CUDA_VISIBLE_DEVICES=0 \
python test.py -d $DATASET \
  --resume $PATH_OF_MODEL
```

## Addition

To recover "Ours+ClusterContrast" results, see "ours+cc" profiles.

## Results

| Datasets | mAP(%)	| R@1(%)	| R@5(%)	| R@10(%) |
|---------|---------|---------|---------|---------|
| Market1501 | 81.6 | 92.6 | 97.4 | 98.2 |
| DukeMTMC | 69.0 | 82.9 | 90.9 | 93 |
| PersonX | 84.1 | 94.4 | 98.7 | 99.5 |
| MSMT17 | 24.6 | 50.2 | 61.4 | 65.7 |

## Acknowledgements

Thanks to [SpCL](https://github.com/yxgeee/SpCL). It is an excellent USL framework and deeply inspires our work.
