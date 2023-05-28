# DESCRIBE
This folder contains code used to reimplement some state-of-the-art solutions for person reid.

I try to combine ideals from 3 papers:

-   [PPLR](https://github.com/yoonkicho/pplr): MMT, Uncertainty modeling
-   [P2LR](https://github.com/JeyesHan/P2LR): Cross agreements
-   [FL2](https://github.com/DJEddyking/LF2): MiniBatchKmeans, local features.
-   [SCRET](https://github.com/LunarShen/SECRET.git): Extra bottleneck

# Installation
```python
python setup.py install
```

# Prepare Datasets

```
datasets
├── dukemtmc
│   └── DukeMTMC-reID
├── market1501
│   └── Market-1501-v15.09.15
└── msmt17
    └── MSMT17_V1
```

I also try to custome some datasets as
*  `modules\datasets\custom.py`
*  `modules\datasets\NoisyShoppingMall.py`

# Training
My approach is UDA includes 2 stages:

- **Stage I: Pre-training on the source domain**
- **Stage II: End-to-end training with P2LR**

```python
!python _finetune.py \
%cd /content/hahaha
!python examples/_finetune.py \
-dt unlabelwCam -b 64 -j 2 --num-clusters 3200 \
-a resnet18part --features 512 \
--lr 0.00035 --alpha 0.999 --soft-ce-weight 0.5 --soft-tri-weight 0.8 \
--flag_ca --k 20 --beta 0.25 --aals-epoch 5 --part 2 \
--epochs 80 --iters 125 --print-freq 50 \
--fast_kmeans \
--data-dir "_MixFrameworks/datasets" \
--logs-dir "_MixFrameworks/logs/stage2" \
--init-1 "_MixFrameworks/logs/stage1/model1/model_best.pth.tar" \
--init-2 "_MixFrameworks/logs/stage1/model2/model_best.pth.tar" \
--offline_test
```

# Convert torch2onnx

read in [deepstream](deepstream/)


