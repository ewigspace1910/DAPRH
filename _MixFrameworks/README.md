# DESCRIBE

I try to combine ideals from 3 papers:

-   [PPLR](https://github.com/yoonkicho/pplr): MMT, Uncertainty modeling
-   [P2LR](https://github.com/JeyesHan/P2LR): Cross agreements
-   [FL2](https://github.com/DJEddyking/LF2): MiniBatchKmeans, local features.

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
-dt unlabelwCam -b 64 -j 2 --num-clusters 2500 \
-a resnet50part --features 2048 \
--lr 0.0003 --alpha 0.999 --soft-ce-weight 0.5 --soft-tri-weight 0.8 \
--epochs 80 --iters 400 --print-freq 100 \
--multiple_kmeans --fast_kmeans \
--data-dir "_MixFrameworks/datasets" \
--logs-dir "_MixFrameworks/logs/stage2" \
--init-1 "_MixFrameworks/logs/stage1/model1/model_best.pth.tar" \
--init-2 "_MixFrameworks/logs/stage1/model2/model_best.pth.tar" \
--offline_test
```



