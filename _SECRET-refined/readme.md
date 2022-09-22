# The clone repository for [SECRET](https://github.com/LunarShen/SECRET.git)

## Installation

```shell
git clone <this repo>
cd SECRET
conda create -n secret python=3.8
conda activate secret
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install tqdm numpy six h5py Pillow scipy scikit-learn metric-learn pyyaml yacs termcolor faiss-gpu==1.6.3 opencv-python Cython
python setup.py develop
```

## Prepare Datasets

```shell
mkdir Data
```
Download the raw datasets [DukeMTMC-reID](https://arxiv.org/abs/1609.01775), [Market-1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf), [MSMT17](https://arxiv.org/abs/1711.08565),
and then unzip them under the directory like
```
SECRET/Data
├── dukemtmc
│   └── DukeMTMC-reID
├── market1501
│   └── Market-1501-v15.09.15
└── msmt17
    └── MSMT17_V1
```

## Training

### Stage I: Pre-training on the source domain

```shell
%cd /content/hahaha
!python main.py --config-file configs/pretrain.yml \
  DATASETS.DIR "/content/hahaha/datasets" \
  DATASETS.SOURCE "dukemtmc|msmt17|lpw|prai" \
  DATASETS.TARGET "market1501" \
  DATALOADER.NUM_WORKERS 2 \
  DATALOADER.BATCH_SIZE 64 \
  DATALOADER.ITERS 200 \
  OUTPUT_DIR "/content/drive/MyDrive/ducanh/Nautilus/projects/_phase1/SECRET/768pretrain" \
  GPU_Device [0] \
  MODE 'pretrain' \
  MODEL.ARCH "resnet50" MODEL.NUM_FEATURE 768
```
* ARGS:
  - MODEL.NUM_FEATURE : (int) size of the embedding vector if set 0, default size is 2048 (resnet50) or 512(resnet18-32)
  - MODEL.ARCH        : backbone resnet18, resnet34,...
  - DATASETS.SOURCE   : specific dataset name "dukemtmc" "msmt17" .... or using a combination many dataset like ""dukemtmc|msmt17|lpw|prai"  

### Stage II: fine-tuning with SECRET

```shell
%cd /content/hahaha
!python main.py --config-file configs/mutualrefine.yml \
  DATASETS.DIR "/content/hahaha/datasets" \
  DATASETS.SOURCE "dukemtmc" \
  DATASETS.TARGET "canifa" \
  CHECKPOING.PRETRAIN_PATH "/content/pretrained/pretrain-mix/checkpoint_new.pth.tar" \
  OUTPUT_DIR "/content/drive/MyDrive/ducanh/Nautilus/projects/_phase1/SECRET/canifa-1cam" \
  GPU_Device [0] OPTIM.EPOCHS 80 \
  DATALOADER.ITERS 400 DATALOADER.BATCH_SIZE 64 \
  MODE 'mutualrefine' OFFLINE_TEST True MAX_LEN_DATA 640\
  MODEL.ARCH "resnet50"  MODEL.NUM_FEATURE 768

```
* ARGS:
  - OFFLINE_TEST : (bool) Save checkpoint of each epoch and do not evaluation
  - MAX_LEN_DATA : (positive int) limit len of dataset (avoid interrupt in finetune :>)

## Evaluation

```shell
# duke-to-market
sh scripts/duke2market/eval.sh
# market-to-duke
sh scripts/market2duke/eval.sh
# market-to-msmt
sh scripts/market2msmt/eval.sh
```

## Citation
If you find this project useful for your research, please cite our paper.
```bibtex
@inproceedings{he2022secret,
  title={SECRET: Self-Consistent Pseudo Label Refinement for Unsupervised Domain Adaptive Person Re-identification},
  author={He, Tao and Shen, Leqi and Guo, Yuchen and Ding, Guiguang and Guo, Zhenhua},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={1},
  pages={879--887},
  year={2022}
}
```
