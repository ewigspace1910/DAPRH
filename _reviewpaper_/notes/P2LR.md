# Delving into Probabilistic Uncertainty for Unsupervised Domain Adaptive Person Re-identification
Jian Han, Ya-Li Li*, Shengjin Wang. _April 2022_
Clustering-based unsupervised domain adaptive (UDA) person re-identification (ReID) reduces exhaustive annotations. In this paper, the propose an approach named probabilistic uncertainty guided progressive label refinery (P2LR) for domain adaptive person reidentification.

* Official paper: [Arxiv](https://arxiv.org/pdf/2112.14025.pdf)
* Official code: [Github](https://github.com/JeyesHan/P2LR)

## Overview
Paper contain the following criterians:
1. Architecture of UDA
2. Mean Teacher
3. Probabilistic Uncertainty Modeling (PUM)
4. Uncertainty guided sample selection

## 1. Architecture of UDA
P2LR is the clustering-based method contructed by 3 components: 
![image](../images/P2LR/overall%20architecture.png)

* Pre-training with source domain (labeled datasets). In this phase, we training 2 model, which are same architecture, respectively.
* Generating pseudo labels for target domain (unlabeled dataset) by clustering algorithms then, base on these p-labels to refine samples by PUM.
* Fine-tuning with target domain attached pseudo labels. To Continue training two model and update mean teacher on Refined Samples.

## 2. Mean Teacher
*  Inspired in [MMT](https://github.com/yxgeee/MMT), Mean Teacher is composed two the exponential moving average of student models (pretrain model) over interations - $\overline{M_{1}} \overline{M_{2}} $.


  
## 3. Probabilistic Uncertainty Modeling

## 4. Uncertainty guided sample selection

