# Delving into Probabilistic Uncertainty for Unsupervised Domain Adaptive Person Re-identification
Jian Han, Ya-Li Li*, Shengjin Wang. _April 2022_
Clustering-based unsupervised domain adaptive (UDA) person re-identification (ReID) reduces exhaustive annotations. In this paper, the propose an approach named probabilistic uncertainty guided progressive label refinery (P2LR) for domain adaptive person reidentification.

* Official paper: [Arxiv](https://arxiv.org/pdf/2112.14025.pdf)
* Official code: [Github](https://github.com/JeyesHan/P2LR)

## Overview
Paper contain the following criterions:
1. Architecture of UDA
2. Mean Teacher
3. Probabilistic Uncertainty Modeling (PUM)
4. Uncertainty guided sample selection

## I. Architecture of UDA
P2LR is the clustering-based method contructed by 3 components: 
![image](../images/P2LR/overall%20architecture.png)

1. Pre-training with source domain (labeled datasets). In this phase, we training 2 model, which are same architecture, respectively.
2. Generating pseudo labels for target domain (unlabeled dataset) by clustering algorithms then, base on these p-labels to refine samples by PUM.
3. Fine-tuning with target domain attached pseudo labels. To Continue training two model and update mean teacher on Refined Samples.

## II. Mean Teacher
*  Inspired in [MMT](https://github.com/yxgeee/MMT), Mean Teacher is composed two the exponential moving average of student models (pretrain model) over interations -  $\overline{M_{i}}$.
MT Inference is computed: 
```python
    p_out_ema = (p_out_t1_ema+p_out_t2_ema)/2
    f_out_ema = (f_out_t1_ema+f_out_t2_ema)/2
```
*  Then, $\overline{M_{i}}$ updated normaly through calculate ema $M_{i}$ weights over interations as:    
```python
    _update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch*len(data_loader_target)+i)
    _update_ema_variables(self.model_2, self.model_2_ema, self.alpha, epoch*len(data_loader_target)+i)
    #with
    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
```

  
## III. Probabilistic Uncertainty Modeling
> Inspired by this observation, we leverage the distribution difference as the probabilistic uncertainty to softly evaluate the noisy level of samples.

1. Each unlabeled sample $x_{i}$ in the target domain is assigned with a pseudo label $y^i$ by clustering. Then we distribution for samples in target domain: 
   $Q(x_{i}, \overline{y_{i}})$ 

2. Mean teacher have the external classifier for predict cluster of $x_{i}$, called $P(x_{i})$
3. Calculate the difference between clustering distribution $Q$ and classification distribution $P$ by Kullback–Leibler (KL) divergence to obtain  a criterion named probabilistic uncertainty $U$   

## IV. Uncertainty guided sample selection

Through probabilistic uncertainty in III., we will choose all samples ($x_{i}$) having U value is smaller than a threshold ${\beta}$ , which is a muable scalar and can update(increase) over interators.

## V. Training
Split to 2 stage: 

1. Pre-training on the source domain
2. End-to-end training with P2LR: 
   1.  generate pseudo labels
   2.  calculate probabilistic uncertainty and refine samples
   3.  Fine-tuning student model and update teacher
   4.  Repeat step 1 until 80 epoch :> 
