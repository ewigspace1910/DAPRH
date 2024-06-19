from __future__ import print_function, absolute_import
import argparse
from ast import arg
import os.path as osp
import random
import numpy as np
import os
import collections
import math
import time
import sys
import random
sys.path.append(os.getcwd())
from collections import OrderedDict
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.preprocessing import normalize

from sklearn.metrics import silhouette_samples
import torch
from torch import nn
from torch.backends import cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from modules import datasets
from modules import models
from modules.trainers import FTTrainer
from modules.evaluators import Evaluator

from modules.datasets.data import IterLoader
from modules.datasets.data import transforms as T
from modules.datasets.data.sampler import RandomIdentitySampler, RandomMultipleGallerySampler, RandomMultipleGallerySampler2
from modules.datasets.data.preprocessor import Preprocessor, TargetPreprocessor

from modules.utils import save_model
from modules.utils.logger import Logger, statistic
from modules.utils.faiss_rerank import compute_jaccard_distance, compute_ranked_list
from modules.utils.serialization import load_checkpoint, copy_state_dict
from modules.utils.osutils import mkdir_if_missing
from modules.utils.plot import plot_clusters
from modules.utils import to_torch
from modules.utils.memory import clean_cuda, mem_usage

import gc
import resource


best_mAP = 0
logger = None
# haha -=1

def get_data(name, data_dir):
    root = osp.join(data_dir)#osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader(dataset, height=256, width=128, batch_size=64, workers=4,
                     num_instances=4, iters=None, **kwargs):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])
    train_set = sorted(dataset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler2(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(TargetPreprocessor(train_set, root=None, transform=train_transformer, mutual=True),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    return train_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))


    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def create_model(args):
    model = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=args.max_clusters,  num_part=args.num_parts)
    model_ema = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=args.max_clusters, num_part=args.num_parts) if args.ema else None
    model = nn.DataParallel(model.cuda())

    initial_weights = load_checkpoint(args.init) 
    copy_state_dict(initial_weights['state_dict'], model)
    if args.ema:
        model_ema = nn.DataParallel(model_ema.cuda()) 
        copy_state_dict(initial_weights['state_dict'], model_ema)
        model_ema.module.classifier.weight.data.copy_(model.module.classifier.weight.data)
        for param in model_ema.parameters():
            param.detach_()

    return model, model_ema


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    mkdir_if_missing(args.logs_dir)
    main_worker(args)

def build_optim(epoch, model, args, **kwargs):
    scale = 1.
    if epoch < 40:
        scale = 1.0
    elif epoch < 70:
        scale = 0.1
    elif epoch < 90:
        scale = 0.01
    LR = args.lr * scale
    logger.log("LR in epoch-{} is {}".format(epoch+1, LR))
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": LR, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    return optimizer

cluster_list = []
def gen_psuedo_labels(target_dataset, models, epoch, cfg):
    global cluster_list
    model, model_ema = models
    cluster_loader = get_test_loader(target_dataset, cfg.height, cfg.width, 4 * cfg.batch_size, cfg.workers, testset=target_dataset.train)
    
    cluser_features = gfeatures = pfeatures = None
    if cfg.use_teacher:
        allfeatures = extract_all_features(model_ema, cluster_loader, finetune=True)
    else:
        allfeatures = extract_all_features(model, cluster_loader, finetune=True)

    cluster_features = gfeatures = torch.cat([allfeatures[f][0].unsqueeze(0) for f, _, _ in sorted(target_dataset.train)], 0) 
    pfeatures = torch.cat([allfeatures[f][1].unsqueeze(0) for f, _, _ in sorted(target_dataset.train)], 0)

    
    #Create first labels for samples in each branch
    logger.log('compute jaccard distance')
    try:
        rerank_dist = compute_jaccard_distance(cluster_features, print_flag=False, search_option=2) #for DBSCAN
    except:
        logger.log("out memory in CPU when computing rerank_dist!!!!")
        rerank_dist = compute_jaccard_distance(cluster_features, print_flag=False, search_option=0) #for DBSCAN
    rerank_dist_p = 0
    if len(cluster_list)==0:
        if cfg.resume:
            tri_mat = np.triu(rerank_dist, 1)
            tri_mat = tri_mat[np.nonzero(tri_mat)]
            tri_mat = np.sort(tri_mat,axis=None)
            rho = 1.6e-3
            top_num = np.round(rho*tri_mat.size).astype(int)
            eps = tri_mat[:top_num].mean()
        else: eps = cfg.cluster_eps
        min_samples = cfg.min_sample if cfg.min_sample > 0 else cfg.num_instances
        logger.log(f"minsample={min_samples}, sample/identity in batch ={cfg.num_instances}")
        cluster_list.append(DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=-1))
        cluster_list.append(DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=-1))

    # assign pseudo-labels
    ids = cluster_list[0].fit_predict(rerank_dist);del rerank_dist
    num_ids = len(set(ids)) - (1 if -1 in ids else 0)
    labels = []
    outliers = 0
    for i, id in enumerate(ids):
        if id != -1:
            labels.append(id)
        else:
            labels.append(num_ids + outliers)
            outliers += 1
    pseudo_labels= torch.Tensor(labels).long().detach()
    #partial

    new_dataset = []
    idxs, pids = [], []
    for i, ((fname, rpid, cid), label) in enumerate(zip(sorted(target_dataset.train), pseudo_labels)):
        pid = label.item()
        if pid >= num_ids:  continue
        else:
            new_dataset.append((fname, pid, cid, i, rpid))
            idxs.append(i)
            pids.append(pid)
    logger.log('Clustered into {} classes, {} outliners'.format(num_ids, len(pseudo_labels) - len(new_dataset)))
    

    # reindex
    idxs, pids = np.asarray(idxs), np.asarray(pids)
    features_g = gfeatures[idxs, :]
    features_p = pfeatures[idxs, :]
    silhouettes_g = silhouette_samples(features_g, pids)
    #statistic(silhouettes_g, name="Global-Silhouette", logger=logger) #let try exploring what happens when uncomment this :)
    # statistic(silhouettes_p, name="Partial-Silhouette", logger=logger)

    pho = cfg.pho

    # compute cluster centroids
    centroids_g, centroids_p, features__g_sil = [], [], []
    for pid in sorted(np.unique(pids)): 
        idxs__ = np.where(pids == pid)[0]
        features__g = features_g[idxs__]
        features__p = features_p[idxs__]
        silhouettes__g = silhouettes_g[idxs__]

        idxs__g = idxs__p = idxs__
        centroids_g.append(features__g.mean(0))
        centroids_p.append(features__p.mean(0))
            
        idxs__gs  = np.where(silhouettes__g > pho)[0] #refine centers
        features__g_sil.append(features__g[idxs__gs].mean(0))
    
    centroids_g = F.normalize(torch.stack(centroids_g), p=2, dim=1).float().cuda()
    model.module.classifier.weight.data[:num_ids].copy_(centroids_g)
    centroids_p = F.normalize(torch.stack(centroids_p), p=2, dim=1).float().cuda()
    model.module.classifier_concat.weight.data[:num_ids].copy_(centroids_p)
    if cfg.ema:
        model_ema.module.classifier.weight.data[:num_ids].copy_(centroids_g)
        model_ema.module.classifier_concat.weight.data[:num_ids].copy_(centroids_p)

    # Compute the centroid based scores
    if cfg.uet_al > 0:
        features__g_sil = F.normalize(torch.stack(features__g_sil), p=2, dim=1).float()
        centroid_scores_g =   compute_centroid_weight(features_g, pids, features__g_sil, temp=cfg.temp)
        centroid_scores_p = centroid_scores_g
    else:
        logger.log("No use label refinement")
        centroid_scores_g = centroid_scores_p = torch.zeros(features_g.shape[0], num_ids)

    clean_cuda(allfeatures, gfeatures, pfeatures, centroids_g, silhouettes_g, features_p)
    return new_dataset, num_ids, centroid_scores_g, centroid_scores_p


def main_worker(args):
    global best_mAP, logger
    best_mAP = 0
    cudnn.benchmark = True
    logger = Logger() if logger is None else logger
    logger.log("PHASE2: Finetuning")
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    dataset_target = get_data(args.dataset_target, args.data_dir)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, 4 * args.batch_size, args.workers)
    args.max_clusters = len(dataset_target.train)

    # Create model
    assert args.arch.find("mulpart") >= 0, "only use [network]_mulpart backbone for the finetune stage"
    model, model_ema = create_model(args)

    # Load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.init)
        copy_state_dict(checkpoint['state_dict'], model)
        args.start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['best_mAP']
        logger.log("=> Start epoch {}  best mAP {:.1%}"
              .format(args.start_epoch, best_mAP))

    # Evaluator
    evaluator= Evaluator(model, logger=logger)
    evaluator_ema= Evaluator(model_ema, logger=logger) if args.ema else None

    for epoch in range(args.start_epoch, args.epochs):
        tmp_dataset, num_clusters,  ce_scores, cep_scores = gen_psuedo_labels(target_dataset=dataset_target, 
                                            models=(model, model_ema), cfg=args, epoch=epoch)
        #plot choi
        if args.plot:
            if (epoch % 10 == 0): 
                p = plot_psuedolabel(datasets=tmp_dataset, epoch=epoch, 
                            spath=os.path.join(args.logs_dir, "figs"), 
                            number_clusters=min(num_clusters, 10))
                logger.log(msg="!Save visualization of psudo plot at epoch {} in {}".format(epoch, p))
    
        train_loader_target = get_train_loader(tmp_dataset, height=args.height, width=args.width, 
                                                batch_size=args.batch_size, workers=args.workers, 
                                                num_instances=args.num_instances, iters=iters)
        optimizer = build_optim(epoch=epoch, model=model, args=args)
        trainer = FTTrainer(model=model, model_ema=model_ema,
                            num_cluster_list=num_clusters,
                            cent_uncertainty=ce_scores, centp_uncertainty=cep_scores, 
                            uetal=args.uet_al,
                            logger=logger)
        train_loader_target.new_epoch()
        trainer.train(epoch, train_loader_target, optimizer,print_freq=args.print_freq, train_iters=len(train_loader_target), ema_weights=(args.ece, args.etri))

        
        if (args.start_epoch == epoch) or (epoch+1 >= 24 and (((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)))):
            if args.offline_test:
                logger.log("save models for offline test in epoch {}".format(epoch))
                save_model(model,is_best=False,best_mAP=0.0,mid=(epoch+1)*10+2, epoch=epoch+1, logdir=args.logs_dir)
            else:
                _, mAP = evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)
                if args.ema:
                    _, mAP_ema = evaluator_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)
                else:
                    mAP_ema = 0
                is_best = (mAP>=best_mAP) or (mAP_ema>=best_mAP)
                best_mAP = max(mAP, mAP_ema, best_mAP)
                save_model(model, (is_best and (mAP>mAP_ema)), best_mAP, 1, epoch=epoch, logdir=args.logs_dir)
                if args.ema:
                    save_model(model_ema, (is_best and (mAP<=mAP_ema)), best_mAP, 2, epoch=epoch, logdir=args.logs_dir)

                logger.log('* Finished epoch {:3d}  model mAP: {:5.1%}   model-ema mAP: {:5.1%} best: {:5.1%}{}'.
                  format(epoch+1, mAP, mAP_ema, best_mAP, ' *' if is_best else ''))
        
        
        gc.collect()
        clean_cuda(trainer, optimizer, tmp_dataset, train_loader_target, ce_scores, cep_scores)
        time.sleep(5)
    return best_mAP


def compute_centroid_weight(features, labels, centroids, temp=0.05, **kwargs):
    logger.log("Compute affinity score...")
    N, D = features.size()
    C, _ = centroids.size()
    B = 64
    features = features.unsqueeze(1)
    centers = centroids.expand(B, C, D)
    scores = torch.Tensor([])

    for i in range(0, N, B):
        feature = features[i:i+B]
        if feature.size()[0] != centers.size()[0]: centers = centroids.expand(feature.size()[0], C, D)
        dists = ((feature - centers)**2).sum(2)
        score = torch.exp(-dists/temp) / torch.exp(-dists/temp).sum(1, keepdim=True)
        scores = torch.cat([scores, score])
    
    logger.log(mem_usage())
    assert scores.size()[0] == N, "scores size 0 inequal {}".format(N)
    return scores





def extract_all_features(model, data_loader, finetune=True, **kwargs):
    model.eval()
    features = OrderedDict()
    preds = OrderedDict()
    fnames_list = []
    with torch.no_grad():
        if isinstance(data_loader, IterLoader):
            data_loader.new_epoch()
            for _ in range(len(data_loader)):
                imgs_1, fnames, _, _, _ = data_loader.next()
                inputs_1 = to_torch(imgs_1).cuda()
                fnames_list += fnames
                _, outputs, _ = model(inputs_1, finetune)
                outputs = [x.data.cpu() for x in outputs]
                
                for index, fname in enumerate(fnames):
                    features[fname] = [x[index] for x in outputs]

        else:
            for _, (imgs_1, fnames, _, _, _) in enumerate(data_loader):
                inputs_1 = to_torch(imgs_1).cuda()
                fnames_list += fnames
                _, outputs, _ = model(inputs_1, finetune)
                outputs = [x.data.cpu() for x in outputs]

                for index, fname in enumerate(fnames):
                    features[fname] = [x[index] for x in outputs]
        return features

def plot_psuedolabel(datasets, number_clusters=5, spath=None, epoch=None):
    cdict = {}
    lendict={}
    ploted_dict = {}
    for ii, (fname, pids, _, _, rpids) in enumerate(datasets):
        # if ii < 10:
        #     print(fname, pids, rpids)
        if pids in cdict: 
            cdict[pids] += [(fname, rpids)]
            lendict[pids] += 1
        else : 
            cdict[pids] = [(fname, rpids)]
            lendict[pids] = 1
            
    keys = list(cdict.keys());random.shuffle(keys)
    mid = sorted(lendict.values())[len(keys) - number_clusters]
    for i in keys:
        if number_clusters == 0 : break
        if len(cdict[i]) >= mid-1:
            ploted_dict[i] = cdict[i]
            number_clusters -= 1
    return plot_clusters(cdict=ploted_dict, 
                save_path=spath, epoch=epoch, 
                columns=min([len(ploted_dict[x]) for x in ploted_dict ] + [10]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finetune Training")
    # dataset
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501', choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")

    # training configs
    parser.add_argument('--init', type=str, default='', metavar='PATH')
    parser.add_argument('--seed', type=int, default=124)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=1)

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'logs'))

    # DBSCAN
    parser.add_argument('--cluster-eps', type=float, default=0.6, help="")
    parser.add_argument("--min-sample", type=int, default=0)
    
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.names())
    parser.add_argument("--num-parts", type=int, default=2)
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035, help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.999)
    parser.add_argument('--moving-avg-momentum', type=float, default=0)
    parser.add_argument('--weight-decay', type=float, default=5e-4)

    #weight loss
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--iters', type=int, default=400)

    #DAUET
    parser.add_argument('--temp', type=float, default=0.05)
    parser.add_argument('--pho', type=float, default=0.0, help="max distance for neighbour consisteny")
    parser.add_argument('--uet-al', type=float, default=0.3, help="alpha for refinement label")


    #EMA
    parser.add_argument('--ema', action='store_true', help='using ema mode')
    parser.add_argument("--ece", type=float, default=0.5, help='weight for ema soft ce loss')
    parser.add_argument("--etri", type=float, default=0.8, help='weight for ema soft  tri loss')

    #misc
    parser.add_argument('--offline_test', action='store_true', help='offline test models')
    parser.add_argument('--use_teacher', action='store_true', help='offline test models')
    parser.add_argument('--resume', action='store_true', help='call if continue train with checkpoint')
    parser.add_argument('--plot', action='store_true', help='call if u want to visualize psuedo label in training')

    main()
