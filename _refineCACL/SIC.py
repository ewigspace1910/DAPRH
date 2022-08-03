from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import numpy
import sys
import collections
import copy
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F        

from cacl import datasets
from cacl import models
from cacl.models.hm import HybridMemory
from cacl.trainers import CACLTrainer_USL,CACLSIC_USL
from cacl.evaluators import Evaluator, extract_features
from cacl.utils.data import IterLoader
from cacl.utils.data import transforms as T
from cacl.utils.data.sampler import RandomMultipleGallerySampler
from cacl.utils.data.preprocessor import Preprocessor
from cacl.utils.logging import Logger
from cacl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from cacl.utils.faiss_rerank import compute_jaccard_distance


import os


def get_data(name, data_dir):
    root = osp.join(data_dir)
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader(args, dataset, height, width, batch_size, workers,
                    num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
	         T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])
    
    train_transformer2 = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.Grayscale(num_output_channels=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
	         T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
        ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform1=train_transformer,transform2 = train_transformer2),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader



def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform1=test_transformer,transform2 = test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0)
    model.cuda()
    model = nn.DataParallel(model)
    return model


def main():    
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    main_worker(args)
    



def main_worker(args):
    global start_epoch, best_mAP
    best_mAP =0
    start_time = time.monotonic()
    cudnn.benchmark = True
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    iters = args.iters if (args.iters>0) else None
    print("==> Load unlabeled dataset")
    args.data_dir = '/home/limingkun/Re-ID/data'
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)
    model1 = create_model(args)
    model2 = create_model(args)
    
    memory1 = HybridMemory(model1.module.num_features, len(dataset.train),
                            temp=args.temp, momentum=args.momentum).cuda()
    memory2 = HybridMemory(model2.module.num_features, len(dataset.train),
                            temp=args.temp, momentum=args.momentum).cuda()
    
    cluster_loader = get_test_loader(dataset, args.height, args.width,
                                    args.batch_size, args.workers, testset=sorted(dataset.train))
    features, _ , _ = extract_features(model1, cluster_loader, print_freq=50)
    features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
    features2, _ ,_= extract_features(model2, cluster_loader, print_freq=50)
    features2 = torch.cat([features2[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
    memory1.features = F.normalize(features, dim=1).cuda()
    memory2.features = F.normalize(features2, dim=1).cuda()
    del cluster_loader, features, features2
    evaluator1 = Evaluator(model1)
    evaluator2 = Evaluator(model2)
    
    params = []
    print('prepare parameter')
    for key, value in model1.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    for key, value in model2.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    
    
    
    
    mAP_add = 0
    rank1_add = 0
    rank5_add = 0
    rank10_add = 0
    cmc_topk = (1 , 5 , 10)
    cmc_topk = [1,5,10]
    cmc_rank = torch.zeros(3)

    # Trainer
    trainer = CACLSIC_USL(model1, model2, memory1, memory2)
    for epoch in range(args.epochs):
        # Calculate distance
        print('==> Create pseudo labels for unlabeled data with self-paced policy')
        features = memory1.features.clone()
        now_time_before_cluster =  time.monotonic()
        rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)
        del features
        
        if (epoch==0):
            eps = args.eps
            eps_tight = eps-args.eps_gap
            eps_loose = eps+args.eps_gap
            print('Clustering criterion: eps: {:.3f}, eps_tight: {:.3f}, eps_loose: {:.3f}'.format(eps, eps_tight, eps_loose))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
            cluster_tight = DBSCAN(eps=eps_tight, min_samples=4, metric='precomputed', n_jobs=-1)
            cluster_loose = DBSCAN(eps=eps_loose, min_samples=4, metric='precomputed', n_jobs=-1)
            
        def generate_pseudo_labels(cluster_id, num):
            labels = []
            outliers = 0
            for i, ((fname, _, cid), id) in enumerate(zip(sorted(dataset.train), cluster_id)):
                if id!=-1:
                    labels.append(id)
                else:
                    labels.append(num+outliers)
                    outliers += 1
            return torch.Tensor(labels).long()
        
        pseudo_labels = cluster.fit_predict(rerank_dist)
        pseudo_labels_tight = cluster_tight.fit_predict(rerank_dist)
        num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
        num_ids_tight = len(set(pseudo_labels_tight)) - (1 if -1 in pseudo_labels_tight else 0)
        
        pseudo_labels = generate_pseudo_labels(pseudo_labels, num_ids)
        pseudo_labels_tight = generate_pseudo_labels(pseudo_labels_tight, num_ids_tight)
        
        
        
        index2label = collections.defaultdict(int)
        for label in pseudo_labels:
            index2label[label.item()]+=1
        index2label = np.fromiter(index2label.values(), dtype=float)
        print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances\n'
                    .format(epoch, (index2label>1).sum(), (index2label==1).sum()))

        pseudo_weight = torch.ones(len(rerank_dist)).float()
        num_label = len(pseudo_labels)
        rerank_dist_tensor = torch.tensor(rerank_dist)
        N = pseudo_labels.size(0)
        label_sim = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()
        label_sim_tight = pseudo_labels_tight.expand(N, N).eq(pseudo_labels_tight.expand(N, N).t()).float()
        
        index2label = collections.defaultdict(int)
        for label in pseudo_labels:
            index2label[label.item()]+=1
        index2label = np.fromiter(index2label.values(), dtype=float)
        print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances\n'
                    .format(epoch, (index2label>1).sum(), (index2label==1).sum()))
        num_label = len(pseudo_labels)
        rerank_dist_tensor = torch.tensor(rerank_dist)
        N = pseudo_labels.size(0)
        label_sim = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()
        label_sim_tight = pseudo_labels_tight.expand(N, N).eq(pseudo_labels_tight.expand(N, N).t()).float()
        sim_distance = rerank_dist_tensor.clone() * label_sim
        dists_label_add = (label_sim.sum(-1))
        for i in range(len(dists_label_add)):
            if dists_label_add[i] > 1 :
                dists_label_add[i]  = dists_label_add[i] -1 
        dists_labels = (label_sim.sum(-1))        
        sim_add_averge =  sim_distance.sum(-1) / torch.pow(dists_labels,2)
        
        cluster_I_average = torch.zeros((torch.max(pseudo_labels).item() + 1))
        for sim_dists, label in (zip(sim_add_averge,pseudo_labels)):
            cluster_I_average[label.item()] = cluster_I_average[label.item()] + sim_dists 
        sim_tight = label_sim.eq(1 - label_sim_tight.clone()).float()
        dists_tight = sim_tight * rerank_dist_tensor.clone() 
        dists_label_tight_add = (1 + sim_tight.sum(-1))
        for i in range(len(dists_label_tight_add)):
            if dists_label_tight_add[i] > 1 :
                dists_label_tight_add[i]  = dists_label_tight_add[i] -1
        sim_add_averge =  dists_tight.sum(-1) / torch.pow(dists_label_tight_add,2)
        cluster_tight_average = torch.zeros((torch.max(pseudo_labels_tight).item()+ 1))
        for sim_dists, label in (zip(sim_add_averge,pseudo_labels_tight)):
            cluster_tight_average[label.item()] = cluster_tight_average[label.item()] + sim_dists
        cluster_final_averge =  torch.zeros(len(sim_add_averge))
        for i , label_tight  in enumerate(pseudo_labels_tight):
            cluster_final_averge[i] = cluster_tight_average[label_tight.item()]        
        # =====================================================
        pseudo_labeled_dataset = []
        outliers = 0
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            D_score = cluster_final_averge[i]
            
            if  args.ratio_cluster * D_score.item() <= cluster_I_average[label.item()]:
                pseudo_labeled_dataset.append((fname,label.item(),cid))
            else:
                pseudo_labeled_dataset.append((fname,len(cluster_I_average)+outliers,cid))
                pseudo_labels[i] = len(cluster_I_average)+outliers
                outliers+=1
        #  =====================================================
        now_time_after_cluster =  time.monotonic()
        print(
            'the time of cluster refinement is {}'.format(now_time_before_cluster-now_time_after_cluster)
        )

        # # #=====================================================
        # pseudo_labeled_dataset = []
        # for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
        #     pseudo_labeled_dataset.append((fname,label.item(),cid))

        # #=====================================================
        index2label = collections.defaultdict(int)
        for label in pseudo_labels:
            index2label[label.item()]+=1
        index2label = np.fromiter(index2label.values(), dtype=float)
        print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances\n'
                    .format(epoch, (index2label>1).sum(), (index2label==1).sum()))
        label_count = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()
        label_count = label_count.sum(-1)
        
        memory1.label_count = label_count
        memory2.label_count = label_count
        
        memory1.labels = pseudo_labels.cuda()
        memory2.labels = pseudo_labels.cuda()
        memory1.sic_weight = torch.tensor(args.sic_weight).cuda()
        memory2.sic_weight = torch.tensor(args.sic_weight).cuda()
        train_loader1 = get_train_loader(args, dataset, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset)
        train_loader2 = get_train_loader(args, dataset, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset)

        train_loader1.new_epoch()
        train_loader2.new_epoch()

        trainer.train(epoch, train_loader1,train_loader2, optimizer,
                    print_freq=args.print_freq, train_iters=len(train_loader1))
        
        
        now_time_after_epoch =  time.monotonic()
         
         
        print(
            'the time of cluster refinement is {}'.format(now_time_after_epoch-now_time_before_cluster)
        )
        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            cmc_socore1,mAP1 = evaluator1.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            mAP = mAP1
            print('model1 is better')
            is_best = (mAP>best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model1.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, args.dataset,args.seed,fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.format(epoch, mAP, best_mAP, ' *' if is_best else ''))
        lr_scheduler.step()

    print ('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'seed_{}_dataset_{}_model_best.pth.tar'.format(args.seed,args.dataset)))
    model1.load_state_dict(checkpoint['state_dict'])
    evaluator1.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cluster-guided Asymmetric Contrastive Learning for Unsupervised Person Re-Identification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.60,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--output_weight', type=float, default=1.0,
                        help="loss outputs for weight ")
    parser.add_argument('--ratio_cluster', type=float, default=1.0,
                        help="cluster hypter ratio ")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--loss-size', type=int, default=2)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    parser.add_argument('--sic_weight', type=float, default=1,
                        help="loss outputs for sic ")
    # training configs
    parser.add_argument('--seed', type=int, default=111)#
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main()
