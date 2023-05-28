from __future__ import print_function, absolute_import
import argparse
from ast import arg
import os.path as osp
import random
import numpy as np
import os
import math
import sys
sys.path.append(os.getcwd())
from collections import OrderedDict
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.preprocessing import normalize

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from modules import datasets
from modules import models
from modules.trainers import MMTTrainer, MMTwCaTrainer
from modules.evaluators import Evaluator, extract_features
from modules.utils.data import IterLoader
from modules.utils.data import transforms as T
from modules.utils.data.sampler import RandomMultipleGallerySampler
from modules.utils.data.preprocessor import Preprocessor
from modules.utils.logging import Logger
from modules.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict

from modules.utils.meters import AverageMeter
from modules.utils import to_torch
from modules.utils.rerank import compute_jaccard_dist
from modules.utils.faiss_rerank import compute_jaccard_distance, compute_ranked_list

best_mAP = 0


def get_data(name, data_dir):
    root = osp.join(data_dir)#osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


class ProbUncertain():
    def __init__(self, alpha=20, epsilon=0.99):
        self.alpha = alpha
        self.epsilon = epsilon
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.kl_loss = torch.nn.KLDivLoss(reduction='none')
    
    def cal_uncertainty(self, features, pseudo_labels, classifier):
        features, classifier = torch.from_numpy(features), torch.from_numpy(classifier)
        pred_probs =  self.logsoftmax(self.alpha * torch.matmul(features, classifier.t()))

        pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long)
        ideal_probs = torch.zeros(pred_probs.shape) + (1-self.epsilon) / (pred_probs.shape[1]-1)
        ideal_probs.scatter_(1, pseudo_labels.unsqueeze(-1), value=self.epsilon)

        uncertainties = self.kl_loss(pred_probs, ideal_probs).sum(1).numpy()
        return uncertainties

prob_uncertainty = ProbUncertain()
len_train_set = 0

def get_train_loader(dataset, height, width, batch_size, workers,
                     num_instances, iters, centers, target_label, cf, pt):

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
    uncertainties = prob_uncertainty.cal_uncertainty(cf, target_label, centers)
    N = len(uncertainties) 
    beta = np.sort(uncertainties)[int(pt * N) - 1]
    Vindicator = [False for _ in range(N)]
    for i in range(N):
        if uncertainties[i] <= beta:
            Vindicator[i] = True
    Vindicator = np.array(Vindicator)
    select_samples_inds = np.where(Vindicator == True)[0]
    select_samples_labels = target_label[select_samples_inds]
    train_set = [dataset.train[ind] for ind in select_samples_inds]

    # change pseudo labels
    for i in range(len(train_set)):
        train_set[i] = list(train_set[i])
        train_set[i][1] = int(select_samples_labels[i])
        train_set[i] = tuple(train_set[i])

    print('select {}/{} samples'.format(len(train_set), N))
    len_train_set = len(train_set)

    train_set = sorted(train_set)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                transform=train_transformer, mutual=True),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader, select_samples_inds, select_samples_labels


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
    model_1 = models.create(args.arch, num_features=args.features, dropout=args.dropout, 
                            num_classes=args.num_clusters, num_parts=args.part,
                            extra_bn=args.extra_bottleneck)
    model_2 = models.create(args.arch, num_features=args.features, dropout=args.dropout, 
                            num_classes=args.num_clusters, num_parts=args.part,
                            extra_bn=args.extra_bottleneck)
    model_1_ema = models.create(args.arch, num_features=args.features, dropout=args.dropout, 
                            num_classes=args.num_clusters, num_parts=args.part,
                            extra_bn=args.extra_bottleneck)
    model_2_ema = models.create(args.arch, num_features=args.features, dropout=args.dropout, 
                            num_classes=args.num_clusters, num_parts=args.part,
                            extra_bn=args.extra_bottleneck)
    model_1.cuda()
    model_2.cuda()
    model_1_ema.cuda()
    model_2_ema.cuda()
    model_1 = nn.DataParallel(model_1)
    model_2 = nn.DataParallel(model_2)
    model_1_ema = nn.DataParallel(model_1_ema)
    model_2_ema = nn.DataParallel(model_2_ema)

    initial_weights = load_checkpoint(args.init_1)
    copy_state_dict(initial_weights['state_dict'], model_1)
    copy_state_dict(initial_weights['state_dict'], model_1_ema)
    model_1_ema.module.classifier.weight.data.copy_(model_1.module.classifier.weight.data)

    initial_weights = load_checkpoint(args.init_2)
    copy_state_dict(initial_weights['state_dict'], model_2)
    copy_state_dict(initial_weights['state_dict'], model_2_ema)
    model_2_ema.module.classifier.weight.data.copy_(model_2.module.classifier.weight.data)

    for param in model_1_ema.parameters():
        param.detach_()
    for param in model_2_ema.parameters():
        param.detach_()

    return model_1, model_2, model_1_ema, model_2_ema


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global best_mAP

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    dataset_target = get_data(args.dataset_target, args.data_dir)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, 4 * args.batch_size, args.workers)
    cluster_loader = get_test_loader(dataset_target, args.height, args.width, 4 * args.batch_size, args.workers, testset=dataset_target.train)

    # Create model
    if args.arch.find("part") >= 0 and not args.flag_ca:
        assert False, "only use [network]_part if you use flag_ca"
    elif args.arch.find("part") < 0 and args.flag_ca:
        assert False, "If you turn on flag_ca, must using [network]_part"
    model_1, model_2, model_1_ema, model_2_ema = create_model(args)

    # Evaluator
    evaluator_1_ema = Evaluator(model_1_ema)
    evaluator_2_ema = Evaluator(model_2_ema)

    # evaluator_1_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)
    # evaluator_2_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)

    dict_f, _ = extract_features(model_1_ema, cluster_loader, print_freq=50)
    cf_1 = torch.stack(list(dict_f.values())).numpy()
    dict_f, _ = extract_features(model_2_ema, cluster_loader, print_freq=50)
    cf_2 = torch.stack(list(dict_f.values())).numpy()
    cf = (cf_1 + cf_2) / 2
    cf = normalize(cf, axis=1)
    del dict_f, cf_1, cf_2


    if args.fast_kmeans == args.dbscan == True: assert False, "only set --fast-kmeans or --dbscan or non call both options"
    if args.fast_kmeans:
        km = MiniBatchKMeans(n_clusters=args.num_clusters, max_iter=250, batch_size=300, init_size=3500).fit(cf)        
        model_1.module.classifier.weight.data.copy_(torch.from_numpy(normalize(km.cluster_centers_, axis=1)).float().cuda())
        model_2.module.classifier.weight.data.copy_(torch.from_numpy(normalize(km.cluster_centers_, axis=1)).float().cuda())
        model_1_ema.module.classifier.weight.data.copy_(torch.from_numpy(normalize(km.cluster_centers_, axis=1)).float().cuda())
        model_2_ema.module.classifier.weight.data.copy_(torch.from_numpy(normalize(km.cluster_centers_, axis=1)).float().cuda())
        centers = normalize(km.cluster_centers_, axis=1)
        target_label = km.labels_
    else:
        km = KMeans(n_clusters=args.num_clusters, random_state=args.seed,max_iter=200).fit(cf)
        model_1.module.classifier.weight.data.copy_(torch.from_numpy(normalize(km.cluster_centers_, axis=1)).float().cuda())
        model_2.module.classifier.weight.data.copy_(torch.from_numpy(normalize(km.cluster_centers_, axis=1)).float().cuda())
        model_1_ema.module.classifier.weight.data.copy_(torch.from_numpy(normalize(km.cluster_centers_, axis=1)).float().cuda())
        model_2_ema.module.classifier.weight.data.copy_(torch.from_numpy(normalize(km.cluster_centers_, axis=1)).float().cuda())
        centers = normalize(km.cluster_centers_, axis=1)
        target_label = km.labels_
    start_percentage = args.p
    del km

    def scheduler(t, T, p0, h=1.5):
        return p0 + 1 / h * math.log(1 + t / T * (pow(math.e, h*(1 - p0)) - 1))

    for epoch in range(args.epochs):
        ############################################
        #Resamples with probabilistic uncertainty
        pt = scheduler(epoch, args.epochs-1, start_percentage, h=1.5)
        print('Current epoch selects {:.4f} unlabeled data'.format(pt))
        train_loader_target, _, _ = get_train_loader(dataset_target,
                                                            args.height, args.width, args.batch_size, args.workers,
                                                            args.num_instances, iters, centers,target_label, cf, pt)
        del cf, centers
        # Setting Optimizer
        params = []
        for key, value in model_1.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
        for key, value in model_2.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
        optimizer = torch.optim.Adam(params)

        if args.flag_ca: 
            assert not args.dbscan, "U shouldnt use both --flag-ca and --dbscan at time, waiting next update"
            ###############################################
            #Calculate cross agreements on refined sample
            """
            Yoonki Cho, Woo Jae Kim, Seunghoon Hong, Sung-Eui Yoon. 28 Mar 2022
            url:https://arxiv.org/abs/2203.14675
            github:https://github.com/yoonkicho/pplr
            """
            ##########
            print("using cross agreement...")
            # calculate parts feature
            features_g, features_p = extract_all_features(model_1, model_2, train_loader_target)
            # Compute the cross-agreement
            score = compute_cross_agreement(features_g, features_p, k=args.k)
            del features_g, features_p
            
            print("Finetune by MMTT...")
            trainer = MMTwCaTrainer(model_1, model_2, model_1_ema, model_2_ema,
                             num_cluster=args.num_clusters, alpha=args.alpha, 
                             aals_epoch=args.aals_epoch, num_part=args.part, batch_size=args.batch_size)
            train_loader_target.new_epoch()
            trainer.train(epoch, train_loader_target, optimizer,
                      ce_soft_weight=args.soft_ce_weight, tri_soft_weight=args.soft_tri_weight,
                      print_freq=args.print_freq, train_iters=len(train_loader_target), 
                      aals_weight=args.beta,cross_agreements=score)

        else:
            trainer = MMTTrainer(model_1, model_2, model_1_ema, model_2_ema,
                             num_cluster=args.num_clusters, alpha=args.alpha)
            train_loader_target.new_epoch()
            trainer.train(epoch, train_loader_target, optimizer,
                      ce_soft_weight=args.soft_ce_weight, tri_soft_weight=args.soft_tri_weight,
                      print_freq=args.print_freq, train_iters=len(train_loader_target))
        
        del trainer
        def save_model(model_ema, is_best, best_mAP, mid):
            save_checkpoint({
                'state_dict': model_ema.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'model'+str(mid)+'_checkpoint.pth.tar'))
        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            if args.offline_test or (epoch==args.epochs-1):
                print("save model 4 offline test in epoch {}".format(epoch+1))
                save_model(model_1_ema,is_best=False,best_mAP=0.0,mid=(epoch+1)*10+1)
                save_model(model_2_ema,is_best=False,best_mAP=0.0,mid=(epoch+1)*10+2)
            else:
                mAP_1 = evaluator_1_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=False)
                mAP_2 = evaluator_2_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=False)
                is_best = (mAP_1>best_mAP) or (mAP_2>best_mAP)
                best_mAP = max(mAP_1, mAP_2, best_mAP)
                save_model(model_1_ema, (is_best and (mAP_1>mAP_2)), best_mAP, 1)
                save_model(model_2_ema, (is_best and (mAP_1<=mAP_2)), best_mAP, 2)

                print('\n * Finished epoch {:3d}  model no.1 mAP: {:5.1%} model no.2 mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP_1, mAP_2, best_mAP, ' *' if is_best else ''))


        ######################################
        dict_f, _ = extract_features(model_1_ema, cluster_loader, print_freq=50)
        cf_1 = torch.stack(list(dict_f.values())).numpy()
        dict_f, _ = extract_features(model_2_ema, cluster_loader, print_freq=50)
        cf_2 = torch.stack(list(dict_f.values())).numpy()
        cf = (cf_1 + cf_2) / 2
        cf = normalize(cf, axis=1)
        del dict_f, cf_1, cf_2
        # using select cf to update centers
        print('\n\t---- Clustering into new classes ----\n')  # num_clusters=500

        if args.dbscan:
            if epoch==0: cluster = DBSCAN(eps=args.dbscan_eps, min_samples=4, metric='precomputed', n_jobs=2)
            cf = torch.from_numpy(cf)
            pseudo_labels, num_class = generate_pseudo_dbscan(epoch=epoch, cf=cf, cluster=cluster, k1=args.dbscan_k1, k2=args.dbscan_k2)
            # generate new dataset with pseudo-labels
            num_outliers = 0
            new_cf = None
            pids = None
            for i, (fea, label) in enumerate(zip(cf, pseudo_labels)):
                pid = label.item()
                if pid >= num_class:  # append data except outliers
                    num_outliers += 1
                else:
                    new_cf = np.vstack((new_cf,fea)) if not new_cf is None else fea[None, :].detach().numpy()
                    pids = np.hstack((pids, pid)) if not pids is None else np.array([pid])

            # statistics of clusters and un-clustered instances
            print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances'.format(epoch, num_class,
                                                                                            num_outliers))
            # compute cluster centroids
            centers = []
            for pid in sorted(np.unique(pids)):  # loop all pids
                idxs_p = np.where(pids == pid)[0]
                centers.append(new_cf[idxs_p].mean(0))
            centers = np.array(centers)
            centers = normalize(centers, axis=1)
            target_label = pids
            cf = new_cf
            # model_1.module.classifier.weight.data[:num_class].copy_(centroids_g)
            # model_2.module.classifier.weight.data[:num_class].copy_(centroids_g)
            model_1.module.classifier.weight.data[:num_class].copy_(torch.from_numpy(centers).float().cuda())
            model_2.module.classifier.weight.data[:num_class].copy_(torch.from_numpy(centers).float().cuda())
            model_1_ema.module.classifier.weight.data[:num_class].copy_(torch.from_numpy(centers).float().cuda())
            model_2_ema.module.classifier.weight.data[:num_class].copy_(torch.from_numpy(centers).float().cuda())
        
        else:
            if args.fast_kmeans:
                km = MiniBatchKMeans(n_clusters=args.num_clusters, max_iter=250, batch_size=300, init_size=3000).fit(cf) 
                centers = normalize(km.cluster_centers_, axis=1)
                target_label = km.labels_
            else:
                km = KMeans(n_clusters=args.num_clusters, random_state=args.seed,max_iter=400).fit(cf)
                centers = normalize(km.cluster_centers_, axis=1)
                target_label = km.labels_
                    
            model_1.module.classifier.weight.data.copy_(torch.from_numpy(centers).float().cuda())
            model_2.module.classifier.weight.data.copy_(torch.from_numpy(centers).float().cuda())
            model_1_ema.module.classifier.weight.data.copy_(torch.from_numpy(centers).float().cuda())
            model_2_ema.module.classifier.weight.data.copy_(torch.from_numpy(centers).float().cuda())
            del km       
        # else:
        #     for id in range(args.num_clusters):
        #         indexs = select_pseudo_samples[np.where(select_pseudo_samples_labels==id)]
        #         if len(indexs)>0:
        #             centers[id] = np.mean(cf[indexs],0)
        

def extract_all_features(model_1, model_2, data_loader):
    model_1.eval()
    model_2.eval()

    features_g = OrderedDict()
    features_p = OrderedDict()


    with torch.no_grad():
        data_loader.new_epoch()
        fnames_list = []
        for i in range(len(data_loader)):
            imgs_1, imgs_2, fnames, _ = data_loader.next()
            inputs_1 = to_torch(imgs_1).cuda()
            inputs_2 = to_torch(imgs_2).cuda()
            if isinstance(model_1, nn.DataParallel):
                outputs_g, outputs_p = model_1.module.extract_all_features(inputs_1)
                outputs_g_, outputs_p_ = model_2.module.extract_all_features(inputs_2)
            else:
                outputs_g, outputs_p = model_1.extract_all_features(inputs_1)
                outputs_g_, outputs_p_ = model_1.extract_all_features(inputs_2)

            outputs_g = (outputs_g + outputs_g_)/2
            outputs_p = (outputs_p + outputs_p_)/2
            del outputs_p_, outputs_g_

            outputs_g, outputs_p = outputs_g.data.cpu(), outputs_p.data.cpu()
            for fname, output_g, output_p in zip(fnames, outputs_g, outputs_p):
                features_g[fname] = output_g
                features_p[fname] = output_p
            fnames_list += fnames
        
        features_g = torch.cat([features_g[f].unsqueeze(0) for f in fnames_list], 0)
        features_p = torch.cat([features_p[f].unsqueeze(0) for f in fnames_list], 0)
        return features_g, features_p



def compute_cross_agreement(features_g, features_p, k, search_option=0):
    print("Compute cross agreement score...")
    N, D, P = features_p.size()
    score = torch.FloatTensor()
    ranked_list_g = compute_ranked_list(features_g, k=k, search_option=search_option, verbose=True) #[N x 20]
    for i in range(P):
        ranked_list_p_i = compute_ranked_list(features_p[:, :, i], k=k, search_option=search_option, verbose=False)
        intersect_i = torch.FloatTensor([len(np.intersect1d(ranked_list_g[j], ranked_list_p_i[j])) for j in range(N)]) #[Nx1] -> scalar array
        union_i = torch.FloatTensor([len(np.union1d(ranked_list_g[j], ranked_list_p_i[j])) for j in range(N)])
        score_i = intersect_i / union_i
        score = torch.cat([score, score_i.unsqueeze(1)], dim=1) #[Nx3]
    return score



def generate_pseudo_dbscan(epoch, cf, cluster, k1=30, k2=6):
    """
    cf: center features
    """
    mat_dist = compute_jaccard_distance(cf, k1=k1, k2=k2)
    ids = cluster.fit_predict(mat_dist)
    num_ids = len(set(ids)) - (1 if -1 in ids else 0)

    labels = []
    outliers = 0
    for i, id in enumerate(ids):
        if id != -1:
            labels.append(id)
        else:
            labels.append(num_ids + outliers)
            outliers += 1

    return torch.Tensor(labels).long().detach(), num_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MMT Training")
    # dataset
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=0)
    parser.add_argument('--num-clusters', type=int, default=500)
    parser.add_argument('--height', type=int, default=256,
                        help="input height")
    parser.add_argument('--width', type=int, default=128,
                        help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--p', type=float, default=0.2)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.999)
    parser.add_argument('--moving-avg-momentum', type=float, default=0)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--soft-ce-weight', type=float, default=0.5)
    parser.add_argument('--soft-tri-weight', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--iters', type=int, default=400)
    # training configs
    parser.add_argument('--init-1', type=str, default='', metavar='PATH')
    parser.add_argument('--init-2', type=str, default='', metavar='PATH')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    # cross agreement configs
    parser.add_argument('--flag-ca', action='store_true',
                        help='using cross agreement')
    parser.add_argument('--part', type=int, default=3, help="number of part")
    parser.add_argument('--k', type=int, default=20,
                        help="hyperparameter for cross agreement score")
    parser.add_argument('--beta', type=float, default=0.5,
                        help="weighting parameter for part-guided label refinement")
    parser.add_argument('--aals-epoch', type=int, default=5,
                        help="starting epoch for agreement-aware label smoothing")

    #Secret
    parser.add_argument('--extra-bottleneck', action='store_true',
                        help='add a bottelneck before split into feature parts')   
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    #clustering
    parser.add_argument('--fast-kmeans', action='store_true',
                        help='using fast clustering with --fast_kmeans')
    parser.add_argument('--dbscan', action='store_true',
                        help='using dbscan instead of default kmean')
    parser.add_argument('--dbscan-k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--dbscan-k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--dbscan-eps', type=float, default=0.5,
                        help="distance threshold for DBSCAN")


    parser.add_argument('--offline_test', action='store_true',
                        help='offline test models')
    main()
