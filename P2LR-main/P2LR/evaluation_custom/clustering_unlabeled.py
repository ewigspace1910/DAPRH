#edit for private own dataset

from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import numpy as np
import torch

#from .evaluation_metrics import cmc, mean_ap
from ..feature_extraction import extract_cnn_feature
from ..utils.meters import AverageMeter
from ..utils.rerank import re_ranking

from collections import defaultdict
import threading
import multiprocessing

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

from ..utils import to_numpy

def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask

def mean_ap(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None, n_threads=1):
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    
    
    #Find top k-nearest
    return indices, matches
   
    
##############################################################
    
def extract_features(model, data_loader, print_freq=10, metric=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels

def pairwise_distance(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x.numpy(), y.numpy()

def clustering_all(distmat_gg, gallery=None, top_k = 15, label_clusters=None, hardmore=False, minimum_sample=2):
    if gallery is not None:
        gallery_path = [pid for pid,_, _ in gallery]
    else:
        print("error 404")
        exit()
    
    clusters = {i:[] for i in set(label_clusters)}
    for index, cluster in enumerate(label_clusters):
        clusters[cluster].append(index)
    
    new_clusters = {} #shape path
    indices = np.argsort(distmat_gg, axis=1)
    
    for cluster in clusters:
        new_clusters[cluster] = []
            
        #match = np.zeros((len(clusters[cluster]), len(clusters[cluster])))
        tmp_index = [i for i in clusters[cluster]]
        for i in tmp_index:
            tmp_cp =  tmp_index.copy()
            tmp_cp.remove(i) 
            distvec_i = indices[i]  #rank k-nearest of i
            #with j<>i must in top k of i => j (= cluster(i)
            sorter_dismat_i = np.argsort(distvec_i) 
            rank_tmp_in_dv_i = sorter_dismat_i[np.searchsorted(distvec_i, tmp_cp, sorter=sorter_dismat_i)] #rank other k-point to i-point
            rank_tmp_in_dv_i = rank_tmp_in_dv_i < top_k
            
            num_j_accept_i = len(tmp_cp)
            if hardmore:
                print("using hard sample")
                #i must in top k/2 of j => i (= cluster(j)  ~  significantly same reranking
                for j in tmp_cp:
                    distvec_j = indices[j]
                    if np.where(distvec_j == i)[0][0] > top_k // 2: num_j_accept_i-= 1             
                       
            #print(rank_tmp_in_dv_i)
            constrain_1 = len(tmp_cp) // 5 * 4   #empritical belief :>>
            constrain_2 = len(tmp_cp) // 3 * 2       #empritical belief :>>
            if np.sum(rank_tmp_in_dv_i.numpy()) > constrain_1  and num_j_accept_i  > constrain_2: 
                new_clusters[cluster].append(i)         
            
    with open("cluster_result.txt", 'wt') as f:
        for cluster in new_clusters:
            if len(new_clusters[cluster]) < minimum_sample: continue
            f.write("{}:".format(cluster))
            for index in new_clusters[cluster]:
               f.write(gallery_path[index]+";")
            f.write("\n")
    
    print("result matches was store in result.txt")
    return indices


class DSCluster(object):
    def __init__(self, model, args):
        super(DSCluster, self).__init__()
        self.model = model
        self.args = args

    def evaluate(self, data_loader, query, gallery, metric=None, rerank=False):
        features, _ = extract_features(self.model, data_loader)

        #distmat, query_features, gallery_features = pairwise_distance(features, query, gallery, metric=metric)
        distmat_gg, _, gallery_features = pairwise_distance(features, gallery, gallery, metric=metric)

        print("run kmeans")
        assert self.args.clusters > 0, "num_clusters arg must be larger than 0"
        cf = normalize(gallery_features, axis=1)
        km = KMeans(n_clusters=self.args.clusters, random_state=self.args.seed, n_jobs=4,max_iter=400).fit(cf)
        centers = normalize(km.cluster_centers_, axis=1)
        target_label = km.labels_              
        
        #calculate dismat
        results = clustering_all(distmat_gg, gallery=gallery, label_clusters=target_label, 
                                hardmore=args.hard_sample, minimum_sample=3)
        
        
        if rerank:
            print('Applying person re-ranking ...')
            distmat = re_ranking(distmat_gg.numpy(), distmat_gg.numpy(), distmat_gg.numpy())
            results = clustering_all(distmat_gg, gallery=gallery, label_clusters=target_label)   
          

        return results
