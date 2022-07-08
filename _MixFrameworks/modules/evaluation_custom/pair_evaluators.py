#edit for private own dataset

from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import numpy as np
import torch

from sklearn.cluster import KMeans

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
from ..utils import to_numpy

def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask

  
    
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

def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None, workers=0,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False, top_k = 10):
    if query is not None and gallery is not None:
        query_path = [pid for  pid,_ , _ in query] #store path for writing
        gallery_path = [pid for pid,_, _ in gallery]

    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)
    
    indices = np.argsort(distmat, axis=1)
    # Compute mean AP
    with open("result.txt", 'wt') as f, open("dismat.txt", "wt") as d:
        for i, query in enumerate(query_path):
            f.write(query+":")
            d.write(query+":")
            for index in indices[i][:top_k]:
               f.write(gallery_path[index]+";")
               d.write(str(float(distmat[i][index].sum())) + ";")
            f.write("\n")
            d.write("\n")
    print("result matches was store in result.txt")
    return indices, dismat, gallery_path, query_path


class Evaluator(object):
    "Edited Evaluator from orginal Evaluator for ensemble solutions"
    def __init__(self, model1, model2, args):
        super(Evaluator, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.args = args

    def evaluate(self, data_loader, query, gallery, metric=None, cmc_flag=False, rerank=False, pre_features1=None, pre_features2=None, use_kmean=False):
        if (pre_features1 is None or pre_features2 is None):
            features1, _ = extract_features(self.model1, data_loader)
            features2, _ = extract_features(self.model2, data_loader)
        else:
            features1 = pre_features1
            features2 = pre_features2
            
        distmat1, query_features, gallery_features = pairwise_distance(features1, query, gallery, metric=metric)
        distmat2, _, _                             = pairwise_distance(features2, query, gallery, metric=metric)
        
        mean_distmat = (distmat1 + distmat2) / 2
        results = evaluate_all(query_features, gallery_features, mean_distmat, query=query, gallery=gallery, cmc_flag=cmc_flag, workers=4)
        
        
        if rerank:
            print('Applying person re-ranking ...')
            distmat_qq1, _, _ = pairwise_distance(features1, query, query, metric=metric)
            distmat_gg1, _, _ = pairwise_distance(features1, gallery, gallery, metric=metric)
        
            distmat_qq2, _, _ = pairwise_distance(features2, query, query, metric=metric)
            distmat_gg2, _, _ = pairwise_distance(features2, gallery, gallery, metric=metric)
        
            distmat_qq = (distmat_qq1 + distmat_qq2) / 2
            distmat_gg = (distmat_gg1 + distmat_gg2) / 2
        
            distmat = re_ranking(mean_distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
            reusults =  evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)
            
        if use_kmean:
            print("using kmeans")
            assert self.args.num_clusters > 0, "num_clusters arg must be larger than 0"
            cf = normalize((features1+features2)/2, axis=1)
            km = KMeans(n_clusters=self.args.clusters, random_state=self.args.seed, n_jobs=8,max_iter=300).fit(cf)
            centers = normalize(km.cluster_centers_, axis=1)
            target_label = km.labels_

           
        return results
