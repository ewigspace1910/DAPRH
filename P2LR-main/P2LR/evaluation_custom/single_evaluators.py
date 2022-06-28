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


def cmc(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False,
        n_threads=1):
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
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = [0]

    def cmc_thread(start_index, stop_index):
        for i in range(start_index, stop_index):
            # Filter out the same id and same camera
            valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                     (gallery_cams[indices[i]] != query_cams[i]))
            if separate_camera_set:
                # Filter out samples from same camera
                valid &= (gallery_cams[indices[i]] != query_cams[i])
            if not np.any(matches[i, valid]): continue
            if single_gallery_shot:
                repeat = 10
                gids = gallery_ids[indices[i][valid]]
                inds = np.where(valid)[0]
                ids_dict = defaultdict(list)
                for j, x in zip(inds, gids):
                    ids_dict[x].append(j)
            else:
                repeat = 1
            for _ in range(repeat):
                if single_gallery_shot:
                    # Randomly choose one instance for each id
                    sampled = (valid & _unique_sample(ids_dict, len(valid)))
                    index = np.nonzero(matches[i, sampled])[0]
                else:
                    index = np.nonzero(matches[i, valid])[0]
                delta = 1. / (len(index) * repeat)
                for j, k in enumerate(index):
                    if k - j >= topk: break
                    if first_match_break:
                        ret[k - j] += 1
                        break
                    ret[k - j] += delta
            num_valid_queries[0] += 1
    if n_threads > 1:
        n_range = np.linspace(0, m, n_threads + 1).astype(int)
        threads = [threading.Thread(target=cmc_thread, args=(n_range[i], n_range[i+1],)) for i in range(n_threads)]
        [thread.start() for thread in threads]
        [thread.join() for thread in threads]
    else:
        cmc_thread(0, m)

    if num_valid_queries[0] == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries[0]


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

def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None, workers=0,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False, top_k = 10, label_clusters=None):
    if query is not None and gallery is not None:
        query_path = [pid for  pid,_ , _ in query]
        gallery_path = [pid for pid,_, _ in gallery]
#        query_ids = [pid for _, pid, _ in query]
#        gallery_ids = [pid for _, pid, _ in gallery]
#        query_cams = [cam for _, _, cam in query]
#        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    indices = np.argsort(distmat, axis=1)
    if label_clusters is None:
        target_label_name = "null.txt"
        label_clusters = [-1] * len(gallery_features)
    else:
        target_label_name = "target_label.txt"
    
    with open("result.txt", 'wt') as f, open("dismat.txt", "wt") as d, open(target_label_name, "wt") as l:
        for i, query in enumerate(query_path):
            f.write(query+":")
            d.write(query+":")
            l.write(query+":")
            for index in indices[i][:top_k]:
               f.write(gallery_path[index]+";")
               d.write(str(float(distmat[i][index].sum())) + ";")
               l.write(str(label_clusters[index])+";")
            f.write("\n")
            d.write("\n")
            l.write("\n")
    print("result matches was store in result.txt")
    return indices


class Evaluator(object):
    def __init__(self, model, args):
        super(Evaluator, self).__init__()
        self.model = model
        self.args = args

    def evaluate(self, data_loader, query, gallery, metric=None, cmc_flag=False, rerank=False, pre_features=None, use_kmean=False):
        if (pre_features is None):
            features, _ = extract_features(self.model, data_loader)
        else:
            features = pre_features
        
        target_label = None
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery, metric=metric)
        
        if use_kmean:
            print("using kmeans")
            assert self.args.clusters > 0, "num_clusters arg must be larger than 0"
            cf = normalize(gallery_features, axis=1)
            km = KMeans(n_clusters=self.args.clusters, random_state=self.args.seed, n_jobs=4,max_iter=400).fit(cf)
            centers = normalize(km.cluster_centers_, axis=1)
            target_label = km.labels_              
        
        #calculate dismat
        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, 
                cmc_flag=cmc_flag, workers=8, label_clusters=target_label)
        
        
        if rerank:
            print('Applying person re-ranking ...')
            distmat_qq, _, _ = pairwise_distance(features, query, query, metric=metric)
            distmat_gg, _, _ = pairwise_distance(features, gallery, gallery, metric=metric)
            distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
            reusults =  evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, 
                cmc_flag=cmc_flag, label_clusters=target_label)      
          

        return results
