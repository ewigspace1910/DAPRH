from __future__ import absolute_import
from collections import defaultdict
import threading
import multiprocessing

import numpy as np
from sklearn.metrics import average_precision_score

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
    aps = []
    # Compute AP for each query using multi threads
    n_threads = 1
    if n_threads > 1:
        def run_thread(start_index, stop_index):
            for i in range(start_index, stop_index):
                # Filter out the same id and same camera
                valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                         (gallery_cams[indices[i]] != query_cams[i]))
                y_true = matches[i, valid]
                y_score = -distmat[i][indices[i]][valid]
                if not np.any(y_true): 
                    continue
                aps.append(average_precision_score(y_true, y_score))

        n_range = np.linspace(0, m, n_threads+1).astype(int)
        threads = [threading.Thread(target=run_thread, args=(n_range[i], n_range[i+1])) for i in range(n_threads)]
        [thread.start() for thread in threads]
        [thread.join() for thread in threads]
    else: #single thread
        for i in range(m):
            # Filter out the same id and same camera
            valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                     (gallery_cams[indices[i]] != query_cams[i]))
            y_true = matches[i, valid]
            y_score = -distmat[i][indices[i]][valid]
        
            if not np.any(y_true): 
                
                continue

            aps.append(average_precision_score(y_true, y_score))


    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)



