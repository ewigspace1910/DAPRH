from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import numpy as np
import torch

from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking


def extract_features(model, data_loader, metric=None,**kwargs):
    model.eval()
    features = OrderedDict()
    labels = OrderedDict()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            outputs = extract_cnn_feature(model, imgs)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid
    return features, labels

def extract_features_joint(model, data_loader, metric=None,**kwargs):
    model.eval()
    features = OrderedDict()
    part_features = OrderedDict()
    labels = OrderedDict()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            outputs, output_parts = extract_cnn_feature(model, imgs, joint=True)
            for fname, output,  part, pid in zip(fnames, outputs, output_parts, pids):
                features[fname] = output
                part_features[fname] = part
                labels[fname] = pid
    return features, part_features



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
                 cmc_topk=(1, 5, 10), cmc_flag=False, logger=None):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    try:
        mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams, n_threads=workers)
    except:
        print("Cant calculate mAP")
        mAP = 0


    if (not cmc_flag):
        logger.validatinglog(mAP=mAP)
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)
                }
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, n_threads=workers, **params)
                  for name, params in cmc_configs.items()}

    # print('CMC Scores:')
    # for k in cmc_topk:
    #     print('  top-{:<4}{:12.1%}'
    #           .format(k,
    #                   cmc_scores['market1501'][k-1]))
    if not logger is None: logger.validatinglog(mAP=mAP,top_k=cmc_scores['market1501'], cmc_topk=cmc_topk)
    return cmc_scores['market1501'][0], mAP


class Evaluator(object):
    def __init__(self, model, logger=None):
        super(Evaluator, self).__init__()
        self.model = model
        self.logger = logger
        assert not logger is None, "Must pass logger into evaluator"

    def evaluate(self, data_loader, query, gallery, metric=None, cmc_flag=False, rerank=False, pre_features=None):
        if (pre_features is None):
            features, _ = extract_features(self.model, data_loader)
        else:
            features = pre_features
        # print(features)
        # print(query)
        
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery, metric=metric)
        results = evaluate_all(query_features, gallery_features, distmat, 
                    query=query, gallery=gallery, cmc_flag=cmc_flag, 
                    workers=8, logger=self.logger)
        if (not rerank):
            return results

        self.logger.log(msg='Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query, metric=metric)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery, metric=metric)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, 
                    query=query, gallery=gallery, cmc_flag=cmc_flag, logger=self.logger)


class EvaluatorEns(object):
    def __init__(self, model, beta, logger=None):
        super(EvaluatorEns, self).__init__()
        self.model1, self.model2 = model
        self.beta = beta
        self.logger = logger
        logger.log(f"-->beta == {beta}")
        assert not logger is None, "Must pass logger into evaluator"

    def evaluate(self, data_loader, query, gallery, metric=None, cmc_flag=False, rerank=False, pre_features=None):
        if (pre_features is None):
            features1, _ = extract_features(self.model1, data_loader)
            features2, _ = extract_features(self.model2, data_loader)
        else:
            features = pre_features
        
        distmat1, query_features, gallery_features = pairwise_distance(features1, query, gallery, metric=metric)
        distmat2, _, _                             = pairwise_distance(features2, query, gallery, metric=metric)
        distmat  = (1-self.beta)*(distmat1) + self.beta * distmat2

        results = evaluate_all(query_features, gallery_features, distmat, 
                    query=query, gallery=gallery, cmc_flag=cmc_flag, 
                    workers=8, logger=self.logger)
        if (not rerank):
            return results

        self.logger.log(msg='Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query, metric=metric)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery, metric=metric)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, 
                    query=query, gallery=gallery, cmc_flag=cmc_flag, logger=self.logger)

class EvaluatorJoint(object):
    def __init__(self, model, beta=0.5, logger=None):
        super(EvaluatorJoint, self).__init__()
        self.model = model
        self.beta = beta
        logger.log(f"-->beta == {beta}")
        self.logger = logger
        assert not logger is None, "Must pass logger into evaluator"

    def evaluate(self, data_loader, query, gallery, metric=None, cmc_flag=False, rerank=False, pre_features=None):
        if (pre_features is None):
            features, part_features = extract_features_joint(self.model, data_loader)
        else:
            raise
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery, metric=metric)
        distmat_part, _, _ = pairwise_distance(part_features, query, gallery, metric=metric)
        # distmat = (distmat + distmat_part)
        distmat = (1-self.beta) * distmat + self.beta * distmat_part
        results = evaluate_all(query_features, gallery_features, distmat, 
                    query=query, gallery=gallery, cmc_flag=cmc_flag, 
                    workers=8, logger=self.logger)
        if (not rerank):
            return results

        self.logger.log(msg='Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query, metric=metric)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery, metric=metric)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, 
                    query=query, gallery=gallery, cmc_flag=cmc_flag, logger=self.logger)
