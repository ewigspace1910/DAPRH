from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from .evaluation_metrics import accuracy
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy, KLDivLoss, CrossEntropyLabelSmoothFilterNoise
from .loss.extra import AALS, PGLR
from .utils.meters import AverageMeter


class PreTrainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(PreTrainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()

    def train(self, epoch, data_loader_source, data_loader_target, optimizer, train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)
            s_features, s_cls_out = self.model(s_inputs)
                        
            # target samples: only forward
            #t_inputs, _ = self._parse_data(target_inputs)
            #t_features, _ = self.model(t_inputs)

            # backward main #
            loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out, targets)
            loss = loss_ce + loss_tr #could improve it?

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec

class MMTTrainer(object):
    def __init__(self, model_1, model_2,
                       model_1_ema, model_2_ema, num_cluster=500, alpha=0.999):
        super(MMTTrainer, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.num_cluster = num_cluster

        self.model_1_ema = model_1_ema
        self.model_2_ema = model_2_ema
        self.alpha = alpha

        # self.criterion_ce = CrossEntropyLabelSmoothFilterNoise(num_cluster).cuda()
        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        # self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_ce_soft = KLDivLoss().cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()

    def train(self, epoch, data_loader_target,
            optimizer, ce_soft_weight=0.5, tri_soft_weight=0.5, print_freq=1, train_iters=200):
        self.model_1.train()
        self.model_2.train()
        self.model_1_ema.train()
        self.model_2_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = [AverageMeter(),AverageMeter()]
        losses_tri = [AverageMeter(),AverageMeter()]
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        precisions = [AverageMeter(),AverageMeter()]

        end = time.time()
        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)
            # process inputs
            inputs_1, inputs_2, targets = self._parse_data(target_inputs)
            # forward
            f_out_t1, p_out_t1 = self.model_1(inputs_1) # features and predictions
            f_out_t2, p_out_t2 = self.model_2(inputs_2)
            p_out_t1 = p_out_t1[:,:self.num_cluster]
            p_out_t2 = p_out_t2[:,:self.num_cluster]

            f_out_t1_ema, p_out_t1_ema = self.model_1_ema(inputs_1)
            f_out_t2_ema, p_out_t2_ema = self.model_2_ema(inputs_2)
            p_out_t1_ema = p_out_t1_ema[:,:self.num_cluster]
            p_out_t2_ema = p_out_t2_ema[:,:self.num_cluster]

            loss_ce_1 = self.criterion_ce(p_out_t1, targets)
            loss_ce_2 = self.criterion_ce(p_out_t2, targets)
            # loss_ce_1 = self.criterion_ce(p_out_t1, targets, epoch)
            # loss_ce_2 = self.criterion_ce(p_out_t2, targets, epoch)

            loss_tri_1 = self.criterion_tri(f_out_t1, f_out_t1, targets)
            loss_tri_2 = self.criterion_tri(f_out_t2, f_out_t2, targets)

            # loss_ce_soft = self.criterion_ce_soft(p_out_t1, p_out_t2_ema) + self.criterion_ce_soft(p_out_t2, p_out_t1_ema)
            # loss_tri_soft = self.criterion_tri_soft(f_out_t1, f_out_t2_ema, targets) + \
            #                 self.criterion_tri_soft(f_out_t2, f_out_t1_ema, targets)
            # Mean Mean model
            p_out_ema = (p_out_t1_ema+p_out_t2_ema)/2
            f_out_ema = (f_out_t1_ema+f_out_t2_ema)/2
            loss_ce_soft = self.criterion_ce_soft(p_out_t1, p_out_ema) + self.criterion_ce_soft(p_out_t2, p_out_ema)
            loss_tri_soft = self.criterion_tri_soft(f_out_t1, f_out_ema, targets) + \
                            self.criterion_tri_soft(f_out_t2, f_out_ema, targets)
            loss = (loss_ce_1 + loss_ce_2)*(1-ce_soft_weight) + \
                     (loss_tri_1 + loss_tri_2)*(1-tri_soft_weight) + \
                     loss_ce_soft*ce_soft_weight + loss_tri_soft*tri_soft_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch*len(data_loader_target)+i)
            self._update_ema_variables(self.model_2, self.model_2_ema, self.alpha, epoch*len(data_loader_target)+i)

            prec_1, = accuracy(p_out_t1.data, targets.data)
            prec_2, = accuracy(p_out_t2.data, targets.data)

            losses_ce[0].update(loss_ce_1.item())
            losses_ce[1].update(loss_ce_2.item())
            losses_tri[0].update(loss_tri_1.item())
            losses_tri[1].update(loss_tri_2.item())
            losses_ce_soft.update(loss_ce_soft.item())
            losses_tri_soft.update(loss_tri_soft.item())
            precisions[0].update(prec_1[0])
            precisions[1].update(prec_2[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} / {:.3f}\t'
                      'Loss_tri {:.3f} / {:.3f}\t'
                      'Loss_ce_soft {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'Prec {:.2%} / {:.2%}\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce[0].avg, losses_ce[1].avg,
                              losses_tri[0].avg, losses_tri[1].avg,
                              losses_ce_soft.avg, losses_tri_soft.avg,
                              precisions[0].avg, precisions[1].avg))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        imgs_1, imgs_2, _, pids = inputs
        inputs_1 = imgs_1.cuda()
        inputs_2 = imgs_2.cuda()
        targets = pids.cuda()
        return inputs_1, inputs_2, targets

class MMTwCaTrainer(object):
    def __init__(self, model_1, model_2,
                       model_1_ema, model_2_ema, num_cluster=500, alpha=0.999, 
                       aals_epoch=5, num_part=3, batch_size=None) :
        super(MMTwCaTrainer, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.num_cluster = num_cluster

        self.model_1_ema = model_1_ema
        self.model_2_ema = model_2_ema
        self.alpha = alpha

        #pplr loss
        self.num_part = num_part
        self.aals_epoch = aals_epoch
        assert batch_size != None, "set batch_size for MMTT"
        self.bs = batch_size
        # self.criterion_ce = CrossEntropyLabelSmoothFilterNoise(num_cluster).cuda()
        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_ce_soft = KLDivLoss().cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()
        self.criterion_aals = AALS().cuda()
        self.criterion_pglr = PGLR(lam=0.5).cuda()

    def train(self, epoch, data_loader_target,
            optimizer, ce_soft_weight=0.5, tri_soft_weight=0.5, print_freq=1, 
            train_iters=200, cross_agreements=None, aals_weight=0.5):
        self.model_1.train()
        self.model_2.train()
        self.model_1_ema.train()
        self.model_2_ema.train()
        assert cross_agreements != None, "scores is None"
        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_sum = AverageMeter()
        precisions = [AverageMeter(),AverageMeter()]

        end = time.time()

        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)
            # process inputs
            inputs_1, inputs_2 ,targets = self._parse_data(target_inputs)
            ca = cross_agreements[i*self.bs:i*self.bs+self.bs].cuda()
            # forward
            g_f_out_t1, p_f_out_t1, g_p_out_t1, p_p_out_t1 = self.model_1(inputs_1) # features and predictions of global and 3 part
            g_f_out_t2, p_f_out_t2, g_p_out_t2, p_p_out_t2 = self.model_2(inputs_2)

            g_p_out_t1 = g_p_out_t1[:,:self.num_cluster] #????????????
            g_p_out_t2 = g_p_out_t2[:,:self.num_cluster]

            g_f_out_t1_ema, p_f_out_t1_ema, g_p_out_t1_ema, _= self.model_1_ema(inputs_1)
            g_f_out_t2_ema, p_f_out_t2_ema, g_p_out_t2_ema, _ = self.model_2_ema(inputs_2)
            
            g_p_out_t1_ema = g_p_out_t1_ema[:,:self.num_cluster]
            g_p_out_t2_ema = g_p_out_t2_ema[:,:self.num_cluster]

            # Mean Mean model
            g_p_out_ema = (g_p_out_t1_ema+g_p_out_t2_ema)/2
            g_f_out_ema = (g_f_out_t1_ema+g_f_out_t2_ema)/2
            p_f_out_ema = (p_f_out_t1_ema+p_f_out_t2_ema)/2

            
            ##########################
            #LOSSES
            ##
        
            #ID loss
            # loss_ce_1 = self.criterion_ce(g_p_out_t1, targets)
            # loss_ce_2 = self.criterion_ce(g_p_out_t2, targets)
            loss_ce_1 = self.criterion_pglr(g_p_out_t1, p_p_out_t1, targets, ca)
            loss_ce_2 = self.criterion_pglr(g_p_out_t1, p_p_out_t2, targets, ca)
            
            #Triplet loss for global
            loss_tri_1 = self.criterion_tri(g_f_out_t1, g_f_out_t1, targets)
            loss_tri_2 = self.criterion_tri(g_f_out_t2, g_f_out_t2, targets)
            
            #aals loss for local feature (local id loss)
            loss_pce = 0.
            if self.num_part > 0:
                if epoch >= self.aals_epoch:
                    for part in range(self.num_part):
                        loss_pce += self.criterion_aals(p_p_out_t1[:, :, part], targets, ca[:, part])
                        loss_pce += self.criterion_aals(p_p_out_t2[:, :, part], targets, ca[:, part])
                else:
                    for part in range(self.num_part):
                        loss_pce += self.criterion_ce(p_p_out_t1[:, :, part], targets)
                        loss_pce += self.criterion_ce(p_p_out_t2[:, :, part], targets)
                loss_pce /= self.num_part
            

            #loss distillation loss(id and feature) between student and teacher
            loss_ce_soft = self.criterion_ce_soft(g_p_out_t1, g_p_out_ema) + self.criterion_ce_soft(g_p_out_t2, g_p_out_ema)
            loss_tri_soft = self.criterion_tri_soft(g_f_out_t1, g_f_out_ema, targets) + \
                            self.criterion_tri_soft(g_f_out_t2, g_f_out_ema, targets)
            
            loss_tri_soft_local = 0
            if self.num_part > 0:
                for part in range(self.num_part):
                    loss_tri_soft_local += self.criterion_tri_soft(p_f_out_t1[:, :, part], p_f_out_ema[:, :, part], targets)
                    loss_tri_soft_local += self.criterion_tri_soft(p_f_out_t2[:, :, part], p_f_out_ema[:, :, part], targets)
                loss_tri_soft_local /= self.num_part

            #total loss
            loss = (loss_ce_1 + loss_ce_2)*(1-ce_soft_weight) + \
                    (loss_tri_1 + loss_tri_2)*(1-tri_soft_weight) + \
                    loss_ce_soft*ce_soft_weight + loss_tri_soft*tri_soft_weight +\
                    aals_weight * loss_pce + aals_weight * loss_tri_soft_local

            
            #optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch*len(data_loader_target)+i)
            self._update_ema_variables(self.model_2, self.model_2_ema, self.alpha, epoch*len(data_loader_target)+i)

            prec_1, = accuracy(g_p_out_t1.data, targets.data)
            prec_2, = accuracy(g_p_out_t2.data, targets.data)

            losses_sum.update(loss)
            precisions[0].update(prec_1[0])
            precisions[1].update(prec_2[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'TOTAL LOSS {:.3f}\t'
                      'Prec {:.2%} / {:.2%}\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_sum.avg,
                              precisions[0].avg, precisions[1].avg))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        imgs_1, imgs_2, _, pids = inputs
        inputs_1 = imgs_1.cuda()
        inputs_2 = imgs_2.cuda()
        targets = pids.cuda()
        return inputs_1, inputs_2, targets

