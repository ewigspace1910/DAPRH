from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, to_pil_image

from .evaluation_metrics import accuracy
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy
from .loss import KLDivLoss, UET
from .loss import PartAveragedTripletLoss, CenterTripletLoss
from .utils.meters import AverageMeter
from .utils.memory import clean_cuda


#Pretrainer only for real image
class PreTrainer(object):

    def __init__(self, model, num_classes, margin=0.0, ce_epsilon=0.1, model_type="resnet", **kwargs):
        self.__typemodel ={
                "resnet" : self._forward_loss,
                "resnetbpart": self._forward_loss_2
                }
        super(PreTrainer, self).__init__()
        print("Normal Pretrainer")
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes, epsilon = ce_epsilon).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        try: self.__forward = self.__typemodel[model_type]
        except:                 
            print(model_type)
            raise ImportError(name="Not support that type of backbone")

    def train(self, epoch, data_loader_source, optimizer, 
                train_iters=200, print_freq=1, logger=None, **kwargs):
        self.model.train()

        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()

        for i in range(train_iters):
            source_inputs = data_loader_source.next()
            s_inputs, targets = self._parse_data(source_inputs)
            s_features, s_cls_out = self.model(s_inputs)
            loss_ce, loss_tr, prec1 = self.__forward(s_features, s_cls_out, targets)
            loss = loss_ce + loss_tr #could improve it?
            
            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((i + 1) % print_freq == 0):
                logger.traininglog(epoch=epoch, i=i+1, iters=train_iters,
                    avgloss=loss, 
                    loss_ce=f"{losses_ce.val:.3f}({losses_ce.avg:.3f})",
                    loss_tr=f"{losses_tr.val:.3f}({losses_tr.avg:.3f})",
                    prec=f"{precisions.val:.2%}({precisions.avg:.2%})")

    def _parse_data(self, inputs):
        imgs, _, pids, _, _= inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward_loss(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec

    def _forward_loss_2(self, s_features, s_outputs, targets):
        [x, part_up, part_down], [prob, prob_part_up, prob_part_down] = s_features, s_outputs
        loss_ce = self.criterion_ce(prob, targets) + self.criterion_ce(prob_part_up, targets) + self.criterion_ce(prob_part_down, targets)
        
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(x, x, targets) + self.criterion_triple(part_up, part_up, targets) + self.criterion_triple(part_down, part_down, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(x, targets) + self.criterion_triple(part_up, targets) + self.criterion_triple(part_down, targets)
        prec, = accuracy(prob.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec

#Pretrainer with both fake and real images
class PreTrainerwSynImgs(object):

    def __init__(self, model, num_classes, margin=0.0, ce_epsilon=0.1, model_type="resnet", lam=1., **kwargs):
        self.__typemodel ={
                "resnet" : self._forward_loss,
                "resnetbpart": self._forward_loss_2
                }
        super(PreTrainerwSynImgs, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes, epsilon = ce_epsilon).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        self.lamda = lam
        try: self.__forward = self.__typemodel[model_type]
        except:                 
            print(model_type)
            raise ImportError(name="Not support that type of backbone")

    def train(self, epoch, data_loader_source, optimizer, 
                train_iters=200, print_freq=1, logger=None, **kwargs):
        self.model.train()

        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()

        for i in range(train_iters):
            source_inputs = data_loader_source.next()
            s_inputs, targets, isreal = self._parse_data(source_inputs)
            s_features, s_cls_out = self.model(s_inputs)
            loss_ce, loss_tr, prec1 = self.__forward(s_features, s_cls_out, targets)
            loss = loss_ce + loss_tr #could improve it?
            # print(loss.shape, isreal.shape)
            # if not isreal:
            #     loss *= self.lamda
            
            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((i + 1) % print_freq == 0):
                logger.traininglog(epoch=epoch, i=i+1, iters=train_iters,
                    avgloss=loss, 
                    loss_ce=f"{losses_ce.val:.3f}({losses_ce.avg:.3f})",
                    loss_tr=f"{losses_tr.val:.3f}({losses_tr.avg:.3f})",
                    prec=f"{precisions.val:.2%}({precisions.avg:.2%})")

    def _parse_data(self, inputs):
        imgs, _, pids, _, isreal= inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets, isreal

    def _forward_loss(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec

    def _forward_loss_2(self, s_features, s_outputs, targets):
        [x, part_up, part_down], [prob, prob_part_up, prob_part_down] = s_features, s_outputs
        loss_ce = self.criterion_ce(prob, targets) + self.criterion_ce(prob_part_up, targets) + self.criterion_ce(prob_part_down, targets)
        
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(x, x, targets) + self.criterion_triple(part_up, part_up, targets) + self.criterion_triple(part_down, part_down, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(x, targets) + self.criterion_triple(part_up, targets) + self.criterion_triple(part_down, targets)
        prec, = accuracy(prob.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec


#Pretrainer only for real image
class PreTrainerwDIM(object):

    def __init__(self, model, num_classes, margin=0.0, ce_epsilon=0.1, model_type="resnet", disnet=None, lam=1, **kwargs):
        self.__typemodel ={
                "resnet" : self._forward_loss,
                "resnetbpart": self._forward_loss_2
                }
        super(PreTrainerwDIM, self).__init__()
        print("init_PretrainerwDIM")
        self.model = model
        self.disnet = disnet
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes, epsilon = ce_epsilon).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        try: self.__forward = self.__typemodel[model_type]
        except:                 
            print(model_type)
            raise ImportError(name="Not support that type of backbone")
        self.ALoss = nn.MSELoss().cuda()
        self.lamda = lam

    def train(self, epoch, data_loader_source, optimizer,  optimizerD, target_loader_source,
                train_iters=200, print_freq=1, logger=None, **kwargs):
        self.model.train()
        self.disnet.train()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        losses_limD = AverageMeter()
        losses_limM = AverageMeter()

        for i in range(train_iters):
            source_inputs = data_loader_source.next()
            s_inputs, targets = self._parse_data(source_inputs)
            s_features, s_cls_out = self.model(s_inputs)
            loss_ce, loss_tr, prec1 = self.__forward(s_features, s_cls_out, targets)
            loss = loss_ce + loss_tr #could improve it?
            
            t_inputs, _, _, _, _ = target_loader_source.next()
            t_features, _ = self.model(t_inputs.cuda())   
            #train model with netD
            loss_lim = 0
            if epoch >= 1:
                D_duke = self.disnet(t_features)
                D_market = self.disnet(s_features)
                loss_lim = (self.ALoss(D_duke, torch.ones_like(D_duke)/2.) + self.ALoss(D_market, torch.ones_like(D_market))/2.)
                loss += loss_lim * self.lamda

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # train discriminator
            D_duke = self.disnet(t_features.detach())
            D_market = self.disnet(s_features.detach())
            d_loss = self.ALoss(D_market, torch.ones_like(D_market)) + self.ALoss(D_duke, torch.zeros_like(D_duke))
            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            losses_limM.update(loss_lim.item() if loss_lim > 0 else 0)
            losses_limD.update(d_loss.item() if d_loss > 0 else 0)

            if ((i + 1) % print_freq == 0):
                logger.traininglog(epoch=epoch, i=i+1, iters=train_iters,
                    avgloss=loss, 
                    loss_ce=f"{losses_ce.val:.3f}({losses_ce.avg:.3f})",
                    loss_tr=f"{losses_tr.val:.3f}({losses_tr.avg:.3f})",
                    loss_G=f"{losses_limM.val:.3f}({losses_limM.avg:.3f})",
                    loss_D=f"{losses_limD.val:.3f}({losses_limD.avg:.3f})")

    def _parse_data(self, inputs):
        imgs, _, pids, _, _= inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward_loss(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec

    def _forward_loss_2(self, s_features, s_outputs, targets):
        [x, part_up, part_down], [prob, prob_part_up, prob_part_down] = s_features, s_outputs
        loss_ce = self.criterion_ce(prob, targets) + self.criterion_ce(prob_part_up, targets) + self.criterion_ce(prob_part_down, targets)
        
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(x, x, targets) + self.criterion_triple(part_up, part_up, targets) + self.criterion_triple(part_down, part_down, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(x, targets) + self.criterion_triple(part_up, targets) + self.criterion_triple(part_down, targets)
        prec, = accuracy(prob.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec


#Pretrainer with both fake and real images + DIM
class PreTrainerMwSynImg(object):

    def __init__(self, model, num_classes, margin=0.0, ce_epsilon=0.1, ratio=[1,1], disnet=None, lam=0.01, **kwargs):
        super(PreTrainerMwSynImg, self).__init__()
        self.model = model
        self.disnet = disnet
        self.islim  = not self.disnet is None
        self.ratio = ratio[0] // ratio[1] + 1 if ratio[0] // ratio[1] > 0 else 1
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes, epsilon = ce_epsilon).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        self.ALoss = nn.MSELoss().cuda()
        self.lamda = lam

    def train(self, epoch, real_loader_source,fake_loader_source,  optimizer, 
                train_iters=200, print_freq=1, logger=None, forDisNet=None, **kwargs):
        self.model.train()
        train_loader_target, optimizerD = None, None
        if self.islim:
            self.disnet.train()
            train_loader_target, optimizerD = forDisNet

        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        losses_limD = AverageMeter()
        losses_limM = AverageMeter()


        flag = True
        for i in range(train_iters):
            if flag and (i+1) % self.ratio == 0:
                source_inputs = fake_loader_source.next()
                i -= 1
                flag=False
            else:
                flag = True
                source_inputs = real_loader_source.next()

            s_inputs, targets, _ = self._parse_data(source_inputs)
            s_features, s_cls_out = self.model(s_inputs)
            loss_ce, loss_tr, prec1 = self._forward_loss(s_features, s_cls_out, targets)
            loss = loss_ce + loss_tr #could improve it?
            
            if self.islim:
                t_inputs, _, _, _, _ = train_loader_target.next()
                t_features, _ = self.model(t_inputs.cuda())   

            #train model with netD
            loss_lim = 0
            if epoch >= 1 and self.islim:
                D_duke = self.disnet(t_features)
                D_market = self.disnet(s_features)
                loss_lim = (self.ALoss(D_duke, torch.ones_like(D_duke)/2.) + self.ALoss(D_market, torch.ones_like(D_market))/2.)
                loss += loss_lim * self.lamda

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # train discriminator
            d_loss=0
            if self.islim:
                D_duke = self.disnet(t_features.detach())
                D_market = self.disnet(s_features.detach())
                d_loss = self.ALoss(D_market, torch.ones_like(D_market)) + self.ALoss(D_duke, torch.zeros_like(D_duke))
                optimizerD.zero_grad()
                d_loss.backward()
                optimizerD.step()

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            losses_limM.update(loss_lim.item() if loss_lim > 0 else 0)
            losses_limD.update(d_loss.item() if d_loss > 0 else 0)

            

            if ((i + 1) % print_freq == 0):
                logger.traininglog(epoch=epoch, i=i+1, iters=train_iters,
                    avgloss=loss, 
                    loss_ce=f"{losses_ce.val:.3f}({losses_ce.avg:.3f})",
                    loss_tr=f"{losses_tr.val:.3f}({losses_tr.avg:.3f})",
                    loss_G=f"{losses_limM.val:.3f}({losses_limM.avg:.3f})",
                    loss_D=f"{losses_limD.val:.3f}({losses_limD.avg:.3f})")


    def _parse_data(self, inputs):
        imgs, _, pids, _, isreal= inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets, isreal

    def _forward_loss(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec



#Finetune Trainer
class FTTrainer(object):
    def __init__(self, model, model_ema=None, num_cluster_list=None,  ce_epsilon=0.1, uetal=0.4, cent_uncertainty=None, alphas=None, num_parts=2,
                    logger=None, **kwargs) :
        super(FTTrainer, self).__init__()
        self.model = model
        self.model_ema = model_ema
        self.num_parts = num_parts
        self.num_clusters = num_cluster_list
        self.cent_uncertainty = cent_uncertainty
        self.alpha_score = alphas

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster_list, epsilon = ce_epsilon).cuda()
        self.criterion_stri = SoftTripletLoss(margin=0).cuda() 
        self.criterion_dauet = UET(alpha=uetal).cuda()

        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()
        self.criterion_ce_soft = KLDivLoss().cuda()

        self.logger = logger

    def train(self, epoch, data_loader_target, optimizer, ema_weights=(0.5, 0.8),
            print_freq=1, train_iters=200,  **kwargs):
        self.model.train()
        if not self.model_ema is None: self.model_ema.train()

        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        losses_ce_ema = AverageMeter()
        losses_tr_ema = AverageMeter()
        losses_total = AverageMeter()


        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            inputs, targets, gcentroidW, alphas  = self._parse_data(target_inputs)

            [x, part], [prob, _] = self.model(inputs[0].cuda())
            [x_ema, part_ema], [prob_ema, _] = self.model_ema(inputs[1].cuda())
            prob = prob[:,:self.num_clusters]
            prob_ema = prob_ema[:,:self.num_clusters]
            
        #####################################################
        #                           LOSSES                  #

        #ID Loss global 
            gloss_ce    = self.criterion_dauet(logits=prob, targets=targets, aff_score=gcentroidW, alphas=alphas)

        #Tri loss
            gloss_tri  = self.criterion_stri(x, x, targets) 
            ploss_tri  = 0 # we may try using triplet loss for subbranches oneday in future :') 
            # for i in range(self.num_parts):
            #         ploss_tri += self.criterion_stri(part[:,:, i], part[:, :, i], targets[1][i]) * ptw[i]
               
        #ema loss:  
            if self.model_ema is None:
                loss_ce      = gloss_ce
                loss_tri     = gloss_tri + ploss_tri
                loss_ce_soft = 0
                loss_tri_soft = 0
                loss =  loss_ce  + loss_tri                    
            else:
                loss_ce      = gloss_ce
                loss_tri     = gloss_tri + ploss_tri
                loss_ce_soft = self.criterion_ce_soft(prob, prob_ema) 
                loss_tri_soft = self.criterion_tri_soft(x, x_ema, targets)
                a, b = ema_weights
                loss =  (1-a) * loss_ce  + (1-b) * loss_tri + a * loss_ce_soft + b * loss_tri_soft 

                   
            #optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not self.model_ema is None: self._update_ema_variables(self.model, self.model_ema, 0.99, epoch*len(data_loader_target)+i)

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tri.item())
            losses_ce_ema.update(0 if loss_ce_soft == 0 else loss_ce_soft.item())
            losses_tr_ema.update(0 if loss_tri_soft == 0 else loss_tri_soft.item())
            losses_total.update(loss.item())
            
            if (i + 1) % print_freq == 0:
                self.logger.traininglog(epoch=epoch+1, i=i+1, iters=train_iters,
                    Lce=f"{losses_ce.avg:.2f}",
                    Ltr=f"{losses_tr.avg:.2f}",
                    Lce_ema=f"{losses_ce_ema.avg:.2f}",
                    Ltr_ema=f"{losses_tr_ema.avg:.2f}",
                    avgloss=losses_total.avg, 
                    )
    def _infer_ema(self, inputs):
        [x, part], [prob, _] = self.model(inputs[0].cuda())
        [x_ema, part_ema], [prob_ema,_] = self.model_ema(inputs[1].cuda())
        prob_ema = prob_ema[:,:self.num_clusters[0]]
        prob = prob[:,:self.num_clusters[0]]
        return [x, part, prob], [x_ema, part_ema, prob_ema]

    def _parse_data(self, inputs):
        imgs, _, gpids, _, (_, newidx) = inputs #(img_1, img_2), fname, pid, camid, (old_idx, new_idx)
        targets = gpids.cuda()
        aga = self.cent_uncertainty[newidx].cuda()
        alpha = self.alpha_score[newidx].cuda() if not self.alpha_score is None else None
        return imgs , targets, aga, alpha

    def _update_ema_variables(self, model, ema_model, alpha=0.99, global_step=0):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)