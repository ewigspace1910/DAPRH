from __future__ import print_function, absolute_import
import time
import numpy as np
import collections

import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils.meters import AverageMeter


class Trainer_UDA(object):
    def __init__(self, encoder, encoder_ema, memory, source_classes, alpha=0.999):
        super(Trainer_UDA, self).__init__()
        self.encoder = encoder
        self.encoder_ema = encoder_ema
        self.memory = memory
        self.source_classes = source_classes
        self.alpha = alpha

    def train(self, epoch, data_loader_source, data_loader_target,
                    optimizer, print_freq=10, train_iters=100):
        self.encoder.train()
        self.encoder_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_s = AverageMeter()
        losses_t = AverageMeter()
        losses_ms = AverageMeter()
        losses_mt = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            s1_in, s2_in, s_targets, _ = self._parse_data(source_inputs)
            t1_in, t2_in, _, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            device_num = torch.cuda.device_count()
            B, C, H, W = s1_in.size()
            def reshape(inputs):
                return inputs.view(device_num, -1, C, H, W)
            s1_in, s2_in, t1_in, t2_in = reshape(s1_in), reshape(s2_in), reshape(t1_in), reshape(t2_in)

            # Student Network forward
            inputs = torch.cat((s1_in, t1_in), 1).view(-1, C, H, W)
            f_out = self.encoder(inputs)
            f_out = f_out.view(device_num, -1, f_out.size(-1))
            s1_out, t1_out = f_out.split(f_out.size(1)//2, dim=1)
            s1_out, t1_out = s1_out.contiguous().view(-1, f_out.size(-1)), t1_out.contiguous().view(-1, f_out.size(-1))

            # Teacher Network forward
            inputs_ema = torch.cat((s2_in, t2_in), 1).view(-1, C, H, W)
            with torch.no_grad():
                f_out_ema = self.encoder_ema(inputs_ema)
            f_out_ema = f_out_ema.view(device_num, -1, f_out_ema.size(-1))
            s2_out, t2_out = f_out_ema.split(f_out_ema.size(1) // 2, dim=1)
            s2_out, t2_out = s2_out.contiguous().view(-1, f_out_ema.size(-1)), t2_out.contiguous().view(-1, f_out_ema.size(-1))

            # compute loss with the hybrid memory
            loss_s, loss_ms = self.memory(s1_out, s2_out, s_targets, self.source_classes)
            loss_t, loss_mt = self.memory(t1_out, t2_out, t_indexes+self.source_classes, 0)
            loss_mse = loss_ms + loss_mt

            loss = loss_s + loss_t + loss_mse
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.encoder, self.encoder_ema, self.alpha, epoch * len(data_loader_target) + i)

            losses_s.update(loss_s.item())
            losses_t.update(loss_t.item())
            losses_ms.update(loss_ms.item())
            losses_mt.update(loss_mt.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.2f} ({:.2f})\t'
                      'Loss_s {:.2f} ({:.2f})\t'
                      'Loss_t {:.2f} ({:.2f})\t'
                      'Loss_ms {:.2f} ({:.2f})\t'
                      'Loss_mt {:.2f} ({:.2f})'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              losses_s.val, losses_s.avg,
                              losses_t.val, losses_t.avg,
                              losses_ms.val, losses_ms.avg,
                              losses_mt.val, losses_mt.avg,))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs[0].cuda(), imgs[1].cuda(), pids.cuda(), indexes.cuda()

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        # alpha = min(1 - 1 / (global_step + 1), alpha)
        if global_step >= 49:
            alpha = min(1 - 1 / (global_step - 48), alpha)
        # else:
        #     alpha = 0

        for (ema_name, ema_param), (model_name, param) in zip(ema_model.named_parameters(), model.named_parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class Trainer_USL(object):
    def __init__(self, encoder, encoder_ema, memory, alpha=0.999):
        super(Trainer_USL, self).__init__()
        self.encoder = encoder
        self.encoder_ema = encoder_ema
        self.memory = memory
        self.alpha = alpha

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=100):
        self.encoder.train()
        self.encoder_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_h = AverageMeter()
        losses_m = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_1, inputs_2, _, indexes = self._parse_data(inputs)

            # forward
            f_out_1 = self.encoder(inputs_1)
            with torch.no_grad():
                f_out_2 = self.encoder_ema(inputs_2)

            # compute loss with the hybrid memory
            loss_h, loss_m = self.memory(f_out_1, f_out_2, indexes, 0)
            loss = loss_h # + loss_m

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.encoder, self.encoder_ema, self.alpha, epoch * len(data_loader) + i)

            losses_h.update(loss_h.item())
            losses_m.update(loss_m.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss_h {:.3f} ({:.3f})\t'
                      'Loss_m {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              data_time.val, data_time.avg,
                              losses_h.val, losses_h.avg,
                              losses_m.val, losses_m.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs[0].cuda(), imgs[1].cuda(), pids.cuda(), indexes.cuda()

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        # alpha = min(1 - 1 / (global_step + 1), alpha)
        if global_step >= 49:
            alpha = min(1 - 1 / (global_step - 48), alpha)
        # else:
        #     alpha = 0

        for (ema_name, ema_param), (model_name, param) in zip(ema_model.named_parameters(), model.named_parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _forward(self, inputs):
        return self.encoder(inputs)
