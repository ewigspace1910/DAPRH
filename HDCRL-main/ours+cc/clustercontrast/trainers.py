from __future__ import print_function, absolute_import
import time
import torch
from .utils.meters import AverageMeter


class ClusterContrastTrainer(object):
    def __init__(self, encoder, encoder_ema, memory=None, alpha=0.999):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.encoder_ema = encoder_ema
        self.memory = memory
        self.alpha = alpha

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
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
            inputs_1, inputs_2, labels, indexes = self._parse_data(inputs)
            
            # forward
            f_out_1 = self.encoder(inputs_1)
            with torch.no_grad():
                f_out_2 = self.encoder_ema(inputs_2)
            
            loss_h, loss_m = self.memory(f_out_1, f_out_2, labels)
            loss = loss_h + loss_m

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
                      'Loss_m {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              losses_h.val, losses_h.avg,
                              losses_m.val, losses_m.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs[0].cuda(), imgs[1].cuda(), pids.cuda(), indexes.cuda()

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        # alpha = min(1 - 1 / (global_step + 1), alpha)
        if global_step >= 99:
            alpha = min(1 - 1 / (global_step - 98), alpha)
        # else:
        #     alpha = 0

        for (ema_name, ema_param), (model_name, param) in zip(ema_model.named_parameters(), model.named_parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

