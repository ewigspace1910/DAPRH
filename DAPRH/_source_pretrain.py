from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import os
sys.path.append(os.getcwd())

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from modules import datasets
from modules import models
from modules.models.gan import DisNet
from modules.trainers import PreTrainerMwSynImg
from modules.evaluators import Evaluator
from modules.datasets.data import IterLoader
from modules.datasets.data import transforms as T
from modules.datasets.data.sampler import RandomMultipleGallerySampler
from modules.datasets.data.preprocessor import SyntheticPreprocessor, Preprocessor
from modules.utils.logger import Logger
from modules.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from modules.utils.lr_scheduler import WarmupMultiStepLR
import time

# haha - 1 #it none sense ::)))

logger = Logger()
start_epoch = best_mAP = 0

def get_data(name, data_dir, height, width, batch_size, workers, num_instances, iters=200, issource=True, is_fake=False, **kwargs):
    root = osp.join(data_dir)#osp.join(data_dir, name)

    if is_fake:
        dataset = datasets.create('synimgs', root, typeds=name, only_fake=True)
    else:
        dataset = datasets.create(name, root, typeds=name)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = sorted(dataset.train)
    num_classes = dataset.num_train_pids

    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             #T.RandomErasing(),
             normalizer
         ])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    if is_fake:
        train_loader = IterLoader(
            DataLoader(SyntheticPreprocessor(train_set, root=dataset.images_dir,
                                        transform=train_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    else:
        train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                        transform=train_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, test_loader


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP

    cudnn.benchmark = True
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    _, _, train_loader_source_fake, _ = \
        get_data(args.dataset_source, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers, args.num_instances, iters, issource=True, is_fake=True)
    dataset_source, num_classes, train_loader_source_real, test_loader_source = \
        get_data(args.dataset_source, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers, args.num_instances, iters, issource=True, is_fake=False)

    dataset_target, _, train_loader_target, test_loader_target = \
        get_data(args.dataset_target, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers, 0, iters, issource=False, is_fake=False)

    # Create model
    model = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=num_classes)
    model.cuda()
    model = nn.DataParallel(model)
    disnet = nn.DataParallel(DisNet().cuda()) if args.dim else None

    # Load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['best_mAP']
        logger.log("=> Start epoch {}  best mAP {:.1%}"
              .format(start_epoch, best_mAP))
    elif args.pretrained != "":
        #load pretrain.
        checkpoint = load_checkpoint(args.pretrained)
        copy_state_dict(checkpoint['state_dict'], model)
        logger.log("=> Load pretrained from {}"
              .format(args.pretrained))
        

    # Evaluator
    evaluator = Evaluator(model, logger=logger)

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)
    
    optimizerD = torch.optim.Adam(disnet.parameters(), lr = args.lr, weight_decay = 2.5e-4) if args.dim else None
    # Trainer
    tmodel = "".join([x for x in args.arch if x.isalpha()])

    trainer = PreTrainerMwSynImg(model, num_classes, margin=args.margin, 
                        model_type=tmodel, lam=args.lamda, ratio=args.ratio, disnet=disnet)

    # Start training
    ii = 0
    train_loader_source_fake.new_epoch()
    for epoch in range(start_epoch, args.epochs):
        train_loader_source_real.new_epoch()
        train_loader_target.new_epoch()
        try: train_loader_source_fake.next()
        except: train_loader_source_fake.new_epoch()


        trainer.train(epoch, train_loader_source_real, train_loader_source_fake, optimizer, 
                    train_iters=len(train_loader_source_real), print_freq=args.print_freq, logger=logger,
                    forDisNet=(train_loader_target, optimizerD))
        lr_scheduler.step()

        time.sleep(2)
        with torch.no_grad():
            torch.cuda.empty_cache()

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):

            _, mAP = evaluator.evaluate(test_loader_source, dataset_source.query, dataset_source.gallery, cmc_flag=True)

            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            logger.log('\n * Finished epoch {:3d}  source mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))
            
                

    logger.log("Final test on target domain:")
    evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True, rerank=args.rerank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-training on the source domain")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='market1501|dukemtmc|msmt17')
    parser.add_argument('-dt', '--dataset-target', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
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
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70], help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--eval-step', type=int, default=40)
    parser.add_argument('--rerank', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--iters', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=25)
    parser.add_argument('--margin', type=float, default=0.0, help='margin for the triplet loss with batch hard')
    

    
    
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    print("working_dir: ", working_dir)
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'datasets'))
    parser.add_argument('--fake-data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'fake-datasets'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument("-pt", "--pretrained", type=str, default="", help="path to pretrained model file")

    #toggle
    parser.add_argument('--ratio', nargs='+', type=int, default=[3, 1], help='number real : number fake')
    parser.add_argument('--dim', action='store_true', help='turnon lim loss of not')
    parser.add_argument('--fonly', action='store_true', help='using only fake imgs for training')
    #control losss
    parser.add_argument('--lamda', type=float, default=1., help='lamda for loss of fake images')
    
    main()