import os
import torch

from torch.backends import cudnn

from config import Configuration
from dataset import make_dataloader
from model import make_model
from loss import make_loss
from processor import do_train
from solver import make_optimizer, WarmupMultiStepLR
from utils.logger import setup_logger

if __name__ == '__main__':
    Cfg = Configuration()
    log_dir = Cfg.DATALOADER.LOG_DIR
    logger = setup_logger('{}'.format(Cfg.PROJECT_NAME), log_dir)
    logger.info("Running with config:\n{}".format(Cfg.PROJECT_NAME))

    os.environ['CUDA_VISIBLE_DEVICES'] = Cfg.DEVICE_ID
    cudnn.benchmark = True
    # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.

    
    train_loader, val_loader = make_dataloader(Cfg) #data
    model = make_model(Cfg) #model
    optimizer = make_optimizer(Cfg, model) #opt
    scheduler = WarmupMultiStepLR(Cfg, optimizer) #lr scheduler
    loss_func = make_loss(Cfg) #loss function
    #training loop
    do_train(
        Cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,  
        loss_func,
    )
