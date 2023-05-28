from __future__ import absolute_import
from .serialization import save_checkpoint
import os.path as osp
import torch



def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def save_model(model_ema, is_best, best_mAP, mid, epoch, logdir, remain=False):
    name = 'model'+str(mid)+'_checkpoint.pth.tar' if mid > 0 else 'checkpoint.pth.tar' 
    save_checkpoint({
        'state_dict': model_ema.state_dict(),
        'epoch': epoch + 1,
        'best_mAP': best_mAP,
    }, is_best, fpath=osp.join(logdir, name), remain=remain)


def statistic(scores, name="haha"):
    silhouettes_g = scores
    logger.log(f"======{name} score=======")
    logger.log("\t--Mean: {}".format(silhouettes_g.mean()))
    logger.log("\t--Quantile=0.2    :{} ".format(np.quantile(silhouettes_g, 0.2)))
    logger.log("\t--Quantile=0.15   :{} ".format(np.quantile(silhouettes_g, 0.15)))
    logger.log("\t--Quantile=0.1    :{} ".format(np.quantile(silhouettes_g, 0.1)))
    logger.log("\t--Quantile=0.08   :{} ".format(np.quantile(silhouettes_g, 0.08)))
    logger.log("\t--Quantile=0.05   :{} ".format(np.quantile(silhouettes_g, 0.05)))
    logger.log("\t--Quantile=0.01   :{} ".format(np.quantile(silhouettes_g, 0.01)))