from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
import urllib
import zipfile

from .data import BaseImageDataset

from .dukemtmc import DukeMTMC
from .market1501 import Market1501
from .msmt17 import MSMT17
from .prai import PRAI
from .lpw import LPW
from .noisyshoppingmall import NSMall

class MergedData(BaseImageDataset):
    """
    Merged Dataset: custom person ReID dataset
    """
    dataset_dir = './datasets'
    factory_ = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMC,
    'msmt17': MSMT17,
    'prai': PRAI,
    'lpw' : LPW,
    'noisy_shoppingmall' : NSMall
    }

    def __init__(self, list=['market1501', 'msmt17', 'lpw','prai'], verbose=True, **kwargs):
        super(MergedData, self).__init__()
        #assert 'dukemtmc' in list, "must using duke"
    
        self.train_dir = []
        for name in list : self.train_dir += self.factory_[name](verbose=True, for_merge=True)._for_merge
        train = self._process_train(self.train_dir, relabel=True)
        
        print("Warning! \n\t-We use duke-MTMC for validation\n\t- If U dont have duke dataset: either download it or change code in modules/datsets/merge.py")
        duke = self.factory_['dukemtmc'](verbose=False)
        query = duke.query
        gallery = duke.gallery

        if verbose:
            print("=> Custom dataset loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
        


    def _process_train(self, raw_dataset, relabel=True):

        pid_container = set()
        cid_container = set()
        for tup in raw_dataset:
            _, pid, cid = tup
            pid_container.add(pid)
            cid_container.add(cid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        cid2label = {cid: label for label, cid in enumerate(cid_container)}

        fin_dataset = []
        for path, pid, cid in raw_dataset:

            if relabel: 
                pid = pid2label[pid]
                cid = cid2label[cid]
                
            fin_dataset.append((path, pid, cid))

        return fin_dataset
    
    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset