from __future__ import print_function, absolute_import
import os
import os.path as osp
import glob
import re
import urllib
import zipfile

from .data import BaseImageDataset
import json

class NSMall(BaseImageDataset):
    'just for training'
    dataset_dir = 'NoisyShoppingMall'

    def __init__(self, root="./datasets", verbose=True, for_merge=True, **kwargs):
        super(NSMall, self).__init__()
        self.dataset_name = 'Noisy_ShoppingMall'
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir)

    
        if for_merge:  
            train = self._for_merge = self._process_merge(self.train_dir, relabel=True)

        else:  
            train = self.train = self._process_dir(self.train_dir, relabel=True)

        if verbose:
            print("=> Noisy shopping mall dataset loaded")
            self.print_dataset_statistics(train, [], [])

        self.train = train    
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        
    def _process_dir(self, dir_path, relabel=False):
        folder_paths = [p.path for p in os.scandir(dir_path)]
        img_paths = []
        for x in folder_paths:
            img_paths += glob.glob(osp.join(x, '*.jpg'))
        #pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            #pid, _ = map(int, pattern.search(img_path).groups())
            #if pid == -1: continue  # junk images are just ignored
            pid = self.dataset_name + "_" + img_path.split("/")[-2]
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            camid = str(1)
            pid = img_path.split("/")[-2]
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

    def _process_merge(self, dir_path, relabel=False):
        folder_paths = [p.path for p in os.scandir(dir_path)]
        img_paths = []
        for x in folder_paths:
            img_paths += glob.glob(osp.join(x, '*.jpg'))
        #pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            #pid, _ = map(int, pattern.search(img_path).groups())
            #if pid == -1: continue  # junk images are just ignored
            pid = self.dataset_name + "_" + img_path.split("/")[-2]
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            #pid, camid = map(int, pattern.search(img_path).groups())
            #if pid == -1: continue  # junk images are just ignored
            #assert 0 <= pid <= 1501  # pid == 0 means background
            #assert 1 <= camid <= 6
            #camid -= 1  # index starts from 0
            camid = self.dataset_name + "_" + str(1)
            pid = self.dataset_name + "_" + img_path.split("/")[-2]
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
