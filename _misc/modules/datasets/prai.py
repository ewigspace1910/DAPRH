# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
from glob import glob

from .data import BaseImageDataset


__all__ = ['PRAI',]



class PRAI(BaseImageDataset):
    dataset_dir = "PRAI-1581"
    dataset_name = 'prai'

    def __init__(self, root='datasets', verbose=False, for_merge=True, **kwargs):
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir, 'images')

        required_files = [self.train_path]
        #self.check_before_run(required_files)

        self._for_merge = self.process_merge(self.train_path)

        super().__init__()
        if verbose:
            print("=> PRAI loaded")
            self.print_dataset_statistics(self._for_merge)

    def process_merge(self, train_path):
        data = []
        img_paths = glob(os.path.join(train_path, "*.jpg"))
        for img_path in img_paths:
            split_path = img_path.split('/')
            img_info = split_path[-1].split('_')
            pid = self.dataset_name + "_" + img_info[0]
            camid = self.dataset_name + "_" + img_info[1]
            data.append((img_path, pid, camid))
        return data
