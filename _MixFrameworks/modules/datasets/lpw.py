# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
from glob import glob

from .data import BaseImageDataset

__all__ = ['LPW', ]


class LPW(BaseImageDataset):
    dataset_dir = "pep_256x128"
    dataset_name = "lpw"

    def __init__(self, root='datasets', verbose=False, for_merge=True, **kwargs):
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir)

        required_files = [self.train_path]
        #self.check_before_run(required_files)
        if for_merge: self._for_merge = self.process_merge(self.train_path)

        super().__init__(**kwargs)
        if verbose:
        
            print("=> LPW loaded")
            self.print_dataset_statistics(self._for_merge)

    def process_merge(self, train_path):
        data = []

        file_path_list = ['scen1', 'scen2', 'scen3']

        for scene in file_path_list:
            cam_list = os.listdir(os.path.join(train_path, scene))
            for cam in cam_list:
                camid = self.dataset_name + "_" + cam
                pid_list = os.listdir(os.path.join(train_path, scene, cam))
                for pid_dir in pid_list:
                    img_paths = glob(os.path.join(train_path, scene, cam, pid_dir, "*.jpg"))
                    for img_path in img_paths:
                        pid = self.dataset_name + "_" + scene + "-" + pid_dir
                        data.append((img_path, pid, camid))
        return data