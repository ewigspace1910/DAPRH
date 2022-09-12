# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import sys
import os
import os.path as osp
import glob 

from .data import BaseImageDataset
##### Log #####
# 22.01.2019
# - add v2
# - v1 and v2 differ in dir names
# - note that faces in v2 are blurred
TRAIN_DIR_KEY = 'train_dir'
TEST_DIR_KEY = 'test_dir'
VERSION_DICT = {
    'MSMT17_V1': {
        TRAIN_DIR_KEY: 'train',
        TEST_DIR_KEY: 'test',
    },
    'MSMT17_V2': {
        TRAIN_DIR_KEY: 'mask_train_v2',
        TEST_DIR_KEY: 'mask_test_v2',
    }
}


class MSMT17(BaseImageDataset):
    """MSMT17.
    Reference:
        Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.
    URL: `<http://www.pkuvmc.com/publications/msmt17.html>`_

    Dataset statistics:
        - identities: 4101.
        - images: 32621 (train) + 11659 (query) + 82161 (gallery).
        - cameras: 15.
    """
    # dataset_dir = 'MSMT17_V2'
    dataset_url = None
    dataset_name = 'msmt17'

    def __init__(self, root='datasets', verbose=False, for_merge=False, **kwargs):
        self.dataset_dir = root

        has_main_dir = False
        for main_dir in VERSION_DICT:
            if osp.exists(osp.join(self.dataset_dir, main_dir)):
                train_dir = VERSION_DICT[main_dir][TRAIN_DIR_KEY]
                test_dir = VERSION_DICT[main_dir][TEST_DIR_KEY]
                has_main_dir = True
                break
        assert has_main_dir, 'Dataset folder not found'

        self.train_dir = osp.join(self.dataset_dir, main_dir, train_dir)
        self.test_dir = osp.join(self.dataset_dir, main_dir, test_dir)
        self.list_train_path = osp.join(self.dataset_dir, main_dir, 'list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, main_dir, 'list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, main_dir, 'list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, main_dir, 'list_gallery.txt')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.test_dir
        ]
        

        train = self.process_dir(self.train_dir, self.list_train_path)
        val = self.process_dir(self.train_dir, self.list_val_path)
        query = self.process_dir(self.test_dir, self.list_query_path, is_train=False)
        gallery = self.process_dir(self.test_dir, self.list_gallery_path, is_train=False)

        num_train_pids = self.parse_data(train)[0]
        query_tmp = []
        for img_path, pid, camid in query:
            query_tmp.append((img_path, pid+num_train_pids, camid))
        del query
        query = query_tmp
        
        gallery_temp = []
        for img_path, pid, camid in gallery:
            gallery_temp.append((img_path, pid+num_train_pids, camid))
        del gallery
        gallery = gallery_temp

        # Note: to fairly compare with published methods on the conventional ReID setting,
        #       do not add val images to the training set.
        #if 'combineall' in kwargs and kwargs['combineall']:
        #    train += val
        train += val
        if for_merge:
            self._for_merge = self.process_merge(self.train_dir, self.list_train_path)
        super(MSMT17, self).__init__(**kwargs)
        
        if verbose:
            print("=> MSMT loaded")
            self.print_dataset_statistics(train, query, gallery)

    def process_dir(self, dir_path, list_path, is_train=True):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        data = []

        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2]) - 1  # index starts from 0
            img_path = osp.join(dir_path, img_path)
            if is_train:
                pid = str(pid)
                camid = str(camid)
            data.append((img_path, pid, camid))

        return data

    def process_merge(self, dir_path, list_path, is_train=True):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        data = []

        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2]) - 1  # index starts from 0
            img_path = osp.join(dir_path, img_path)
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data
        
    def parse_data(self, data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.
        Args:
            data (list): contains tuples of (img_path(s), pid, camid)
        """
        pids = set()
        cams = set()
        
        for info in data:
            pids.add(info[1])
            cams.add(info[2])
        return len(pids), len(cams)