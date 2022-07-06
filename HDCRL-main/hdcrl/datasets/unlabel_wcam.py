from __future__ import print_function, absolute_import
import os
import os.path as osp
import glob
import re
import urllib
import zipfile


from ..utils.data import BaseImageDataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json
import json

class UnlabelwCamDs(BaseImageDataset):
    dataset_dir = 'UnlabelDS_wCamid'

    def __init__(self, root="./datasets", verbose=True, **kwargs):
        super(UnlabelwCamDs, self).__init__()
        self.dataset_name = 'unlabeled_wcam_dataset'
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Unlabel with CamID DataSet loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

        

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

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
            pid = 1
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            camid = int(img_path.split("/")[-2].split("_")[-1])  #/unlabeled_wcam_dataset/bounding_box_test/cam_1/00001.jpg
            pid = int(1)
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
