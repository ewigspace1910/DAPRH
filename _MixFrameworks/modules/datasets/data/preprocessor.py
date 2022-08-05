from __future__ import absolute_import
import os.path as osp
from torch.utils.data import  Dataset
from PIL import Image

class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform1=None, transform2=None, mutual=False):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform1 = transform1
        self.transform2 = transform2
        self.mutual = mutual

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if self.mutual:
            return self._get_mutual_item(indices)
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform1 is not None:
            img = self.transform1(img)
        elif self.transform2 is not None:
            img = self.transform2(img)
        else:
            assert False, "tranform1 and tranform2 are None!!!"

        return img, fname, pid, camid

    def _get_mutual_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img_1 = Image.open(fpath).convert('RGB')
        img_2 = img_1.copy()

        if self.transform1 is not None and self.transform1 is not None:
            img_1 = self.transform1(img_1)
            img_2 = self.transform2(img_2)
        else: assert False, "tranform1 and tranform2 are None!!!"

        return img_1, img_2, fname, pid, camid, index #need return img1, img2, fname, pid, camid, index2
