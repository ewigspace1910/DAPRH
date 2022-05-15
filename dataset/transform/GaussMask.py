import numpy as np
from PIL import Image
import math
import random

class GaussianMask(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img
        width = img.size[0]
        height = img.size[1]
        mask = np.zeros((height, width))
        mask_h = np.zeros((height, width))
        mask_h += np.arange(0, width) - width / 2
        mask_v = np.zeros((width, height))
        mask_v += np.arange(0, height) - height / 2
        mask_v = mask_v.T

        numerator = np.power(mask_h, 2) + np.power(mask_v, 2)
        denominator = 2 * (height * height + width * width)
        mask = np.exp(-(numerator / denominator))

        img = np.asarray(img)
        new_img = np.zeros_like(img)
        new_img[:, :, 0] = np.multiply(mask, img[:, :, 0])
        new_img[:, :, 1] = np.multiply(mask, img[:, :, 1])
        new_img[:, :, 2] = np.multiply(mask, img[:, :, 2])

        return Image.fromarray(new_img)