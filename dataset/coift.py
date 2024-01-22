import os
import cv2
import glob
from torch.utils.data import Dataset
from PIL import Image
import math

import random
import numpy as np

from torchvision import transforms
import dataset.custom_transforms as tr


class COIFTDataset(Dataset):
    def __init__(self, 
        root, 
        transforms=None, 
        pad_size=None, 
        adp=False, 
        adp_ratio=[], 
        res_blur=False
    ):
        self.transfroms = transforms
        self.pad_size = pad_size
        self.data_path = os.path.join(root, 'COIFT')
        self.gt_thin_path = os.path.join(root, "thin_regions", 'coift')
        self.adp = len(adp_ratio) > 0
        self.adp_ratio = adp_ratio
        self.C = 5000 / 15
        self.res_blur = res_blur

        self.kernel_list = []
        for r in range(50):
            side = 1 + 2 * r
            self.kernel_list.append(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(side, side)))

        with open(os.path.join(self.data_path, 'list', 'test.txt')) as f:
            content = f.readlines()

        self.filename_list = [x.strip() for x in content]

    
    def __len__(self):
        return len(self.filename_list)
    

    def adptive_size(self, size):
        ratio = random.choice(self.adp_ratio)
        ps = int(3 + ratio * size / self.C)
        ps = max(ps, 2)

        return ps


    def quantify(self, x):
        y = None
        if x < 60:
            y = x
        elif x < 100:
            y = 5 * round(x / 5)
        else:
            y = 10 * round(x / 10)

        return y


    def __getitem__(self, idx):
        ps = random.randint(2, 6) if self.pad_size is None else self.pad_size
        
        filename = self.filename_list[idx]
        img_path = os.path.join(self.data_path, 'images', filename + ".jpg")
        mask_path = os.path.join(self.data_path, 'masks', filename + ".png")
        gt_thin_path = os.path.join(self.gt_thin_path, 'gt_thin', filename + "-255.png")

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)
        gt_thin = cv2.imread(gt_thin_path, 0)
        
        if self.adp:
            size = (mask > 128).sum()
            size = math.sqrt(size)
            ps = self.adptive_size(size)
        try:
            kernel = self.kernel_list[ps]
        except:
            side = 1 + ps * 2
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(side, side))

        pad = cv2.dilate(gt_thin, kernel, iterations=1)
        bgr = cv2.inpaint(img, pad, 3, cv2.INPAINT_TELEA)

        syth = np.repeat(np.expand_dims(pad, -1), 3, axis=-1)
        syth = 0.5 * syth + 0.5 * img

        if self.res_blur:
            img_blur = cv2.medianBlur(img, 5)
        else:
            img_blur = img
        res = np.abs(img_blur - bgr)

        pad[pad >= 50] = 255
        pad[pad < 50] = 0
        gt_thin[gt_thin >= 50] = 255
        gt_thin[gt_thin < 50] = 0

        sample = {"img" : img.astype(np.float32), "mask" : mask.astype(np.float32), 
            "bgr" : bgr, 'pad' : pad, 'gt' : gt_thin, 'res' : res, 'syth' : syth}
        sample['meta'] = {"filename" : filename, 'kernel_r': ps}

        if self.transfroms is not None:
            sample = self.transfroms(sample)

        return sample


if __name__ == "__main__":
    import imageio

    for ar in [8, 16, 24]:
        test = COIFTDataset("../data", transforms=None, adp_ratio=[ar])
        for i in [6]:
            data = test[i]
            filename = data['meta']['filename']
            cv2.imwrite("./scribble_visual/{}_image.jpg".format(filename), data['img'])
            cv2.imwrite("./scribble_visual/{}_{}_scribble.png".format(filename, ar), data['pad'])

