import os
import cv2
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
import math

import random
import numpy as np

from torchvision import transforms
import dataset.custom_transforms as tr


class HRSODDataset(Dataset):
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
        self.data_path = os.path.join(root, 'HRSOD')
        self.gt_thin_path = os.path.join(root, "thin_regions", 'hrsod')
        self.adp = adp
        self.adp = len(adp_ratio) > 0
        self.adp_ratio = adp_ratio
        self.C = 5000 / 15
        self.res_blur = res_blur

        self.kernel_list = []
        for r in range(100):
            side = 1 + 2 * r
            self.kernel_list.append(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(side, side)))

        with open(os.path.join(self.data_path, 'list', 'instance.txt')) as f:
            content = f.readlines()
        content = [line.strip() for line in content]
        pairs = [line.split() for line in content]

        self.filename_list = []
        self.index_list = []
        for filename, instance_num in pairs:
            for ind in range(1, int(instance_num)): 
                self.filename_list.append(filename)
                self.index_list.append(ind)

        print("use ori hrsod")

    
    def __len__(self):
        return len(self.filename_list)
    

    def adptive_size(self, size):
        ratio = random.choice(self.adp_ratio)
        ps = int(3 + ratio * size / self.C)
        ps = max(ps, 2)

        return ps
    
    
    def __getitem__(self, idx):
        ps = random.randint(2, 6) if self.pad_size is None else self.pad_size
        
        filename = self.filename_list[idx]
        index = self.index_list[idx]
        img_path = os.path.join(self.data_path, 'images', filename + ".jpg")
        mask_path = os.path.join(self.data_path, 'masks', filename + ".png")
        gt_thin_path = os.path.join(self.gt_thin_path, 'gt_thin', "{}-{}.png".format(filename, index))

        img = cv2.imread(img_path)
        gt_thin = cv2.imread(gt_thin_path, 0)

        mask = cv2.imread(mask_path, 0)
        mask[mask == index] = 255
        mask[mask < 126] = 0
        
        if self.adp:
            size = (mask > 128).sum()
            size = math.sqrt(size)
            ps = self.adptive_size(size)

        assert gt_thin.shape == img.shape[:2], "name {}, index {}, img shape {}, thin shape {}".format(filename, index, img.shape[:2], gt_thin.shape)

        try:
            kernel = self.kernel_list[ps]
        except:
            side = 1 + ps * 2
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(side, side))

        bgr_path = os.path.join(self.data_path, 'bgr_{}'.format(ps))
        pad_path = os.path.join(self.data_path, 'pad_{}'.format(ps))
        if not os.path.isdir(pad_path):
            os.mkdir(pad_path)
        if not os.path.isdir(bgr_path):
            os.mkdir(bgr_path)

        bgr_path = os.path.join(bgr_path, "{}-{}.jpg".format(filename, index))
        pad_path = os.path.join(pad_path, "{}-{}.png".format(filename, index))

        # if ps > 6:
        pad_broken, bgr_broken = False, False
        pad, bgr = None, None
        while pad is None:
            if pad_broken:
                os.remove(pad_path)
            if os.path.isfile(pad_path):
                pad = cv2.imread(pad_path, 0)
            else:
                try:
                    kernel = self.kernel_list[ps]
                except:
                    side = 1 + ps * 2
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(side, side))
                pad = cv2.dilate(gt_thin, kernel, iterations=1)
                cv2.imwrite(pad_path, pad)
            pad_broken = True
        if len(pad.shape) == 2:
            pad = np.expand_dims(cv2.imread(pad_path, 0), -1)

        while bgr is None:
            if bgr_broken:
                os.remove(bgr_path)
            if os.path.isfile(bgr_path):
                bgr = cv2.imread(bgr_path)
            else:
                print(f"{filename} background online")
                bgr = cv2.inpaint(img, pad, 3, cv2.INPAINT_TELEA)
                cv2.imwrite(bgr_path, bgr)
            bgr_broken = True

        syth = None
        pad[pad >= 50] = 255
        pad[pad < 50] = 0
        gt_thin[gt_thin >= 50] = 255
        gt_thin[gt_thin < 50] = 0

        sample = {"img" : img.astype(np.float32), "mask" : mask, 
            "bgr" : bgr, 'pad' : pad, 'gt' : gt_thin, "syth" : syth}
        sample['meta'] = {"filename" : filename, 'index' : index, 'kernel_r': ps}

        if self.transfroms is not None:
            sample = self.transfroms(sample)

        return sample


    def pad_thin(self, img):  
        kernel = np.ones((3, 3), np.uint8)  

        erosion = cv2.erode(img, kernel, iterations=1)
        dilation = cv2.dilate(erosion, kernel, iterations=random.randint(2, 6))

        return dilation


if __name__ == "__main__":
    for ar in [8]:
        test = HRSODDataset("../data", transforms=None,
            adp_ratio=[ar])
        for i in range(15):
            data = test[i]
            filename = data['meta']['filename']
            cv2.imwrite("./scribble_visual/{}_image.jpg".format(filename), data['img'])
            # cv2.imwrite("./tmp/hrsod_visual/{}_{}_syth.jpg".format(filename, ar), data['syth'])
            # cv2.imwrite("./tmp/hrsod_visual/{}_{}_bgr_cv.jpg".format(filename, ar), data['bgr'])
            # cv2.imwrite("./scribble_visual/{}_{}_scribble.png".format(filename, ar), data['pad'])
