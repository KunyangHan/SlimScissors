import os
import cv2
import glob
import math
import random
import argparse
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import dataset.custom_transforms as tr
from dataset.helpers import get_bbox, crop2fullmask


class ThinObjectDataset(Dataset):
    def __init__(self, 
        root, 
        split='train', 
        transforms=None, 
        pad_size=None, 
        region_size=3, 
        adp_ratio=[], 
        dilate_upbound=6, 
        res_blur=False,
        adp=False
    ):
        self.split = split
        self.transfroms = transforms
        self.pad_size = pad_size
        self.region_size = region_size
        self.data_path = os.path.join(root, 'ThinObject5K')
        self.gt_thin_path = os.path.join(root, "thin_regions", 'thinobject5k_' + split)
        self.dilate_upbound = dilate_upbound
        self.adp = len(adp_ratio) > 0
        self.adp_ratio = adp_ratio
        self.C = 5000 / 15
        self.res_blur = res_blur

        self.kernel_list = []
        for r in range(100):
            side = 1 + 2 * r
            self.kernel_list.append(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(side, side)))

        with open(os.path.join(self.data_path, 'list', split + '.txt')) as f:
            content = f.readlines()
        fullname_list = [x.strip() for x in content]
        self.filename_list = [x[:x.rfind('.')] for x in fullname_list]


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
        ps = random.randint(2, self.dilate_upbound) if self.pad_size is None else self.pad_size
        rs = self.region_size

        filename = self.filename_list[idx]
        img_path = os.path.join(self.data_path, 'images', filename + ".jpg")
        region_path = os.path.join(self.data_path, 'ce_region_{}'.format(rs), filename + ".png")
        mask_path = os.path.join(self.data_path, 'masks', filename + ".png")
        gt_thin_path = os.path.join(self.gt_thin_path, 'gt_thin', filename + ".png-0.png")

        img = cv2.imread(img_path)
        gt_thin = cv2.imread(gt_thin_path, 0)
        mask = cv2.imread(mask_path, 0)

        assert gt_thin is not None, "{} gt_thin is None".format(idx)

        if self.adp:
            size = (mask > 128).sum()
            size = math.sqrt(size)
            ps = self.adptive_size(size)
            ps = self.quantify(ps)

        bgr_path = os.path.join(self.data_path, 'bgr_{}'.format(ps))
        pad_path = os.path.join(self.data_path, 'pad_{}'.format(ps))
        if not os.path.isdir(pad_path):
            os.mkdir(pad_path)
        if not os.path.isdir(bgr_path):
            os.mkdir(bgr_path)

        bgr_path = os.path.join(self.data_path, 'bgr_{}'.format(ps), filename + ".jpg")
        pad_path = os.path.join(self.data_path, 'pad_{}'.format(ps), filename + ".png")

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
                bgr = cv2.inpaint(img, pad, 3, cv2.INPAINT_TELEA)
                cv2.imwrite(bgr_path, bgr)
            bgr_broken = True

        syth = None
        res = None

        if self.split == 'train':
            region = np.expand_dims(cv2.imread(region_path, 0), -1)
        else:
            region = np.zeros(pad.shape)

        assert img is not None, "img at {} is None".format(img_path)
        assert bgr is not None, "bgr at {} is None".format(bgr_path)
        assert pad is not None, "pad at {} is None".format(pad_path)
        assert mask is not None, "mask at {} is None".format(mask_path)
        assert gt_thin is not None, "gt_thin at {} is None".format(gt_thin_path)

        pad[pad >= 50] = 255
        pad[pad < 50] = 0
        gt_thin[gt_thin >= 50] = 255
        gt_thin[gt_thin < 50] = 0

        sample = {"img" : img.astype(np.float32), "mask" : mask, "bgr" : bgr, 
            'pad' : pad, 'gt' : gt_thin, 'region': region}
        sample['meta'] = {"filename" : filename, 'kernel_r': ps}

        if self.transfroms is not None:
            sample = self.transfroms(sample)

        return sample


    def pad_thin(self, img):  
        kernel = np.ones((3, 3), np.uint8)  

        erosion = cv2.erode(img, kernel, iterations=1)
        dilation = cv2.dilate(erosion, kernel, iterations=random.randint(2, 6))

        return dilation


def save_mask(gt, bbox, mask, th, name, data_root):
    mask_th = crop2fullmask(gt, bbox, mask, zero_pad=True, relax=30, mask_relax=True)
    cv2.imwrite(os.path.join(data_root, "ThinObject5K/ce_region_{}/{}.png".format(th, name)), mask_th)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train refine')
    parser.add_argument('--data_root', type=str, default='../data')

    args = parser.parse_args()

    gen_ce_regeion(args.data_root)


def gen_ce_regeion(data_root):
    composed_transforms_ts = transforms.Compose([
        tr.CropFromMask(crop_elems=('gt', 'mask'), mask_elem='mask', relax=30, zero_pad=True),
        tr.FixedResize(
            resolutions={'crop_gt': (512, 512), 'crop_mask': (512, 512), 'mask': None}, 
            flagvals={'crop_gt': cv2.INTER_NEAREST, 'crop_mask': cv2.INTER_NEAREST, 'mask': cv2.INTER_NEAREST})])

    train = ThinObjectDataset(data_root, "train", transforms=composed_transforms_ts)
    length = len(train)

    kernel = np.ones((7, 7), dtype=np.uint8)
    for i, sample in enumerate(train):
        gt = sample["crop_gt"]
        mask = np.squeeze(sample["mask"])
        filename = sample['meta']['filename']

        mask = mask.astype(np.float32) / 255
        mask = np.float32(mask > 0.3)
        bbox = get_bbox(mask, pad=30, zero_pad=True)

        gt_thin = np.squeeze(gt)
        thin_num = (gt_thin > 128).sum()
        total_num = thin_num

        c , flag_2 , flag_3 = 0, True, True
        while True:
            gt_thin = cv2.dilate(gt_thin, kernel, iterations=1)
            total_num = (gt_thin > 128).sum()

            if flag_2 and thin_num * (2 + 1) <= total_num:
                save_mask(gt_thin, bbox, mask, 2, filename, data_root)
                flag_2 = False

            if flag_3 and thin_num * (3 + 1) <= total_num:
                save_mask(gt_thin, bbox, mask, 3, filename, data_root)
                flag_3 = False

                break
            
            c += 1
            if c > 20 or total_num >= 260_000:
                if flag_2:
                    save_mask(gt_thin, bbox, mask, 2, filename, data_root)
                if flag_3:
                    save_mask(gt_thin, bbox, mask, 3, filename, data_root)
                break

        print("[{}]/[{}]".format(i, length), end='\r')
