import argparse
import cv2
import torch
import os
import shutil
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from threading import Thread
from tqdm import tqdm
from PIL import Image

from dataset import custom_transforms as tr
from dataset import ThinObjectDataset, COIFTDataset#, HRSODDataset
from dataset.hrsod_ori import HRSODDataset
from dataset.helpers import get_bbox, crop2fullmask
from BGM_IOG import Network

from eval_script import f_boundary, jaccard

import imageio

parser = argparse.ArgumentParser(description='Train refine')

parser.add_argument('--model-type', type=str, default='mattingrefine')
parser.add_argument('--model-backbone', type=str, choices=['resnet18', 'resnet101', 'resnet50', 'mobilenetv2'], default='resnet18')
parser.add_argument('--model-backbone-scale', type=float, default=0.25)
parser.add_argument('--model-refine-mode', type=str, default='full', choices=['full', 'sampling', 'thresholding'])
parser.add_argument('--model-refine-sample-pixels', type=int, default=80000)
parser.add_argument('--model-refine-threshold', type=float, default=0.1)
parser.add_argument('--model-refine-kernel-size', type=int, default=3)

parser.add_argument('--gpu', default='0', type=str, help='GPU ID')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--ckpt', help='ckpt folder', default='ckpt')

parser.add_argument('--pad_size', nargs='+', type=int)
parser.add_argument('--gca', type=int, default=1)
parser.add_argument('--data_root', help='root folder of dataset', default='../data')
parser.add_argument('--dataset', help='which dataset to test on', default='ThinObject')

parser.add_argument('--adaptive_relax', type=bool, default=False)

parser.add_argument('--ada_ratio', nargs='+', type=float, default=[8, 16, 24])
parser.add_argument('--thr', nargs='+', type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
CKPT_DIR = os.path.join(THIS_DIR, args.ckpt)
if not os.path.isdir(CKPT_DIR):
    os.makedirs(CKPT_DIR)

IOG_Input_Channel = 5
BGM_Input_Channel = 7
Refine_Input_Channel = 7


def main():
    net = Network(BGMInputChannels=BGM_Input_Channel, 
                    IOGInputChannels=IOG_Input_Channel, 
                    output_stride=16,
                    sync_bn=None,
                    freeze_bn=False,
                    refine_mode=args.model_refine_mode,
                    backbone=args.model_backbone,
                    iog_backbone=args.model_backbone,
                    refine_in_channels=Refine_Input_Channel)
    net.cuda()

    if args.resume:
        if os.path.isfile(args.resume): 
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'".format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    composed_transforms_ts = transforms.Compose([
        tr.CropFromMask_h(crop_elems=('img', 'gt','bgr', 'mask', 'pad'), 
            mask_elem='mask', relax=30, zero_pad=True, adaptive_relax=args.adaptive_relax),
        tr.FixedResize(
            resolutions={'crop_img': (512, 512), 'crop_gt': (512, 512), 'crop_bgr': (512, 512),
                'crop_mask': (512, 512), 'crop_pad': (512, 512)}, 
            flagvals={'crop_img': cv2.INTER_LINEAR,'crop_gt': cv2.INTER_NEAREST,'crop_bgr': cv2.INTER_LINEAR, 
                'crop_mask': cv2.INTER_NEAREST, 'crop_pad' : cv2.INTER_NEAREST}),
        tr.IOGPoints(sigma=10, elem='crop_mask', pad_pixel=10),
        tr.ToImage(norm_elem='IOG_points'),
        tr.ToTensor()])

    for ar in args.ada_ratio:
        test(net, composed_transforms_ts, -1, [ar])


def test(net, composed_transforms_ts, pad_size, ada_ratio):
    if args.dataset == 'ThinObject':
        test_dateset = ThinObjectDataset(args.data_root, "test", 
            transforms=composed_transforms_ts, 
            pad_size=pad_size, adp_ratio=ada_ratio)
    elif args.dataset == 'COIFT':
        test_dateset = COIFTDataset(args.data_root, 
            transforms=composed_transforms_ts, 
            pad_size=pad_size, adp_ratio=ada_ratio)
    elif args.dataset == 'HRSOD':
        test_dateset = HRSODDataset(args.data_root, 
            transforms=composed_transforms_ts, 
            pad_size=pad_size, adp_ratio=ada_ratio)

    dataloader = DataLoader(test_dateset, batch_size=1, shuffle=False, num_workers=32)
    net.eval()

    totol_len = len(dataloader)
    iou_list = {str(t) : np.zeros(totol_len) for t in args.thr}
    thin_iou_list = {str(t) : np.zeros(totol_len) for t in args.thr}
    f_boundary_list = {str(t) : np.zeros(totol_len) for t in args.thr}

    if args.dataset == 'ThinObject':
        data_path = os.path.join(args.data_root, "ThinObject5K/masks")
        gt_thin_path = os.path.join(args.data_root, "thin_regions/thinobject5k_test/eval_mask")
    elif args.dataset == 'COIFT':
        data_path = os.path.join(args.data_root, "COIFT/masks")
        gt_thin_path = os.path.join(args.data_root, "thin_regions/coift/eval_mask")
    elif args.dataset == 'HRSOD':
        data_path = os.path.join(args.data_root, "HRSOD/masks")
        gt_thin_path = os.path.join(args.data_root, "thin_regions/hrsod/eval_mask")

    length = len(dataloader)
    for i, sample_batched in enumerate(dataloader):
        src = sample_batched["crop_img"]
        bgr = sample_batched["crop_bgr"]
        gt = sample_batched["crop_gt"]
        pad = sample_batched["crop_pad"]
        mask = sample_batched["crop_mask"]
        point = sample_batched["IOG_points"]
        filename = sample_batched['meta']['filename'][0]
        if args.dataset == 'HRSOD':
            index = sample_batched['meta']['index'][0].item()
        if args.adaptive_relax:
            relax = int(sample_batched['meta']['relax'][0])
        else:
            relax = 30

        src, bgr, gt =  src.cuda(), bgr.cuda(), gt.cuda()
        pad, mask, point = pad.cuda(), mask.cuda(), point.cuda()

        IOG_input = torch.cat([src, point], dim=1)
        BGM_input = torch.cat([src, bgr, pad], dim=1)
        refine_concat = torch.cat([bgr, pad], dim=1)

        pha, pha_sm, err_sm, ref_sm, coarse_outs, fine_out, final_result = net(src, refine_concat, IOG_input, BGM_input)

        final_mask = torch.sigmoid(final_result)

        mask_path = os.path.join(data_path, filename + ".png")
        if args.dataset == 'ThinObject':
            eval_mask_path = os.path.join(gt_thin_path, filename + ".png-0.png")
        elif args.dataset == 'COIFT':
            eval_mask_path = os.path.join(gt_thin_path, filename + "-255.png")
        elif args.dataset == 'HRSOD':
            eval_mask_path = os.path.join(gt_thin_path, filename + "-{}.png".format(index))

        mask_ori = cv2.imread(mask_path, 0)
        if args.dataset == 'HRSOD':
            mask_ori[mask_ori == index] = 255
            mask_ori[mask_ori < 126] = 0
        mask_ori = mask_ori.astype(np.float32) / 255
        mask_ori = np.float32(mask_ori > 0.3)
        mask_eval = np.array(Image.open(eval_mask_path)).astype(np.float32)

        final_mask = final_mask.to(torch.device('cpu'))
        pred = np.transpose(final_mask.data.numpy()[0, :, :, :], (1, 2, 0))
        pred = np.squeeze(pred)
        bbox = get_bbox(mask_ori, pad=relax, zero_pad=True)
        result = crop2fullmask(pred, bbox, mask_ori, zero_pad=True, relax=relax, mask_relax=True)

        assert mask_ori.shape == result.shape, "result : {}, mask {}".format(result.shape, mask_ori.shape)
        assert mask_eval.shape == result.shape

        for threshold in args.thr:
            mask = np.float32(result > threshold)

            iou_list[str(threshold)][i] = jaccard.jaccard(mask_ori, mask)
            void_thin = np.float32(mask_eval == 255)
            thin_iou_list[str(threshold)][i] = jaccard.jaccard(mask_ori, mask, void_thin)
            f_boundary_list[str(threshold)][i] = f_boundary.db_eval_boundary(mask, mask_ori)

        print("[{}]/[{}]".format(i, length), end='\r')

    print("Ada ratio: {}".format(ada_ratio))
    for threshold in args.thr:
        info =  "Thr {}, IOU: {:.6f} ".format(threshold, iou_list[str(threshold)].mean()) + \
                "IOU thin: {:.6f} ".format(thin_iou_list[str(threshold)].mean()) + \
                "F boundary: {:.6f} ".format(f_boundary_list[str(threshold)].mean())

        print(info)


if __name__ == "__main__":
    main()
