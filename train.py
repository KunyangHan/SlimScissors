import argparse
import cv2
import torch
import os
import shutil
import numpy as np

from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from torchvision.utils import save_image

from dataset import ThinObjectDataset
from dataset import custom_transforms as tr
from networks.lr_schedule import LR_Scheduler

from networks.loss import class_cross_entropy_loss
from loss_function import normalized_l1_loss
from model.ritm_loss import NormalizedFocalLossSigmoid
from BGM_IOG import Network


# --------------- Arguments ---------------
parser = argparse.ArgumentParser(description='Train refine')

parser.add_argument('--model-type', type=str, default='mattingrefine')
parser.add_argument('--model-backbone', type=str, choices=['resnet101', 'resnet18', 'resnet50', 'mobilenetv2'], default='resnet18')
parser.add_argument('--model-backbone-scale', type=float, default=0.25)
parser.add_argument('--model-refine-mode', type=str, default='full', choices=['full', 'sampling', 'thresholding'])
parser.add_argument('--model-refine-sample-pixels', type=int, default=80000)
parser.add_argument('--model-refine-threshold', type=float, default=0.1)
parser.add_argument('--model-refine-kernel-size', type=int, default=3)

parser.add_argument('--batch_size', default=25, type=int, metavar='BT',
                    help='batch size')

parser.add_argument('--lr', '--learning_rate', default='1e-4', type=str,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='initial momentum')
parser.add_argument('--weight_decay', '--wd', default='0.0001', type=str,
                    help='initial weight decay')
parser.add_argument('--maxepoch', default=30, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--test_freq', default=5, type=int,
                    help='test frequency')
parser.add_argument('--gpu', default='0', type=str, help='GPU ID')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--ckpt', help='ckpt folder', default='ckpt')

parser.add_argument("--optim", type=str, default='adam')
parser.add_argument("--lr_scheduler", type=str, default='linear')
parser.add_argument("--lr_step", type=float, default=20)
parser.add_argument("--warmup_epochs", type=float, default=5)
parser.add_argument("--target_lr", type=str, default='1e-5')
parser.add_argument("--start_lr", type=float, default=0)

parser.add_argument('--adp_ratio', type=float, nargs='+', default=[8, 12, 16, 20, 24])
parser.add_argument('--threshold', type=int, default=2)
parser.add_argument('--dataset', help='root folder of dataset', default='../data')

parser.add_argument('--adaptive_relax', type=bool, default=False)
parser.add_argument('--base_weight', type=float, default=0.5)
parser.add_argument('--coarse_weight', type=float, default=0.5)

parser.add_argument('--use_deep_inpaint', type=bool, default=False)
parser.add_argument('--delete_err', type=bool, default=False)

args = parser.parse_args()

args.lr = float(args.lr)
args.weight_decay = float(args.weight_decay)
args.target_lr = float(args.target_lr)

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
    bgm_dic = {'resnet101' : './best_deeplabv3_resnet101_voc_os16.pth', 
        'resnet50': './best_deeplabv3_resnet50_voc_os16.pth',
        'resnet18': './resnet18-5c106cde.pth'}
    iog_dic = {'resnet101' : "./resnet101-5d3b4d8f.pth", 
        'resnet50': './resnet50-19c8e357.pth',
        'resnet18': './resnet18-5c106cde.pth'}
    net = Network(BGMInputChannels=BGM_Input_Channel, 
                    IOGInputChannels=IOG_Input_Channel, 
                    output_stride=16,
                    sync_bn=None,
                    freeze_bn=False,
                    bgm_pretrained=bgm_dic[args.model_backbone],
                    iog_pretrained=iog_dic[args.model_backbone], 
                    refine_mode='full',
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

    base_params = list(map(id, net.backbone.parameters())) + \
        list(map(id, net.iog_backbone.parameters()))
    logits_params = filter(lambda p: id(p) not in base_params, net.parameters())
    train_params = [
        {"params": net.backbone.parameters(), "lr": args.lr},
        {"params": net.iog_backbone.parameters(), "lr": args.lr},
        {"params": logits_params, "lr": args.lr * 10},
    ]

    if args.optim == "adam":
        print("Using Adam as optim")
        optimizer = optim.Adam(train_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        print("Using SGD as optim")
        optimizer = optim.SGD(train_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    opti_id = set()
    for i, group in enumerate(optimizer.param_groups):
        g_id = list(map(id, group['params']))
        opti_id.update(g_id)

    for net_m in net.named_parameters():
        if id(net_m[1]) not in opti_id:
            print(net_m[0])

    composed_transforms_ts = transforms.Compose([
        tr.CropFromMask_h(crop_elems=('img', 'gt','bgr', 'mask', 'pad', 'region'), 
            mask_elem='mask', relax=30, zero_pad=True, adaptive_relax=args.adaptive_relax),
        tr.FixedResize(
            resolutions={'crop_img': (512, 512), 'crop_gt': (512, 512), 'crop_bgr': (512, 512),
                'crop_mask': (512, 512), 'crop_region': (512, 512), 'crop_pad': (512, 512)}, 
            flagvals={'crop_img': cv2.INTER_LINEAR,'crop_gt': cv2.INTER_NEAREST,'crop_bgr': cv2.INTER_LINEAR, 
                'crop_mask': cv2.INTER_NEAREST, 'crop_region' : cv2.INTER_NEAREST, 'crop_pad' : cv2.INTER_NEAREST}),
        tr.IOGPoints(sigma=10, elem='crop_mask', pad_pixel=10),
        tr.ToImage(norm_elem='IOG_points'),
        tr.ToTensor()])

    composed_transforms_tr = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25)),
        composed_transforms_ts])

    train_dateset = ThinObjectDataset(args.dataset, "train", transforms=composed_transforms_tr, 
        region_size=args.threshold, adp_ratio=args.adp_ratio)
    train_loader = DataLoader(train_dateset, batch_size=args.batch_size, shuffle=True, num_workers=32)

    schedule = LR_Scheduler(args.lr_scheduler, args.lr, args.maxepoch, len(train_loader), 
        lr_step=args.lr_step, warmup_epochs=args.warmup_epochs, target_lr=args.target_lr,
        start_lr=args.start_lr)

    for epoch in range(args.start_epoch, args.maxepoch):
        train(net, train_loader, optimizer, epoch, schedule)

        if (epoch + 1) % 5 == 0:
            save_file = os.path.join(CKPT_DIR, 'checkpoint_epoch_{}.pth'.format(epoch))
            torch.save({
                'epoch' : epoch,
                'state_dict' : net.state_dict(),
            }, save_file)


def train(net, dataloader, optimizer, epoch, schedule):
    net.train()

    focal_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_acm = 0
    for i, sample_batched in enumerate(dataloader):
        src = sample_batched["crop_img"]
        bgr = sample_batched["crop_bgr"]
        pad = sample_batched["crop_pad"]
        gt = sample_batched["crop_gt"]
        mask = sample_batched["crop_mask"]
        region = sample_batched["crop_region"]
        point = sample_batched["IOG_points"]

        void_thin = torch.ones(region.size())
        void_thin[region > 0.3] = 0 

        src, bgr, gt = src.cuda(), bgr.cuda(), gt.cuda() 
        mask, point, void_thin = mask.cuda(), point.cuda(), void_thin.cuda()
        pad = pad.cuda()

        '''
        src, (batch_size, 3, h, w), original image
        bgr, (batch_size, 3, h, w), opencv inpainted background image
        gt, (batch_size, 1, h, w), thin region mask
        mask, (batch_size, 1, h, w), whole mask
        point, (batch_size, 2, h, w), foreground and background point
        '''

        IOG_input = torch.cat([src, point], dim=1)
        BGM_input = torch.cat([src, bgr, pad], dim=1)
        refine_concat = torch.cat([bgr, pad], dim=1)

        pha, pha_sm, err_sm, ref_sm, coarse_outs, fine_out, final_result = net(src, refine_concat, IOG_input, BGM_input)

        fg_region = (gt > 0.5).type(torch.cuda.FloatTensor)
        gt_base_err = torch.abs(gt - pha_sm)

        loss_base_alpha = class_cross_entropy_loss(pha_sm, gt)
        loss_base_err = F.mse_loss(err_sm, gt_base_err)
        loss_refine_alpha = class_cross_entropy_loss(pha, gt, void_pixels=void_thin)
        
        bgm_loss = args.base_weight * loss_base_alpha + args.base_weight * loss_base_err + loss_refine_alpha

        loss_coarse_outs1 = class_cross_entropy_loss(coarse_outs[0], mask)
        loss_coarse_outs2 = class_cross_entropy_loss(coarse_outs[1], mask)
        loss_coarse_outs3 = class_cross_entropy_loss(coarse_outs[2], mask)
        loss_coarse_outs4 = class_cross_entropy_loss(coarse_outs[3], mask)
        loss_fine_out = class_cross_entropy_loss(fine_out, mask)
        iog_loss = args.coarse_weight * loss_coarse_outs1 + \
            args.coarse_weight * loss_coarse_outs2 + \
            args.coarse_weight * loss_coarse_outs3 + \
            args.coarse_weight * loss_coarse_outs4 + \
            loss_fine_out

        final_mask_loss = class_cross_entropy_loss(final_result, mask)
        final_loss = bgm_loss + iog_loss + final_mask_loss

        schedule(optimizer, i, epoch, ten_time_group=[2])

        final_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_acm += final_loss.item()

        if (i + 1) % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, i, len(dataloader)) + \
                'Loss {} '.format(loss_acm / args.print_freq) + \
                'backbone lr {:.10f} '.format(optimizer.param_groups[0]['lr']) + \
                'iog backbone lr {:.10f} '.format(optimizer.param_groups[1]['lr']) + \
                'other lr {:.10f} '.format(optimizer.param_groups[2]['lr'])
            print(info)
            loss_acm = 0


def ce_mask_loss(pred, target, th):
    kernel = np.ones((7, 7), dtype=np.uint8)

    mask = torch.zeros(pred.size())
    mask[target > 0.3] = 1
    mask = torch2np(mask)

    bs, c, h, w = pred.size()
    mask_list = []
    for n in range(bs):
        mask_n = np.squeeze(mask[n])

        thin_num = (mask_n > 128).sum()
        total_num = thin_num
        c = 0
        while thin_num * (th + 1) >= total_num and total_num < 260_000:
            mask_n = cv2.dilate(mask_n, kernel, iterations=1)
            total_num = (mask_n > 128).sum()
            c += 1
            if c > 20:
                break
        
        mask_list.append(np.expand_dims(mask_n, 0))

    mask_list = np.concatenate(mask_list)
    mask_list = np.expand_dims(mask_list, -1)
    mask_list = np2torch(mask_list)

    assert mask_list.size() == pred.size()

    void_thin = torch.ones(pred.size())
    void_thin[mask_list > 0.3] = 0
    void_thin = void_thin.cuda()

    return class_cross_entropy_loss(pred, target, void_pixels=void_thin)
    

def torch2np(a):
    return np.array(a.permute(0, 2, 3, 1).cpu() * 255).astype(np.uint8)


def np2torch(a):
    a = torch.from_numpy(a.transpose((0, 3, 1, 2))).type(torch.float) / 255
    return a.cuda()


def save_result(save_dir, filename, img_list, name_list):
    for img, name in zip(img_list, name_list):
        save_image(img, os.path.join(save_dir, filename + name + '.png'))


if __name__ == "__main__":
    main()
