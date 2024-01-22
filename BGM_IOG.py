import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision.models.segmentation.deeplabv3 import ASPP

from model.decoder import Decoder
from model.mobilenet import MobileNetV2Encoder
from model.refiner import Refiner
from model.resnet import ResNetEncoder
from model.utils import load_matched_state_dict

from model import MattingBase
from networks.mainnetwork import PSPModule
from networks.backbone import build_backbone
from networks.CoarseNet import CoarseNet
from networks.FineNet import FineNet


class IOGBGMNetwork(MattingBase):
    def __init__(self,
                 backbone: str = 'resnet101',
                 backbone_scale: float = 1/4,
                 refine_mode: str = 'sampling',
                 refine_sample_pixels: int = 80_000,
                 refine_threshold: float = 0.1,
                 refine_kernel_size: int = 3,
                 refine_prevent_oversampling: bool = True,
                 refine_patch_crop_method: str = 'unfold',
                 refine_patch_replace_method: str = 'scatter_nd',
                 refine_in_channels: int = 6,
                 refine_out_channels: int = 12,
                 in_channels: int = 6,
                 iog_backbone: str ='resnet101', 
                 output_stride: int = 16, 
                 num_classes: int = 1,
                 nInputChannels: int = 3,
                 sync_bn=True, 
                 freeze_bn: bool = False):
        super().__init__(backbone, in_channels, out_channels=(1 + 1 + 32))
        self.backbone_scale = backbone_scale
        self.refiner = Refiner(refine_mode,
                               refine_sample_pixels,
                               refine_threshold,
                               refine_kernel_size,
                               refine_prevent_oversampling,
                               refine_patch_crop_method,
                               refine_patch_replace_method,
                               refine_in_channels)

        print("In_channels: ", in_channels)

        output_shape = 128
        channel_settings = [512, 256, 128, 64] if iog_backbone == 'resnet18' else [512, 1024, 512, 256]      
        self.Coarse_net = CoarseNet(channel_settings, output_shape, num_classes)
        self.Fine_net = FineNet(256, output_shape, num_classes) 
        BatchNorm =  nn.BatchNorm2d
        self.iog_backbone = build_backbone(iog_backbone, output_stride, BatchNorm,nInputChannels,pretrained=False)
        psp_in_feature = 512 if iog_backbone == 'resnet18' else 2048
        self.psp4 = PSPModule(in_features=psp_in_feature, out_features=512, sizes=(1, 2, 3, 6), n_classes=256)
        self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
        if freeze_bn:
            self.freeze_bn()

        self.hid_conv1 = nn.Conv2d(256, 12, 1)
        self.hid_conv2 = nn.Conv2d(12, 1, kernel_size=3, stride=1, padding=1)


    def forward(self, src, bgr, iog_input, bgm_inputs=None):
        src_sm = src
        bgr_sm = bgr

        # Base
        if bgm_inputs is None:
            bgm_inputs = torch.cat([src_sm, bgr_sm], dim=1)
        x, *shortcuts = self.backbone(bgm_inputs)
        x = self.aspp(x)
        x = self.decoder(x, *shortcuts)
        pha_sm = x[:, 0:1]
        err_sm = torch.sigmoid(x[:, 1:2])
        hid_sm = x[:, 2: ].relu_()

        # Refiner
        pha, ref_sm, hid_bgm_12 = self.refiner(src, bgr, pha_sm, err_sm, hid_sm)

        low_level_feat_4, low_level_feat_3,low_level_feat_2,low_level_feat_1 = self.iog_backbone(iog_input)
        low_level_feat_4 = self.psp4(low_level_feat_4)   
        res_out = [low_level_feat_4, low_level_feat_3,low_level_feat_2,low_level_feat_1]   
        coarse_fms, coarse_outs = self.Coarse_net(res_out)
        fine_out, hid_iog_256 = self.Fine_net(coarse_fms)
        coarse_outs[0] = self.upsample(coarse_outs[0])
        coarse_outs[1] = self.upsample(coarse_outs[1])
        coarse_outs[2] = self.upsample(coarse_outs[2])
        coarse_outs[3] = self.upsample(coarse_outs[3])
        fine_out = self.upsample(fine_out)

        hid_iog_12 = self.hid_conv1(hid_iog_256)
        hid_iog_12 = self.upsample(hid_iog_12)
        hid_12 = hid_iog_12 + hid_bgm_12
        final_result = self.hid_conv2(hid_12)
        
        return pha, pha_sm, err_sm, ref_sm, coarse_outs, fine_out, final_result


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def Network(BGMInputChannels=6, IOGInputChannels=5,output_stride=16,
            sync_bn=None,freeze_bn=False, bgm_pretrained=False, iog_pretrained=False, **other):

    model = IOGBGMNetwork(in_channels=BGMInputChannels, nInputChannels=IOGInputChannels,
                output_stride=output_stride,sync_bn=sync_bn,freeze_bn=freeze_bn, **other)
    if bgm_pretrained:
        if 'resnet18' in bgm_pretrained:
            pretrain_dict = torch.load(bgm_pretrained)
            conv1_weight_new = np.zeros( (64,BGMInputChannels,7,7) )
            conv1_weight_new[:,:3,:,:] = pretrain_dict['conv1.weight'].cpu().data
            pretrain_dict['conv1.weight'] = torch.from_numpy(conv1_weight_new)
            state_dict = model.state_dict()
            model_dict = state_dict

            count = 0
            for k, v in pretrain_dict.items():
                kk='backbone.'+k
                if kk in state_dict:
                    model_dict[kk] = v
                    count += 1
                    print(kk, 'loaded', end='\r')
                else:
                    print(kk, 'not found')
            state_dict.update(model_dict)
            model.load_state_dict(state_dict)
            print("{} loaded in {}".format(count, bgm_pretrained))
        else:
            deeplab_dict = torch.load(bgm_pretrained)
            model.load_pretrained_deeplabv3_state_dict(deeplab_dict["model_state"])
    if iog_pretrained:
        pretrain_dict = torch.load(iog_pretrained)
        conv1_weight_new=np.zeros( (64,IOGInputChannels,7,7) )
        conv1_weight_new[:,:3,:,:]=pretrain_dict['conv1.weight'].cpu().data
        pretrain_dict['conv1.weight']=torch.from_numpy(conv1_weight_new  )
        state_dict = model.state_dict()
        model_dict = state_dict
        for k, v in pretrain_dict.items():
            kk='iog_backbone.'+k
            if kk in state_dict:
                model_dict[kk] = v
                print(kk, 'loaded', end='\r')
            else:
                print(kk, 'not found')
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)
        print("{} loaded in {}".format('iog params', iog_pretrained))

    return model