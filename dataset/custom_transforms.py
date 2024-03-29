import torch, cv2
import numpy.random as random
import numpy as np
import scipy.misc as sm

import dataset.helpers as helpers
from dataset.helpers import *

class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
        2.  rots [list]: list of fixed possible rotation angles
            scales [list]: list of fixed possible scales
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25), semseg=False):
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales
        self.semseg = semseg

    def __call__(self, sample):

        if type(self.rots) == tuple:
            # Continuous range of scales and rotations
            rot = (self.rots[1] - self.rots[0]) * random.random() - \
                  (self.rots[1] - self.rots[0])/2

            sc = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1
        elif type(self.rots) == list:
            # Fixed range of scales and rotations
            rot = self.rots[random.randint(0, len(self.rots))]
            sc = self.scales[random.randint(0, len(self.scales))]

        for elem in sample.keys():
            if 'meta' in elem:
                continue
            
            tmp = sample[elem]

            # try:
            h, w = tmp.shape[:2]
            # except:
            #     import pdb;pdb.set_trace()
            center = (w / 2, h / 2)
            assert(center != 0)  # Strange behaviour warpAffine
            M = cv2.getRotationMatrix2D(center, rot, sc)

            if ((tmp == 0) | (tmp == 1)).all():
                flagval = cv2.INTER_NEAREST
            elif 'gt' in elem and self.semseg:
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC
            tmp = cv2.warpAffine(tmp, M, (w, h), flags=flagval)

            sample[elem] = tmp

        return sample

    def __str__(self):
        return 'ScaleNRotate:(rot='+str(self.rots)+',scale='+str(self.scales)+')'


class FixedResize(object):
    """Resize the image and the ground truth to specified resolution.
    Args:
        resolutions (dict): the list of resolutions
    """
    def __init__(self, resolutions=None, flagvals=None):
        self.resolutions = resolutions
        self.flagvals = flagvals
        if self.flagvals is not None:
            assert(len(self.resolutions) == len(self.flagvals))

    def __call__(self, sample):

        # Fixed range of scales
        if self.resolutions is None:
            return sample

        elems = list(sample.keys())

        for elem in elems:

            if 'meta' in elem or 'bbox' in elem or ('extreme_points_coord' in elem and elem not in self.resolutions):
                continue
            if 'extreme_points_coord' in elem and elem in self.resolutions:
                bbox = sample['bbox']
                crop_size = np.array([bbox[3]-bbox[1]+1, bbox[4]-bbox[2]+1])
                res = np.array(self.resolutions[elem]).astype(np.float32)
                sample[elem] = np.round(sample[elem]*res/crop_size).astype(np.int)
                continue
            if elem in self.resolutions:
                if self.resolutions[elem] is None:
                    continue
                if isinstance(sample[elem], list):
                    if sample[elem][0].ndim == 3:
                        output_size = np.append(self.resolutions[elem], [3, len(sample[elem])])
                    else:
                        output_size = np.append(self.resolutions[elem], len(sample[elem]))
                    tmp = sample[elem]
                    sample[elem] = np.zeros(output_size, dtype=np.float32)
                    for ii, crop in enumerate(tmp):
                        if self.flagvals is None:
                            sample[elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem])
                        else:
                            sample[elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem], flagval=self.flagvals[elem])
                else:
                    if self.flagvals is None:
                        sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem])
                    else:
                        sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem], flagval=self.flagvals[elem])
            else:
                del sample[elem]

        return sample

    def __str__(self):
        return 'FixedResize:'+str(self.resolutions)


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                if 'meta' in elem:
                    continue
                tmp = sample[elem]
                tmp = cv2.flip(tmp, flipCode=1)
                sample[elem] = tmp

        return sample

    def __str__(self):
        return 'RandomHorizontalFlip'


class Inpaint(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
    def __init__(self, target='crop_img', scribble='crop_pad'):
        self.target = target
        self.scribble = scribble
        # self.pad_pixel =pad_pixel

    def __call__(self, sample):
        target = sample[self.target]
        scribble = sample[self.scribble]

        target = np.copy(target).astype(np.uint8)
        bgr = cv2.inpaint(target, scribble, 3, cv2.INPAINT_TELEA)

        sample['crop_bgr'] = bgr

        return sample

    def __str__(self):
        return 'RandomHorizontalFlip'


class IOGPoints(object):
    """
    Returns the IOG Points (top-left and bottom-right or top-right and bottom-left) in a given binary mask
    sigma: sigma of Gaussian to create a heatmap from a point
    pad_pixel: number of pixels fo the maximum perturbation
    elem: which element of the sample to choose as the binary mask
    """
    def __init__(self, sigma=10, elem='crop_gt', pad_pixel=10):
        self.sigma = sigma
        self.elem = elem
        self.pad_pixel =pad_pixel

    def __call__(self, sample):

        if sample[self.elem].ndim == 3:
            raise ValueError('IOGPoints not implemented for multiple object per image.')
        _target = sample[self.elem]

        targetshape=_target.shape
        if np.max(_target) == 0:
            sample['IOG_points'] = np.zeros([targetshape[0],targetshape[1],2], dtype=_target.dtype) #  TODO: handle one_mask_per_point case
        else:
            _points = helpers.iog_points(_target, self.pad_pixel)
            sample['IOG_points'] = helpers.make_gt(_target, _points, sigma=self.sigma, one_mask_per_point=False)

        return sample

    def __str__(self):
        return 'IOGPoints:(sigma='+str(self.sigma)+', pad_pixel='+str(self.pad_pixel)+', elem='+str(self.elem)+')'


class ThinPartPoint(object):
    def __init__(self, sigma=10, elem='IOG_points', thin_pad='crop_pad', gt_mask='crop_mask'):
        self.sigma = sigma
        self.elem = elem
        self.thin_pad = thin_pad
        self.gt_mask = gt_mask

    def __call__(self, sample):
        iog_point = sample[self.elem]
        thin_pad = sample[self.thin_pad]
        gt_mask = sample[self.gt_mask]

        # print(np.unique(thin_pad))
        # print(np.unique(gt_mask))
        # import pdb;pdb.set_trace()

        binary = thin_pad.astype(np.uint8)

        ret, labels = cv2.connectedComponents(binary)

        thin_center_list = helpers.thin_center_point(thin_pad, labels, ret)
        thin_bg = np.copy(thin_pad)
        thin_bg[gt_mask > 128] = 0
        thin_bg_list = helpers.thin_bg_point(thin_bg, labels, ret)

        iog_thin_point = helpers.make_thin_points(iog_point, thin_center_list, thin_bg_list, sigma=self.sigma)

        # import imageio
        # imageio.imwrite("tmp/thin_pad.png", thin_pad)
        # imageio.imwrite("tmp/gt_mask.png", gt_mask)
        # imageio.imwrite("tmp/thin_bg.png", thin_bg)
        # imageio.imwrite("tmp/point_0.png", iog_thin_point[:, :, 0])
        # imageio.imwrite("tmp/point_1.png", iog_thin_point[:, :, 1])
        # import pdb;pdb.set_trace()
        sample['meta']['thin_count'] = ret - 1

        sample[self.elem] = iog_thin_point

        return sample

    def __str__(self):
        return 'sim thin parts center and backgorund points'


class ThinPartPointDilate(object):
    def __init__(self, sigma=10, elem='IOG_points', thin_gt='crop_gt', gt_mask='crop_mask'):
        self.sigma = sigma
        self.elem = elem
        self.thin_gt = thin_gt
        self.gt_mask = gt_mask
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def __call__(self, sample):
        iog_point = sample[self.elem]
        thin_gt = sample[self.thin_gt]
        gt_mask = sample[self.gt_mask]
        k_r = sample['meta']['kernel_r']

        # print(np.unique(thin_pad))
        # print(np.unique(gt_mask))
        # import pdb;pdb.set_trace()

        binary = thin_gt.astype(np.uint8)
        # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.open_kernel)
        ret, labels = cv2.connectedComponents(binary)
        thin_center_list = helpers.thin_center_point(thin_gt, labels, ret)

        thin_pad_bg_list = []
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + 2 * k_r, 1 + 2 * k_r))
        for i in range(1, ret):
            thin_bg_pad_ = np.zeros_like(thin_gt, dtype=np.uint8)
            thin_bg_pad_[labels == i] = 255
            thin_bg_pad_ = cv2.dilate(thin_bg_pad_, kernel, iterations=1)
            thin_bg_pad_[gt_mask > 128] = 0
            thin_pad_bg_list.append(thin_bg_pad_)

        # thin_bg = np.copy(thin_pad)
        # thin_bg[gt_mask > 128] = 0
        thin_bg_list = helpers.thin_bg_point_dilate(thin_pad_bg_list)

        iog_thin_point = helpers.make_thin_points(iog_point, thin_center_list, thin_bg_list, sigma=self.sigma)

        # import imageio
        # imageio.imwrite("tmp/thin_pad.png", thin_pad)
        # imageio.imwrite("tmp/gt_mask.png", gt_mask)
        # imageio.imwrite("tmp/thin_bg.png", thin_bg)
        # imageio.imwrite("tmp/point_0.png", iog_thin_point[:, :, 0])
        # imageio.imwrite("tmp/point_1.png", iog_thin_point[:, :, 1])
        # import pdb;pdb.set_trace()
        sample['meta']['thin_count'] = ret - 1

        sample[self.elem] = iog_thin_point

        return sample

    def __str__(self):
        return 'sim thin parts center and backgorund points'


class ConcatInputs(object):

    def __init__(self, elems=('image', 'point')):
        self.elems = elems

    def __call__(self, sample):

        res = sample[self.elems[0]]

        for elem in self.elems[1:]:
            assert(sample[self.elems[0]].shape[:2] == sample[elem].shape[:2])

            # Check if third dimension is missing
            tmp = sample[elem]
            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            res = np.concatenate((res, tmp), axis=2)

        sample['concat'] = res
        return sample

    def __str__(self):
        return 'ExtremePoints:'+str(self.elems)


class CropFromMask_h(object):
    """
    Returns image cropped in bounding box from a given mask
    """
    def __init__(self, crop_elems=('img', 'gt','bgr'),
                 mask_elem='gt',
                 relax=0,
                 zero_pad=False,
                 adaptive_relax=False):

        self.crop_elems = crop_elems
        self.mask_elem = mask_elem
        self.relax = relax
        self.zero_pad = zero_pad
        self.adaptive_relax = adaptive_relax

    def __call__(self, sample):
        _target = sample[self.mask_elem]
        if _target.ndim == 2:
            _target = np.expand_dims(_target, axis=-1)

        bbox_list = []
        for k in range(0, _target.shape[-1]):
            # bbox_list.append(get_bbox(_target[..., k], pad=self.relax, zero_pad=self.zero_pad))
            if self.adaptive_relax:
                bbox = helpers.get_bbox(_target[..., k], pad=0, zero_pad=self.zero_pad)
                if bbox is None:
                    bbox_list.append(None)
                    sample['meta']['relax'] = self.relax
                else:
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    edge = (w + h) / 2
                    relax = int(edge / 162. * self.relax)
                    sample['meta']['relax'] = relax
                    bbox_list.append(helpers.get_bbox(_target[..., k], pad=relax, zero_pad=self.zero_pad))
            else:
                sample['meta']['relax'] = self.relax
                bbox_list.append(helpers.get_bbox(_target[..., k], pad=self.relax, zero_pad=self.zero_pad))


        for elem in self.crop_elems:
            _img = sample[elem]
            _crop = []
            if self.mask_elem == elem:
                if _img.ndim == 2:
                    _img = np.expand_dims(_img, axis=-1)
                for k in range(0, _target.shape[-1]):
                    _tmp_img = _img[..., k]
                    _tmp_target = _target[..., k]
                    if np.max(_target[..., k]) == 0:
                        _crop.append(np.zeros(_tmp_img.shape, dtype=_img.dtype))
                    else:
                        _crop.append(helpers.crop_from_bbox(_tmp_img, bbox_list[k], zero_pad=self.zero_pad))
            else:
                for k in range(0, _target.shape[-1]):
                    if np.max(_target[..., k]) == 0:
                        _crop.append(np.zeros(_img.shape, dtype=_img.dtype))
                    else:
                        _tmp_target = _target[..., k]
                        _crop.append(helpers.crop_from_bbox(_img, bbox_list[k], zero_pad=self.zero_pad))
            if len(_crop) == 1:
                sample['crop_' + elem] = _crop[0]
            else:
                sample['crop_' + elem] = _crop

        return sample

    def __str__(self):
        return 'CropFromMask:(crop_elems='+str(self.crop_elems)+', mask_elem='+str(self.mask_elem)+\
               ', relax='+str(self.relax)+',zero_pad='+str(self.zero_pad)+')'


class CropFromMask(object):
    """
    Returns image cropped in bounding box from a given mask
    """
    def __init__(self, crop_elems=('img', 'gt','bgr'),
                 mask_elem='gt',
                 relax=0,
                 zero_pad=False):

        self.crop_elems = crop_elems
        self.mask_elem = mask_elem
        self.relax = relax
        self.zero_pad = zero_pad

    def __call__(self, sample):
        _target = sample[self.mask_elem]
        if _target.ndim == 2:
            _target = np.expand_dims(_target, axis=-1)
        for elem in self.crop_elems:
            _img = sample[elem]
            _crop = []
            if self.mask_elem == elem:
                if _img.ndim == 2:
                    _img = np.expand_dims(_img, axis=-1)
                for k in range(0, _target.shape[-1]):
                    _tmp_img = _img[..., k]
                    _tmp_target = _target[..., k]
                    if np.max(_target[..., k]) == 0:
                        _crop.append(np.zeros(_tmp_img.shape, dtype=_img.dtype))
                    else:
                        _crop.append(helpers.crop_from_mask(_tmp_img, _tmp_target, relax=self.relax, zero_pad=self.zero_pad))
            else:
                for k in range(0, _target.shape[-1]):
                    if np.max(_target[..., k]) == 0:
                        _crop.append(np.zeros(_img.shape, dtype=_img.dtype))
                    else:
                        _tmp_target = _target[..., k]
                        _crop.append(helpers.crop_from_mask(_img, _tmp_target, relax=self.relax, zero_pad=self.zero_pad))
            if len(_crop) == 1:
                sample['crop_' + elem] = _crop[0]
            else:
                sample['crop_' + elem] = _crop

        return sample

    def __str__(self):
        return 'CropFromMask:(crop_elems='+str(self.crop_elems)+', mask_elem='+str(self.mask_elem)+\
               ', relax='+str(self.relax)+',zero_pad='+str(self.zero_pad)+')'


class ToImage(object):
    """
    Return the given elements between 0 and 255
    """
    def __init__(self, norm_elem='image', custom_max=255.):
        self.norm_elem = norm_elem
        self.custom_max = custom_max

    def __call__(self, sample):
        if isinstance(self.norm_elem, tuple):
            for elem in self.norm_elem:
                tmp = sample[elem]
                sample[elem] = self.custom_max * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        else:
            tmp = sample[self.norm_elem]
            sample[self.norm_elem] = self.custom_max * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        return sample

    def __str__(self):
        return 'NormalizeImage'


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem:
                continue
            elif 'bbox' in elem:
                tmp = sample[elem]
                sample[elem] = torch.from_numpy(tmp)
                continue

            tmp = sample[elem]

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            # from 0-255 to 0-1
            tmp = tmp.transpose((2, 0, 1))
            tmp = torch.from_numpy(tmp).type(torch.float)
            sample[elem] = tmp / 255

        return sample

    def __str__(self):
        return 'ToTensor'
