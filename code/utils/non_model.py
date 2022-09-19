import numpy as np
import SimpleITK as sitk
import cv2
import os
import sys
import json
import random
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from builtins import range
import math
from math import exp

from config import opt

################################## For config ##################################
def read_kwargs(kwargs):
    ####### Common Setting #######
    if 'path_key' not in kwargs:
        print('Error: no path key')
        sys.exit()
    else:
        dict_path = '../config/%s_dict.json' % kwargs['path_key']
        with open(dict_path, 'r') as f:
            data_info_dict = json.load(f)
        kwargs['path_img'] = data_info_dict['path_img']

    if 'net_idx' not in kwargs:
        print('Error: no net idx')
        sys.exit()

    ####### Special Setting #######
    # cycle learning
    if 'cycle_r' in kwargs and int(kwargs['cycle_r']) > 0:
        if 'Tmax' not in kwargs:
            kwargs['Tmax'] = 100
        kwargs['cos_lr'] = True
        kwargs['epoch'] = int(kwargs['cycle_r']) * 2 * kwargs['Tmax']
        kwargs['gap_epoch'] = kwargs['epoch'] + 1
        kwargs['optim'] = 'SGD'

    # optim set
    if 'optim' in kwargs and kwargs['optim'] == 'SGD':
        if 'wd' not in kwargs:
            kwargs['wd'] = 0.00001
        if 'lr' not in kwargs:
            kwargs['lr'] = 0.01

    return kwargs, data_info_dict

def update_kwargs(init_model_path, kwargs):
    save_dict = torch.load(init_model_path, map_location=torch.device('cpu'))
    config_dict = save_dict['config_dict']
    del save_dict

    config_dict.pop('gpu_idx')
    config_dict['mode'] = 'test'

    if 'val_bs' in kwargs:
        config_dict['val_bs'] = kwargs['val_bs']

    return config_dict

################################## For Metric ##################################
def cal_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 40
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def cal_ssim(img1, img2, cuda_use=None):
    img1 = Variable(torch.from_numpy(deepcopy(img1))).unsqueeze(0).unsqueeze(0)
    img2 = Variable(torch.from_numpy(deepcopy(img2))).unsqueeze(0).unsqueeze(0)

    if cuda_use != None:
        img1 = img1.cuda(cuda_use)
        img2 = img2.cuda(cuda_use)

    return ssim(img1, img2).data.cpu().numpy()


