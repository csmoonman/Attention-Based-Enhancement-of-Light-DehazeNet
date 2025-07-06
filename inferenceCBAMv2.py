# @author: hayat (modified for CBAMv2)
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import numpy as np
from torchvision import transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import lightdehazeNetCBAMv2

def image_haze_removel(input_image, gt_image=None):

    # 处理有雾图像
    hazy_image = (np.asarray(input_image)/255.0)
    hazy_image = torch.from_numpy(hazy_image).float()
    hazy_image = hazy_image.permute(2,0,1)
    hazy_image = hazy_image.cuda().unsqueeze(0)

    # 加载模型和权重
    ld_net = lightdehazeNetCBAMv2.LightDehaze_Net().cuda()
    ld_net.load_state_dict(torch.load('./trained_weights/trained_LDNetCBAM-3.pth'), strict=False)
    ld_net.eval()

    # 进行去雾处理
    with torch.no_grad():
        dehaze_image = ld_net(hazy_image)
    
    # 计算图像质量指标（如果提供了Ground Truth图像）
    metrics = {}
    if gt_image is not None:
        # 处理Ground Truth图像
        gt_np = np.array(gt_image) / 255.0
        
        # 将去雾后的图像转换为numpy数组
        dehazed_np = dehaze_image.squeeze().permute(1, 2, 0).cpu().numpy()
        dehazed_np = np.clip(dehazed_np, 0, 1)
        
        # 确保两个图像具有相同的尺寸和通道数
        if gt_np.shape == dehazed_np.shape:
            # 计算PSNR
            metrics['psnr'] = psnr(gt_np, dehazed_np)
            
            # 计算SSIM (多通道)
            metrics['ssim'] = ssim(gt_np, dehazed_np, multichannel=True, channel_axis=2)
            
            # 计算MSE
            metrics['mse'] = mse(gt_np, dehazed_np)
    
    return dehaze_image, metrics