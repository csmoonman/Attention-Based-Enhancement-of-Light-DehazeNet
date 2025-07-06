# @author: hayat
import os
import sys
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random
import cv2

random.seed(1143)


def preparing_training_data(hazefree_images_dir, hazeeffected_images_dir):
    train_data = []
    validation_data = []
    
    # 使用os.path.join构建查找路径，避免路径分隔符问题
    hazy_data = glob.glob(os.path.join(hazeeffected_images_dir, "*.jpg"))

    data_holder = {}

    for h_image in hazy_data:
        # 提取文件名，避免路径分隔符问题
        h_image_name = os.path.basename(h_image)
        id_ = h_image_name.split("_")[0] + "_" + h_image_name.split("_")[1] + ".jpg"
        if id_ in data_holder.keys():
            data_holder[id_].append(h_image_name)
        else:
            data_holder[id_] = []
            data_holder[id_].append(h_image_name)


    train_ids = []
    val_ids = []

    num_of_ids = len(data_holder.keys())
    for i in range(num_of_ids):
        if i < num_of_ids*9/10:
            train_ids.append(list(data_holder.keys())[i])
        else:
            val_ids.append(list(data_holder.keys())[i])


    for id_ in list(data_holder.keys()):
        if id_ in train_ids:
            for hazy_image in data_holder[id_]:
                # 使用os.path.join正确拼接路径
                train_data.append([
                    os.path.join(hazefree_images_dir, id_), 
                    os.path.join(hazeeffected_images_dir, hazy_image)
                ])
        else:
            for hazy_image in data_holder[id_]:
                # 使用os.path.join正确拼接路径
                validation_data.append([
                    os.path.join(hazefree_images_dir, id_), 
                    os.path.join(hazeeffected_images_dir, hazy_image)
                ])

    random.shuffle(train_data)
    random.shuffle(validation_data)

    return train_data, validation_data

    

class hazy_data_loader(data.Dataset):

    def __init__(self, hazefree_images_dir, hazeeffected_images_dir, mode='train'):
        # 保持原有调用方式，仅修复传入参数
        self.train_data, self.validation_data = preparing_training_data(hazefree_images_dir, hazeeffected_images_dir)

        if mode == 'train':
            self.data_dict = self.train_data
            print("Number of Training Images:", len(self.train_data))
        else:
            self.data_dict = self.validation_data
            print("Number of Validation Images:", len(self.validation_data))

        

    def __getitem__(self, index):
        hazefree_image_path, hazy_image_path = self.data_dict[index]

        try:
            hazefree_image = Image.open(hazefree_image_path)
            hazy_image = Image.open(hazy_image_path)

            hazefree_image = hazefree_image.resize((480,640), Image.ANTIALIAS)
            hazy_image = hazy_image.resize((480,640), Image.ANTIALIAS)

            hazefree_image = (np.asarray(hazefree_image)/255.0) 
            hazy_image = (np.asarray(hazy_image)/255.0) 

            hazefree_image = torch.from_numpy(hazefree_image).float()
            hazy_image = torch.from_numpy(hazy_image).float()

            return hazefree_image.permute(2,0,1), hazy_image.permute(2,0,1)
            
        except Exception as e:
            # 添加错误处理，防止程序中断
            print(f"Error loading image: {e}")
            print(f"Problem with paths: {hazefree_image_path} or {hazy_image_path}")
            # 返回零张量作为替代
            empty_tensor = torch.zeros(3, 640, 480)
            return empty_tensor, empty_tensor

    def __len__(self):
        return len(self.data_dict)