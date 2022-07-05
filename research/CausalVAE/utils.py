#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import os
import math
import random
import torch.nn as nn
from torch.utils import data
import argparse
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
def prune(A):
  zero = torch.zeros_like(A).to(device)
  A = torch.where(A < 0.3, zero, A)
  return A
def gumble_dag_loss(A):
    expm_A = torch.exp(F.gumbel_softmax(A))
    l = torch.trace(expm_A)-A.size()[0]
    return l
def filldiag_zero(A):
    mask = torch.eye(A.size()[0], A.size()[0]).byte().to(device)
    A.masked_fill_(mask, 0)
    return A
def matrix_poly(matrix, d):
    x = torch.eye(d).to(device)+ torch.div(matrix.to(device), d).to(device)
    return torch.matrix_power(x, d)
    
def mask_threshold(x):
  x = (x+0.5).int().float()
  return x
def matrix_poly(matrix, d):
    x = torch.eye(d).to(device)+ torch.div(matrix.to(device), d).to(device)
    return torch.matrix_power(x, d)
    
    
def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A
    
    
def get_parse_args():
    # parse some given arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--every_degree', '-N', type=int, default=10,
                        help='every N degree as a partition of dataset')
    args = parser.parse_args()
    return args

def weights_init(m):
    if (type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif (type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif (type(m) == nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

def compute_theta_class(theta):
    # return: the class of theta where the theta fall into
    classes_num = 18
    interval = np.linspace(0, 360, num=classes_num)
    i = 0
    for start, end in zip(interval[:-1], interval[1:]):
        if theta == 360:
            return 17
        elif start <= theta < end:
            return i
        i = i + 1

class dataload(data.Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        pil_img = Image.open(img_path)
        array = np.asarray(pil_img)
        data = torch.from_numpy(array)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img).reshape(96,96,4)
            data = torch.from_numpy(pil_img)
        return data

    def __len__(self):
        return len(self.imgs)
        
def read_label(root,idl):
    with open(root, 'r') as f:
        reader = f.readlines()
        reader = [x.replace("  "," ") for x in reader]
        reader = np.array([np.array(list(map(int,x[10:].strip().split(" ")))) for x in reader[2:]])
        reader = reader[:, idl]
        return reader[:200000]
        
        
        
class dataload_withlabel(data.Dataset):
    def __init__(self, root, dataset="train"):
        root = root + "/" + dataset
       
        imgs = os.listdir(root)

        self.dataset = dataset
        
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.imglabel = [list(map(int,k[:-4].split("_")[1:]))  for k in imgs]
        #print(self.imglabel)
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        #print(idx)
        img_path = self.imgs[idx]
        
        label = torch.from_numpy(np.asarray(self.imglabel[idx]))
        #print(len(label))
        pil_img = Image.open(img_path)
        array = np.asarray(pil_img)
        array1 = np.asarray(label)
        label = torch.from_numpy(array1)
        data = torch.from_numpy(array)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img).reshape(96,96,4)
            data = torch.from_numpy(pil_img)
        
        return data, label.float()

    def __len__(self):
        return len(self.imgs)
        

def get_partitions(every_n_degree):
    """
    :param every_n_degree: every n degree as a partition
    :return: a list of intervals where training data should fall into
    """
    partitions_anchors_num = math.floor(360 / every_n_degree)
    partitions_anchors = np.linspace(0, partitions_anchors_num * every_n_degree, num=partitions_anchors_num + 1,
                                     endpoint=True, dtype=int)
    if 360 % every_n_degree== 0:
        pass
    else:
        partitions_anchors = np.append(partitions_anchors, 360)

    partitions_list = []
    for start, end in zip(partitions_anchors[:-1], partitions_anchors[1:]):
        partitions_list.append([start, end])

    training_data_partitions = partitions_list[0::2]
    # test_data_partitions = partitions_list[1::2]
    return training_data_partitions

def whether_num_fall_into_intevals(number, intervals_list):
   """
   :param number: given number need to determine
   :param intervals_list: a list including many intervals
   :return: boolean, whether the number will fall into one of the interval of the list,
   if falls, return True; otherwise, return False
   """
   for interval in intervals_list:
       if number >= interval[0] and number < interval[1]:
           return True
   if number == interval[1]:  # consider the most endpoint
       return True
   return False
   

def get_batch_unin_dataset_withlabel(dataset_dir, batch_size, dataset="train"):
  
	dataset = dataload_withlabel(dataset_dir, dataset)
	dataset = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

	return dataset
 