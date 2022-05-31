#!/usr/bin/env python

import os
import sys
import glob
import h5py
import numpy as np
import torch
import json
import cv2
from torch.utils.data import Dataset

from config import args
import provider
    

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud

def get_data():
    TRAINING_FILE_LIST = args.input_list
    BATCH_SIZE = args.batch_size
    train_file_list = provider.getDataFiles(TRAINING_FILE_LIST)
    num_train_file = len(train_file_list)

    train_file_idx = np.arange(0, len(train_file_list))
    np.random.shuffle(train_file_idx)

    ## load all data into memory
    all_data = []
    all_group = []
    all_seg = []
    all_score = []
    for i in range(num_train_file):
        cur_train_filename = train_file_list[train_file_idx[i]]
        # printout(flog, 'Loading train file ' + cur_train_filename + '\t' + str(i) + '/' + str(num_train_file))
        cur_data, cur_group, _, _, cur_score = provider.loadDataFile_with_groupseglabel_stanfordindoor(
            cur_train_filename)
        # cur_data = cur_data.reshape([-1,4096,3])

        all_data += [cur_data]
        all_group += [cur_group]
        all_score += [cur_score]

    all_data = np.concatenate(all_data, axis=0)
    all_group = np.concatenate(all_group, axis=0)
    all_score = np.concatenate(all_score, axis=0)

    num_data = all_data.shape[0]
    num_batch = num_data // BATCH_SIZE
    return all_data, all_group, all_score

class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.group, self.score = get_data()
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.score[item]
        group = self.group[item]
        # if self.partition == 'train':
        #     pointcloud = translate_pointcloud(pointcloud)
        #     np.random.shuffle(pointcloud)
        return pointcloud, label, group

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    data, label = train[0]
    print(data.shape)
    print(label.shape)
