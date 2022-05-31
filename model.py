#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import pdb


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return -pairwise_distance, idx

def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            p_distance, idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            p_distance, idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return p_distance, feature      # (batch_size, 2*num_dims, num_points, k)

def convert_groupandcate_to_one_hot(grouplabels, NUM_GROUPS=50):
    # grouplabels: BxN instance lable or pid

    group_one_hot = np.zeros((grouplabels.shape[0], grouplabels.shape[1], NUM_GROUPS))
    pts_group_mask = np.zeros((grouplabels.shape[0], grouplabels.shape[1]))

    un, cnt = np.unique(grouplabels, return_counts=True)
    group_count_dictionary = dict(zip(un, cnt))
    totalnum = 0

    # as if iteritems had be replaced with items in Python 3.X
    # for k_un, v_cnt in group_count_dictionary.iteritems():
    for k_un, v_cnt in group_count_dictionary.items():
        if k_un != -1:
            totalnum += v_cnt

    for idx in range(grouplabels.shape[0]):
        un = np.unique(grouplabels[idx])
        grouplabel_dictionary = dict(zip(un, range(len(un))))
        for jdx in range(grouplabels.shape[1]):
            if grouplabels[idx, jdx] != -1:
                group_one_hot[idx, jdx, grouplabel_dictionary[grouplabels[idx, jdx]]] = 1
                # pts_group_mask[idx, jdx] = float(totalnum) / float(group_count_dictionary[grouplabels[idx, jdx]])
                pts_group_mask[idx, jdx] = 1. - float(group_count_dictionary[grouplabels[idx, jdx]]) / totalnum

    return group_one_hot.astype(np.float32), pts_group_mask

def clip_by_value(data, v_min, v_max):
    data = data.float()
    result = (data >= v_min).float() * data + (data < v_min).float() * v_min
    result = (result <= v_max).float() * result + (result > v_max).float() * v_max
    return result

class DGCNN_cls(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_cls, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        # self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.ReLU())
        self.conv1_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn1_1,
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.ReLU())
        self.conv2_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2_2,
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.ReLU())
        # self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
        #                            self.bn4,
        #                            nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(64*3, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.ReLU())

        # (batch_size, num_points, 1, 1024) -> (batch_size, 1, 1, 1024)
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size = (args.num_points, 1), stride = 1)

        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm2d(256)
        self.bn8 = nn.BatchNorm2d(128)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

        self.conv6 = nn.Sequential(nn.Conv2d(1216, 512, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, bias=False),
                                   nn.ReLU())
        self.conv9 = nn.Conv2d(128, 1, kernel_size=1, bias=False)
        self.conv10 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, bias=False),
                                   nn.ReLU())
        self.conv11 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, bias=False),
                                   nn.ReLU())
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        p_distance, x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x = self.conv1_1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x1 = torch.unsqueeze(x1, -1)

        _, x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x = self.conv2_2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x2 = torch.unsqueeze(x2, -1)

        _, x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x3 = torch.unsqueeze(x3, -1)

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.conv5(x)
        x = self.max_pool2d(x)
        expand = torch.cat([x for i in range(self.args.num_points)], dim=2)
        x = torch.cat((expand, x1, x2, x3), dim=1)

        net = self.conv6(x)
        net = self.conv7(net)

        # center prediction
        Center = self.conv8(net)

        ptscenter_logits = self.conv9(Center)
        ptscenter_logits = torch.squeeze(ptscenter_logits)
        ptscenter = self.sigmoid(ptscenter_logits)

        # Similarity matrix
        Fsim = self.conv11(net)

        Fsim = torch.squeeze(Fsim)

        Fsim = Fsim.permute(0, 2, 1)
        rr = torch.sum(Fsim * Fsim, dim=2)
        rr = torch.reshape(rr, (batch_size, -1, 1))
        D = rr - 2 * torch.matmul(Fsim, Fsim.permute(0, 2, 1)) + rr.permute(0, 2, 1)

        # simmat_logits = tf.maximum(m*D, 0.)
        # simmat is the Similarity Matrix
        simmat_logits = torch.maximum(D, torch.tensor(0.0).cuda())

        return {'center_score': ptscenter,
                'point_features':Fsim,
                'simmat': simmat_logits,
                '3d_distance':p_distance}

    def predict(self, x):
        x = torch.from_numpy(x).type(torch.float32).cuda()
        x = x.permute(0, 2, 1)
        batch_size = x.size(0)
        p_distance, x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x = self.conv1_1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x1 = torch.unsqueeze(x1, -1)

        _, x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x = self.conv2_2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x2 = torch.unsqueeze(x2, -1)

        _, x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x3 = torch.unsqueeze(x3, -1)

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.conv5(x)
        x = self.max_pool2d(x)
        expand = torch.cat([x for i in range(self.args.num_points)], dim=2)
        x = torch.cat((expand, x1, x2, x3), dim=1)

        net = self.conv6(x)
        net = self.conv7(net)

        # center prediction
        Center = self.conv8(net)

        ptscenter_logits = self.conv9(Center)
        ptscenter_logits = torch.squeeze(ptscenter_logits)
        ptscenter = self.sigmoid(ptscenter_logits)

        # Similarity matrix
        Fsim = self.conv11(net)

        Fsim = torch.squeeze(Fsim)

        Fsim = Fsim.permute(0, 2, 1)
        return {'center_score': ptscenter,
                'point_features':Fsim,
                'simmat': None,
                '3d_distance':p_distance}


    def get_loss(self, net_output, labels, vdm=True, asm=True, d_max=1, margin=[.5, 1.]):
        """
        input:
            net_output:{'center_score', 'point_features','simmat','3d_distance'}
            labels:{'ptsgroup', 'center_score', 'group_mask'}
        """
        dis_matrix = net_output['3d_distance']  # (B,N,N)
        pred_simmat = net_output['simmat']

        pts_group_label = labels['ptsgroup']
        pts_score_label = labels['center_score']

        # Similarity Matrix loss
        B = pts_group_label.shape[0]
        N = pts_group_label.shape[1]

        # onediag = torch.ones([B, N], dtype=torch.float32)

        group_mat_label = torch.matmul(pts_group_label, pts_group_label.permute(0, 2, 1))
        for unit in group_mat_label:
            unit[range(N), range(N)] = 1.0

        valid_distance_matrix = torch.less(dis_matrix, d_max).type(torch.float32)

        samegroup_mat_label = group_mat_label  # same instance matrix
        diffgroup_mat_label = torch.sub(1.0, group_mat_label)  # diferent instance matrix

        if vdm == True:
            diffgroup_mat_label = torch.mul(diffgroup_mat_label,
                                              valid_distance_matrix)  # diferent instance matrix * valid distance

        diffgroup_samesem_mat_label = diffgroup_mat_label  # FPCC does not deel with different objetcs

        num_samegroup = torch.sum(samegroup_mat_label)
        # Double hinge loss

        C_same = torch.tensor(margin[0])  # same semantic category
        C_diff = torch.tensor(margin[1])  # different semantic category
        same_ins_loss = 2 * torch.mul(samegroup_mat_label, torch.maximum(torch.sub(pred_simmat, C_same),
                                                                        torch.tensor(0).cuda()))  # minimize distances if in the same instance
        diff_ins_loss = torch.mul(diffgroup_samesem_mat_label, torch.maximum(torch.sub(C_diff, pred_simmat),
                                                                            torch.tensor(0).cuda()))  # maximum distances if in the diff instance

        # simmat_loss = alpha * neg_samesem + pos
        simmat_loss = same_ins_loss + diff_ins_loss

        # attention score matrix
        score_mask = pts_score_label
        score_mask = torch.unsqueeze(score_mask, -1)
        #multiples = torch.tensor([1,1,4096]).type(torch.int32)
        #score_mask = torch.tile(score_mask, multiples)
        score_mask = torch.cat([score_mask for i in range(self.args.num_points)], dim=2)
        score_mask = torch.add(score_mask, score_mask.permute(0, 2, 1))

        score_mask = clip_by_value(score_mask, 0, 1)

        if asm == True:
            simmat_loss = torch.mul(simmat_loss, score_mask)

        simmat_loss = torch.mean(simmat_loss)

        # loss of center point

        sigma_squared = 2
        regression_diff = torch.sub(net_output['center_score'], pts_score_label)
        regression_diff = torch.abs(regression_diff)
        regression_loss = torch.where(
            torch.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * torch.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        ptscenter_loss = torch.mean(regression_loss)

        ng_label = group_mat_label
        ng_label = torch.greater(ng_label, torch.tensor(0.5))

        ng = torch.less(pred_simmat, torch.tensor(margin[0]))

        loss = simmat_loss + 3 * ptscenter_loss

        grouperr = torch.abs(ng.type(torch.float32) - ng_label.type(torch.float32))
        # 计算分group错误的点的数量。group_error

        return loss, ptscenter_loss, torch.mean(grouperr)
