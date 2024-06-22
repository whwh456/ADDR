import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import cKDTree
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
# from . import pytorch_utils as pt_utils

import numpy as np
from sklearn.metrics import confusion_matrix


class PCDR(nn.Module):

    def __init__(self, num_coarse):
        super(PCDR,self).__init__()

        self.num_coarse = num_coarse
        self.latent_dim = 1024

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 3, 1),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True)
        )

        self.num_layers = 3
        self.d_out = [8, 32, 128]

        self.dilated_res_blocks = nn.ModuleList()
        d_in = 3
        for i in range(self.num_layers):
            d_out = self.d_out[i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out
        d_out = d_in

        self.att_pooling_1 = Att_pooling_2(d_out,d_out)
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.Drb = Dilated_res_block(512, 512)
        self.att_pooling_2 = Att_pooling_2(1024, 1024)

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )

    def forward(self, xyz):
        B, N, _  = xyz.shape
        features = self.first_conv(xyz.transpose(1,2))
        features = features.unsqueeze(dim=3)

        neighbour_idx = self.get_neigh_index(xyz)
        neigh_idx= neighbour_idx.to("cuda:0")

        f_encoder_list = []
        for i in range(self.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](features, xyz, neigh_idx)
            features = f_encoder_i
            f_encoder_list.append(f_encoder_i)


        features = f_encoder_list[-1]
        feature = features.squeeze(-1)
        feature_global = self.att_pooling_1(feature)
        feature_agg = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)
        feature_agg = self.second_conv(feature_agg)
        feature_agg = feature_agg.unsqueeze(dim=3)

        features = self.Drb(feature_agg, xyz, neigh_idx)
        feature = features.squeeze(-1)
        feature_global = self.att_pooling_2(feature).squeeze(-1)

        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)
        return coarse.contiguous()

    def get_neigh_index(self,point):
        points = point.detach().cpu()
        Bath_size = points.shape[0]
        neigh_idxs = []
        for i in range(Bath_size):
            kdtree = cKDTree(points[i])
            k = 16  # 指定每个点的相邻点个数为16
            distances, neigh_idx = kdtree.query(points[i], k=k + 1)
            neigh_idx = neigh_idx[:, 1:]  # 排除自身
            neigh_idxs.append(neigh_idx)
        neigh_idxs = np.array(neigh_idxs)
        return torch.tensor(neigh_idxs)


class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = nn.Sequential(
            nn.Conv2d(d_in, d_out // 2, kernel_size=(1, 1)),
            nn.BatchNorm2d(d_out//2),
        )
        self.lfa = Building_block(d_out)
        self.mlp2 = nn.Sequential(
            nn.Conv2d(d_out, d_out*2, kernel_size=(1, 1)),
            nn.BatchNorm2d(d_out*2),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(d_in, d_out * 2, kernel_size=(1, 1)),
            nn.BatchNorm2d(d_out * 2),
        )

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)
        f_pc = self.lfa(xyz, f_pc, neigh_idx)
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc + shortcut, negative_slope=0.2)


class Building_block(nn.Module):
    def __init__(self, d_out):  # d_in = d_out//2
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv2d(10, d_out // 2, kernel_size=(1, 1)),
            nn.BatchNorm2d(d_out // 2),
        )
        self.att_pooling_1 = Att_pooling(d_out, d_out // 2)

        self.mlp2 = nn.Sequential(
            nn.Conv2d(d_out // 2, d_out // 2, kernel_size=(1, 1)),
            nn.BatchNorm2d(d_out // 2),
        )
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  #
        f_xyz = f_xyz.permute((0, 3, 1, 2))
        f_xyz = self.mlp1(f_xyz)
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)),
                                             neigh_idx)  #
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat)

        f_xyz = self.mlp2(f_xyz)
        f_neighbours = self.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)),
                                             neigh_idx)
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)   #
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1],
                                           1)
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = torch.sqrt(
            torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz],
                                     dim=-1)
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):

        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)  #
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[
            2]))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1],
                                    d)
        return features


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = nn.Sequential(
            nn.Conv2d(d_in, d_out, kernel_size=(1, 1)),
            nn.BatchNorm2d(d_out),
        )

    def forward(self, feature_set):
        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg


class Att_pooling_2(nn.Module):
    def __init__(self, d_in, d_out):
        super(Att_pooling_2, self).__init__()
        self.fc = nn.Conv1d(d_in, d_out, 1, bias=False)

        self.softmax = nn.Softmax(dim =-1)

    def forward(self, feature_set):
        att_activation = self.fc(feature_set)

        att_scores = self.softmax(att_activation)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=2, keepdim=True)

        return f_agg 
