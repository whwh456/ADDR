import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import cKDTree
# import os
# import sys
# sys.path.append("../visualization")
# from SOR import SORDefense
 
from sklearn.neighbors import KDTree

class BGFAD(nn.Module):
    """统计异常值移除作为防御方法。
    """

    def __init__(self, k=3, alpha=1.1, sor_batch=None):#5
        """

        Args:
            k (int, optional): k最近邻。默认为2。
            alpha (float, optional): \miu + \alpha * std。默认为1.1。
            sor_batch (int, optional): SOR批处理大小。默认为None。
        """
        super(BGFAD, self).__init__()

        self.k = k
        self.alpha = alpha
        self.sor_batch = sor_batch

    def outlier_removal(self, x):
        """移除较大的k最近邻距离的点。

        Args:
            x (torch.FloatTensor): 批量输入点云，[B, K, 3]

        Returns:
            torch.FloatTensor: 移除异常值后的点云，[B, N, 3]
        """
        pc = x.clone().detach().double()
        B, K = pc.shape[:2]
        pc = pc.transpose(2, 1)  # [B, 3, K]
        inner = -2. * torch.matmul(pc.transpose(2, 1), pc)  # [B, K, K]
        xx = torch.sum(pc ** 2, dim=1, keepdim=True)  # [B, 1, K]
        dist = xx + inner + xx.transpose(2, 1)  # [B, K, K]
        assert dist.min().item() >= -1e-6
        # 最小值是自身，所以我们取前（k + 1）个
        neg_value, _ = (-dist).topk(k=self.k + 1, dim=-1)
        # [B, K, k + 1]
        value = -(neg_value[..., 1:])  # [B, K, k]
        value = torch.mean(value, dim=-1)  # [B, K]
        mean = torch.mean(value, dim=-1)  # [B]
        std = torch.std(value, dim=-1)  # [B]

        threshold = mean + self.alpha * std #[B]
        bool_mask = (value <= threshold[:, None])  # [B, K]

        min_threshold = mean -  0.15 * std  # Shape [B]
        max_threshold = mean + self.alpha * std  # Shape [B]

        # Find points that are neither outliers nor too close to their neighbors
        bool_mask = (value > min_threshold[:, None]) & (value < max_threshold[:, None])

        sel_pc = [x[i][bool_mask[i]] for i in range(B)]
        return sel_pc  #返回一个list列表

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.FloatTensor): 输入点云，[B, N, 3]

        Returns:
            torch.FloatTensor: SOR处理后的点云，[B, N', 3]
        """
        if self.sor_batch is None:
            # 自适应找到kNN批处理大小
            flag = False
            self.sor_batch = x.shape[0]
            while not flag:
                if self.sor_batch < 1:
                    print('CUDA OUT OF MEMORY in kNN of repulsion loss')
                    exit(-1)
                try:
                    with torch.no_grad():
                        sor_x = self.batch_sor(x)
                    flag = True
                except RuntimeError:
                    torch.cuda.empty_cache()
                    self.sor_batch = self.sor_batch // 2
        else:
            with torch.no_grad():
                sor_x = self.batch_sor(x)  # [B, N, k]
        return sor_x #返回的是一个长度为B的list，每个元素为torch([n,3])，每个点云的n不同

    def batch_sor(self, x):
        """批量SOR处理。

        Args:
            x (torch.FloatTensor): 输入点云，[B, N, 3]

        Returns:
            list: 包含每个批次SOR处理后的点云列表，[[N1, 3], [N2, 3], ...]
        """
        sigma_s = 0.03  # 高斯核函数参数
        sigma_r = 10  # 高斯核函数参数
        k_neighboor = 3 #5
        
        out = []
        for i in range(0, x.shape[0], self.sor_batch):
            batch_x = x[i:i + self.sor_batch]
            sor_x = self.outlier_removal(batch_x)
            for x in sor_x:
                if len(x) > 10:
                    cur_x = self.cur_one(x,k_neighboor)
                # normals = self.compute_normal(cur_x)  # 计算法向量
                # fi_x = self.bilateral_filter(cur_x, normals, sigma_s, sigma_r)
                # fi_x = torch.from_numpy(fi_x).cuda()
                # cur_x = cur_x.cuda()
                    cur_X = torch.from_numpy(cur_x)
                if len(x) < 10:
                    cur_X = x
                out.append(cur_X)
            # out += sor_x
        return out
    

    #
    def calculate_curvature(self, point_cloud, k, tree):
        curvatures = []
        for point in point_cloud:
            _, indices = tree.query(point, k=k+1)  # +1 to exclude the point itself
            neighbors = point_cloud[indices[1:]]   # Get neighbor points excluding the point itself
            
            # Calculate covariance matrix
            covariance_matrix = np.cov(neighbors.T)

            # Eigen decomposition
            eigenvalues, _ = np.linalg.eigh(covariance_matrix)
            
            # Compute curvature
            a1 = eigenvalues[0]
            a2 = eigenvalues[1]
            a3 = eigenvalues[2]
            curvature = a1 / (a1 + a2 + a3 + 1e-6)

            curvatures.append(curvature)

        return np.array(curvatures)

    
    def cur_one(self, point_clouds, k, alpha=1.8):
        point_cloud = point_clouds.detach().cpu().numpy()
        # Function to calculate curvature for each point
        tree = cKDTree(point_cloud)

        # Calculate curvature for each point
        curvatures = self.calculate_curvature(point_cloud, k, tree)

        # Calculate mean and standard deviation of curvature for each point and its k neighbors
        mean_curvatures = []
        std_curvatures = []
        for i, point in enumerate(point_cloud):
            _, indices = tree.query(point, k=k+1)  # +1 to include the point itself
            neighbor_curvatures = curvatures[indices[1:]]  # Exclude the point itself
            mean_curvature = np.mean(neighbor_curvatures)
            std_curvature = np.std(neighbor_curvatures)
            mean_curvatures.append(mean_curvature)
            std_curvatures.append(std_curvature)

        mean_curvatures = np.array(mean_curvatures)
        std_curvatures = np.array(std_curvatures)

        # Filter points based on curvature criteria
        filtered_indices = []
        for i, curvature in enumerate(curvatures):
            if curvature >= mean_curvatures[i] - alpha * std_curvatures[i] and curvature <= mean_curvatures[i] + alpha * std_curvatures[i]:
                filtered_indices.append(i)

        # Return filtered point cloud
        filtered_point_cloud = point_cloud[filtered_indices]

        radius = 0.08
        radius2 = 0.25
        min_points = 3
        mim_points2 = 20
        filtered_indices = []
        tree = cKDTree(filtered_point_cloud)
        for i, point in enumerate(filtered_point_cloud):
            # Find neighbors within the specified radius
            neighbor_indices = tree.query_ball_point(point, radius)
            neighbor_indices2 = tree.query_ball_point(point, radius2)

            if len(neighbor_indices) >= min_points and len(neighbor_indices2)>mim_points2:
                filtered_indices.append(i)

        filtered_point_cloud = filtered_point_cloud[filtered_indices]
        return filtered_point_cloud.astype(np.float32)





