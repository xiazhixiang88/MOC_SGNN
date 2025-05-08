import torch
import torch.nn as nn
import math
from torch_geometric.nn import GATConv, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention
import numpy as np
import scipy.stats as stats

class GAT_KDE(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 n_layers,
                 pooling_list,
                 activation=torch.nn.ReLU(),
                 feat_drop=0.2,
                 attn_drop=0.2,
                 n_heads=4,
                 kde_quantiles=np.linspace(0, 1, 20),
                 kde_grid_points=500):
        super(GAT_KDE, self).__init__()

        self.n_layers = n_layers
        self.activation = activation
        self.feat_drop = nn.Dropout(p=feat_drop)
        heads = [n_heads] * (n_layers)
        self.pooling_list = pooling_list

        # GAT层
        self.convs = torch.nn.ModuleList()
        for l in range(n_layers):
            in_channels = in_dim if l == 0 else hidden_dim * heads[l - 1]
            out_channels = hidden_dim
            heads_current = heads[l]
            self.convs.append(
                GATConv(in_channels, out_channels, heads=heads_current,
                        dropout=attn_drop, concat=True)
            )

        # 注意力池化的门控网络
        self.att_gate_nns = torch.nn.ModuleList()
        for layer in range(n_layers + 1):
            if layer == 0:
                gate_nn = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, 1),
                    torch.nn.Sigmoid()
                )
            else:
                gate_nn = torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim * heads[layer - 1], 1),
                    torch.nn.Sigmoid()
                )
            self.att_gate_nns.append(gate_nn)

        # 预测层维度适配
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(n_layers + 1):
            linear_in = in_dim if layer == 0 else hidden_dim * heads[layer - 1]
            self.linears_prediction.append(torch.nn.Linear(linear_in, out_dim))

        self.h_weights = torch.nn.Parameter(torch.ones(n_layers + 1) / (n_layers + 1))

        # 生存分析参数
        self.h0 = torch.nn.Parameter(torch.randn(1))
        self.beta = torch.nn.Parameter(torch.randn(out_dim))

        # 存储池化操作的字典（使用PyG池化函数）
        self.pooling_functions = {
            'att': lambda x, i, batch: GlobalAttention(self.att_gate_nns[i])(x, batch),
            'mean': global_mean_pool,
            'max': global_max_pool,
            'min': lambda x, _, batch: torch.cat([x[batch == i].min(dim=0)[0].unsqueeze(0) for i in range(batch.max().item() + 1)]),
            'skewness': lambda x, _, batch: torch.tensor(stats.skew(x.detach().cpu().numpy(), axis=0), dtype=torch.float32).unsqueeze(0).to(x.device),
            'kurtosis': lambda x, _, batch: torch.tensor(stats.kurtosis(x.detach().cpu().numpy(), axis=0), dtype=torch.float32).unsqueeze(0).to(x.device)
        }

        self.pooling_weights = nn.Parameter(torch.ones(len(pooling_list)) / len(pooling_list))

        # KDE参数
        self.kde_quantiles = torch.tensor(kde_quantiles)
        self.kde_grid_points = kde_grid_points
        self.num_quantiles = len(kde_quantiles)

        # 分位数投影层（每个图生成固定维度）
        self.kde_projectors = nn.ModuleList()
        self.kde_projectors.append(nn.Linear(in_dim * self.num_quantiles, out_dim))
        for _ in range(n_layers):
            # in_channels = in_dim if _ == 0 else hidden_dim * heads[_ - 1]
            self.kde_projectors.append(nn.Linear(hidden_dim * n_heads * self.num_quantiles, out_dim))

    def _kde_per_graph(self, x, quantiles, grid_points):
        """
        单图KDE分位数计算（可导实现）
        输入:
            x: [num_nodes, hidden_dim] 当前图的节点特征
            quantiles: [Q] 分位数列表（如[0.25, 0.5, 0.75]）
            grid_points: 网格点数
        返回:
            quantile_features: [hidden_dim * Q] 分位数特征
        """
        if x.size(0) < 2:  # 处理单节点图
            return torch.zeros(x.size(1) * len(quantiles), device=x.device)

        # 动态网格生成（保留梯度）
        min_val = x.min(dim=0)[0] - 1e-6
        max_val = x.max(dim=0)[0] + 1e-6
        grid = torch.linspace(0, 1, grid_points, device=x.device).unsqueeze(1) * (max_val - min_val) + min_val  # [grid_points, hidden_dim]

        # 带宽计算（Silverman规则）
        std = x.std(dim=0, unbiased=False) + 1e-8 / 3  # [hidden_dim]
        h = 1.06 * std * (x.size(0) ** -0.2)  # [hidden_dim]

        # 核密度估计（向量化加速）
        u = (grid.unsqueeze(1) - x.unsqueeze(0)) / h.unsqueeze(0)  # [grid_points, num_nodes, hidden_dim]
        kernel = torch.exp(-0.5 * (u ** 2)) / (h * math.sqrt(2 * math.pi)).unsqueeze(0)  # [grid_points, num_nodes, hidden_dim]
        density = kernel.mean(dim=1)  # [grid_points, hidden_dim]

        # 计算累积分布函数（CDF）
        cdf = torch.cumsum(density, dim=0)  # [grid_points, hidden_dim]
        cdf = cdf / cdf[-1:].clamp(min=1e-8)  # 归一化 [grid_points, hidden_dim]

        # 可导分位数估计（软权重）
        distance = (cdf.unsqueeze(2) - quantiles.to(x.device)).abs()  # [grid_points, hidden_dim, Q]
        weights = torch.sigmoid(- 100 * distance)  # [grid_points, hidden_dim, Q]
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)  # [grid_points, hidden_dim, Q]

        # 计算加权分位数值
        quantile_vals = (grid.unsqueeze(-1) * weights).sum(dim=0)  # [hidden_dim, Q]
        return quantile_vals.flatten()  # [hidden_dim * Q]

    def forward(self, data):
        # 数据准备
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if batch is None:
            # 假设输入只有一个图
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 初始化各层输出
        h_list = []
        kde_list = []
        current_x = x

        num_graphs = batch.max().item() + 1

        for i in range(self.n_layers):
            # 第i层GAT
            current_x = self.feat_drop(current_x) if i != 0 else current_x

            # 原有池化路径
            pooled_features = []
            for pooling in self.pooling_list:
                if pooling == 'att':
                    pooled = self.pooling_functions[pooling](current_x, i, batch)
                else:
                    pooled = self.pooling_functions[pooling](current_x, batch)
                pooled_features.append(pooled)
            pooled_features = torch.stack(pooled_features, dim=0)  # [num_pooling, num_graphs, hidden_dim]

            # 加权平均池化结果
            weighted_pooled = torch.sum(pooled_features * self.pooling_weights.unsqueeze(1).unsqueeze(2), dim=0)

            # 预测层
            h = self.linears_prediction[i](weighted_pooled)
            h_list.append(h)

            # KDE分位数池化路径
            kde_features = []
            for graph_idx in range(num_graphs):
                graph_x = current_x[batch == graph_idx]
                kde_feat = self._kde_per_graph(graph_x, self.kde_quantiles, self.kde_grid_points).float()
                kde_features.append(kde_feat)
            kde_features = torch.stack(kde_features)
            projected = self.kde_projectors[i](kde_features)
            kde_list.append(projected)

            current_x = self.convs[i](current_x, edge_index)
            if self.activation is not None and i != self.n_layers - 1:
                current_x = self.activation(current_x)

        # 最后一层处理
        pooled_features = []
        for pooling in self.pooling_list:
            if pooling == 'att':
                pooled = self.pooling_functions[pooling](current_x, -1, batch)
            else:
                pooled = self.pooling_functions[pooling](current_x, batch)
            pooled_features.append(pooled)
        pooled_features = torch.stack(pooled_features, dim=0)
        weighted_pooled = torch.sum(pooled_features * self.pooling_weights.unsqueeze(1).unsqueeze(2), dim=0)
        h_final = self.linears_prediction[-1](weighted_pooled)
        h_list.append(h_final)

        # KDE最后一层处理
        kde_features = []
        for graph_idx in range(num_graphs):
            graph_x = current_x[batch == graph_idx]
            kde_feat = self._kde_per_graph(graph_x, self.kde_quantiles, self.kde_grid_points).float()
            kde_features.append(kde_feat)
        kde_features = torch.stack(kde_features)
        projected = self.kde_projectors[-1](kde_features)
        kde_list.append(projected)

        # 最终特征融合（均值聚合）
        main_out = torch.stack(h_list).mean(dim=0)
        kde_out = torch.stack(kde_list).mean(dim=0)

        # 风险预测
        risk_score = torch.matmul(main_out + kde_out, self.beta) + self.h0
        return risk_score

    