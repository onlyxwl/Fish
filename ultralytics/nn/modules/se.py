import torch
import torch.nn as nn


# class SE(nn.Module):
#     """Squeeze-and-Excitation注意力模块 (CVPR 2018)"""
#
#     def __init__(self, c1, reduction_ratio=16):
#         """
#         Args:
#             c1: 输入通道数
#             reduction_ratio: 压缩比例 (默认为16)
#         """
#         super().__init__()
#         reduced_channels = max(1, c1 // reduction_ratio)
#
#         # Squeeze操作: 全局平均池化
#         self.gap = nn.AdaptiveAvgPool2d(1)
#
#         # Excitation操作: 两个全连接层
#         self.fc = nn.Sequential(
#             nn.Linear(c1, reduced_channels, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(reduced_channels, c1, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         """前向传播"""
#         b, c, _, _ = x.shape
#
#         # 1. Squeeze: 全局平均池化
#         y = self.gap(x).view(b, c)
#
#         # 2. Excitation: 通道注意力权重
#         y = self.fc(y).view(b, c, 1, 1)
#
#         # 3. Scale: 应用注意力权重
#         return x * y.expand_as(x)

# SE变体
# class SE(nn.Module):
#     """针对水下鱼识别优化的注意力模块"""
#
#     def __init__(self, c1, reduction_ratio=8, depthwise=True):
#         """
#         Args:
#             c1: 输入通道数
#             reduction_ratio: 压缩比例 (默认为8，比标准SE更小)
#             depthwise: 是否使用深度可分离卷积减少计算量
#         """
#         super().__init__()
#         reduced_channels = max(1, c1 // reduction_ratio)
#
#         # 使用深度可分离卷积减少参数
#         if depthwise:
#             self.conv1 = nn.Sequential(
#                 nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1, bias=False),
#                 nn.BatchNorm2d(c1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(c1, reduced_channels, kernel_size=1, bias=False)
#             )
#         else:
#             self.conv1 = nn.Conv2d(c1, reduced_channels, kernel_size=1, bias=False)
#
#         self.conv2 = nn.Conv2d(reduced_channels, c1, kernel_size=1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#         # 添加位置编码增强空间感知
#         self.position_encoding = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         # 生成位置编码
#         y_coords = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(b, 1, h, w).to(x.device)
#         x_coords = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(b, 1, h, w).to(x.device)
#         pos_enc = self.position_encoding(torch.cat([x_coords, y_coords], dim=1))
#
#         # 通道注意力
#         y = self.conv1(x)
#         y = torch.relu(y)
#         y = self.conv2(y)
#         channel_weights = self.sigmoid(y)
#
#         # 结合位置信息
#         spatial_weights = self.sigmoid(pos_enc)
#
#         # 组合通道和位置注意力
#         combined_weights = channel_weights * spatial_weights
#
#         return x * combined_weights.expand_as(x)

# class SE(nn.Module):
#     """轻量级注意力模块，适合水桶鱼识别场景"""
#
#     def __init__(self, c1, reduction_ratio=4):
#         """
#         Args:
#             c1: 输入通道数
#             reduction_ratio: 压缩比例 (默认为4)
#         """
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         # 共享的全连接层
#         self.fc = nn.Sequential(
#             nn.Linear(c1, max(1, c1 // reduction_ratio), bias=False),
#             nn.ReLU(inplace=True)
#         )
#
#         # 分离的通道权重预测
#         self.avg_fc = nn.Linear(max(1, c1 // reduction_ratio), c1, bias=False)
#         self.max_fc = nn.Linear(max(1, c1 // reduction_ratio), c1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         b, c, _, _ = x.shape
#
#         # 平均池化分支
#         avg_out = self.avg_pool(x).view(b, c)
#         avg_out = self.fc(avg_out)
#         avg_out = self.avg_fc(avg_out)
#
#         # 最大池化分支
#         max_out = self.max_pool(x).view(b, c)
#         max_out = self.fc(max_out)
#         max_out = self.max_fc(max_out)
#
#         # 组合注意力权重
#         channel_weights = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
#
#         return x * channel_weights.expand_as(x)

# class SE(nn.Module):
#     """高效多尺度注意力（ICASSP 2023）"""
#
#     def __init__(self, c1, reduction_ratio=4):
#         super().__init__()
#         reduced_channels = max(1, c1 // reduction_ratio)
#
#         # 多尺度特征提取
#         self.conv1 = nn.Conv2d(c1, reduced_channels, 1)
#         self.conv3 = nn.Conv2d(c1, reduced_channels, 3, padding=1)
#         self.conv5 = nn.Conv2d(c1, reduced_channels, 5, padding=2)
#
#         # 通道注意力
#         self.channel_att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(reduced_channels * 3, c1, 1),
#             nn.Sigmoid()
#         )
#
#         # 空间注意力
#         self.spatial_att = nn.Sequential(
#             nn.Conv2d(3, 1, 7, padding=3),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         # 多尺度特征
#         f1 = self.conv1(x)
#         f3 = self.conv3(x)
#         f5 = self.conv5(x)
#         multi_scale = torch.cat([f1, f3, f5], dim=1)
#
#         # 通道注意力
#         channel_weights = self.channel_att(multi_scale)
#
#         # 空间注意力
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         spatial_weights = self.spatial_att(torch.cat([avg_out, max_out, channel_weights], dim=1))
#
#         return x * channel_weights * spatial_weights

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class SE(nn.Module):
#     """鱼群行为感知模块（针对鱼识别优化）"""
#
#     def __init__(self, c1, reduction=4, num_vectors=8):
#         super().__init__()
#         self.c1 = c1
#         self.num_vectors = num_vectors
#         reduced_c = max(1, c1 // reduction)
#
#         # 运动矢量生成
#         self.motion_vectors = nn.Conv2d(c1, 2 * num_vectors, 3, padding=1)
#
#         # 运动特征提取
#         self.motion_feat = nn.Sequential(
#             nn.Conv2d(c1, reduced_c, 1),
#             nn.ReLU(inplace=True))
#
#         # 注意力生成
#         self.attn_gen = nn.Sequential(
#             nn.Conv2d(reduced_c * num_vectors, c1, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         # 生成运动矢量
#         vectors = self.motion_vectors(x).view(b, 2, self.num_vectors, h, w)
#
#         # 采样运动特征
#         motion_feats = []
#         for i in range(self.num_vectors):
#             # 采样偏移位置
#             grid = torch.stack(torch.meshgrid(
#                 torch.linspace(-1, 1, h),
#                 torch.linspace(-1, 1, w)
#             ), dim=-1).unsqueeze(0).to(x.device)
#
#             offset = vectors[:, :, i].permute(0, 2, 3, 1)
#             warped_grid = grid + offset * 0.1  # 小幅度偏移
#
#             # 采样特征
#             warped_feat = F.grid_sample(
#                 self.motion_feat(x),
#                 warped_grid,
#                 mode='bilinear',
#                 align_corners=False
#             )
#             motion_feats.append(warped_feat)
#
#         # 融合运动特征
#         motion_feats = torch.cat(motion_feats, dim=1)
#         attn_weights = self.attn_gen(motion_feats)
#
#         return x * attn_weights

# import torch
# import torch.nn as nn
# from einops import rearrange
#
#
# class SE(nn.Module):
#     """选择性状态空间模型（VMamba, ArXiv 2023）"""
#
#     def __init__(self, c1, d_state=16, d_conv=3, expand=2):
#         super().__init__()
#         self.d_model = c1
#         self.d_state = d_state
#         self.d_conv = d_conv
#         self.expand = expand
#         self.d_inner = int(self.expand * self.d_model)
#
#         # 投影层
#         self.in_proj = nn.Linear(c1, self.d_inner * 2)
#
#         # 卷积层
#         self.conv2d = nn.Conv2d(
#             in_channels=self.d_inner,
#             out_channels=self.d_inner,
#             groups=self.d_inner,
#             kernel_size=d_conv,
#             padding=(d_conv - 1) // 2,
#         )
#
#         # 状态空间参数
#         self.x_proj = nn.Linear(self.d_inner, d_state + d_state)
#         self.dt_proj = nn.Linear(self.d_inner, d_state)
#
#         # 输出层
#         self.out_proj = nn.Linear(self.d_inner, c1)
#
#     def forward(self, x):
#         # 输入投影
#         x = rearrange(x, 'b c h w -> b h w c')
#         xz = self.in_proj(x)
#         x, z = xz.chunk(2, dim=-1)
#
#         # 2D卷积
#         x = rearrange(x, 'b h w c -> b c h w')
#         x = self.conv2d(x)
#         x = rearrange(x, 'b c h w -> b h w c')
#
#         # 状态空间模型
#         B, H, W, D = x.shape
#         x_dbl = self.x_proj(x)
#         dt, A = torch.split(x_dbl, [self.d_state, self.d_state], dim=-1)
#         dt = self.dt_proj(dt)
#
#         # 选择性扫描
#         A = -torch.exp(A)
#         dt = torch.exp(dt)
#         y = torch.zeros_like(x)
#
#         # 简化的扫描实现
#         for h in range(H):
#             for w in range(W):
#                 state = torch.zeros(B, self.d_state).to(x.device)
#                 for i in range(H):
#                     for j in range(W):
#                         idx = min(i, j)
#                         state = state * dt[:, i, j] + x[:, i, j] * A[:, i, j]
#                         y[:, i, j] = state[:, idx]
#
#         # 门控和输出
#         y = y * F.silu(z)
#         y = self.out_proj(y)
#         y = rearrange(y, 'b h w c -> b c h w')
#
#         return y
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
#
#
# class SE(nn.Module):
#     """优化版选择性状态空间模型"""
#
#     def __init__(self, c1, d_state=16, d_conv=3, expand=2):
#         super().__init__()
#         self.c1 = c1
#         self.d_state = d_state
#         self.d_conv = d_conv
#         self.expand = expand
#         self.d_inner = int(expand * c1)
#
#         # 投影层
#         self.in_proj = nn.Linear(c1, self.d_inner * 2)
#
#         # 深度可分离卷积
#         self.conv = nn.Sequential(
#             nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv,
#                       padding=d_conv // 2, groups=self.d_inner),
#             nn.BatchNorm2d(self.d_inner),
#             nn.SiLU()
#         )
#
#         # 状态空间参数
#         self.x_proj = nn.Linear(self.d_inner, d_state + d_state)
#         self.dt_proj = nn.Sequential(
#             nn.Linear(self.d_inner, d_state),
#             nn.Softplus()  # 确保dt为正
#         )
#
#         # 输出层
#         self.out_proj = nn.Linear(self.d_inner, c1)
#
#     def forward(self, x):
#         # 保存原始输入
#         identity = x
#
#         # 输入投影 [B, C, H, W] -> [B, H, W, D*2]
#         x = rearrange(x, 'b c h w -> b h w c')
#         xz = self.in_proj(x)
#         x, z = xz.chunk(2, dim=-1)  # [B, H, W, D], [B, H, W, D]
#
#         # 卷积处理 [B, H, W, D] -> [B, D, H, W] -> 卷积 -> [B, H, W, D]
#         x = rearrange(x, 'b h w c -> b c h w')
#         x = self.conv(x)
#         x = rearrange(x, 'b c h w -> b h w c')
#
#         # 状态空间参数
#         x_dbl = self.x_proj(x)  # [B, H, W, 2*D_state]
#         A, B = torch.split(x_dbl, [self.d_state, self.d_state], dim=-1)
#         dt = self.dt_proj(x)  # [B, H, W, D_state]
#
#         # 状态空间扫描 (优化实现)
#         y = torch.zeros(B, H, W, self.d_state, device=x.device)
#         state = torch.zeros(B, self.d_state, device=x.device)
#
#         # 按行扫描
#         for i in range(H):
#             for j in range(W):
#                 # 更新状态
#                 state = state * torch.exp(-dt[:, i, j] * A[:, i, j]) + B[:, i, j]
#                 y[:, i, j] = state
#
#         # 门控和输出
#         y = y[..., :self.d_inner]  # 取前d_inner维
#         y = y * F.silu(z)
#         y = self.out_proj(y)
#
#         # 输出 [B, H, W, C] -> [B, C, H, W]
#         y = rearrange(y, 'b h w c -> b c h w')
#
#         # 残差连接
#         return identity + y
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SE(nn.Module):
    """优化版选择性状态空间模型"""

    def __init__(self, c1, d_state=16, d_conv=3, expand=2):
        """
        Args:
            c1: 输入通道数
            d_state: 状态维度大小
            d_conv: 卷积核大小
            expand: 扩展比例
        """
        super().__init__()
        self.c1 = c1
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * c1)

        # 投影层
        self.in_proj = nn.Linear(c1, self.d_inner * 2)

        # 深度可分离卷积
        self.conv = nn.Sequential(
            nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv,
                      padding=d_conv // 2, groups=self.d_inner),
            nn.BatchNorm2d(self.d_inner),
            nn.SiLU()
        )

        # 状态空间参数
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state)
        self.dt_proj = nn.Sequential(
            nn.Linear(self.d_inner, d_state),
            nn.Softplus()  # 确保dt为正
        )

        # 输出层
        self.out_proj = nn.Linear(self.d_inner, c1)

    def forward(self, x):
        # 保存原始输入用于残差连接
        identity = x

        # 获取输入特征图的尺寸
        B, C, H, W = x.shape  # [批量大小, 通道数, 高度, 宽度]

        # 输入投影 [B, C, H, W] -> [B, H, W, C]
        x = rearrange(x, 'b c h w -> b h w c')
        xz = self.in_proj(x)  # [B, H, W, d_inner * 2]
        x, z = xz.chunk(2, dim=-1)  # 各为 [B, H, W, d_inner]

        # 卷积处理 [B, H, W, d_inner] -> [B, d_inner, H, W] -> 卷积 -> [B, H, W, d_inner]
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.conv(x)
        x = rearrange(x, 'b c h w -> b h w c')

        # 状态空间参数计算
        x_dbl = self.x_proj(x)  # [B, H, W, d_state * 2]
        A, B = torch.split(x_dbl, [self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(x)  # [B, H, W, d_state]

        # 状态空间扫描 (优化实现)
        y = torch.zeros(B, H, W, self.d_state, device=x.device)
        state = torch.zeros(B, self.d_state, device=x.device)

        # 按行扫描
        for i in range(H):
            # 按列扫描
            for j in range(W):
                # 更新状态
                # 状态更新公式: state = state * exp(-dt * A) + B
                state = state * torch.exp(-dt[:, i, j] * A[:, i, j]) + B[:, i, j]
                y[:, i, j] = state

        # 门控和输出
        y = y[..., :self.d_inner]  # 取前d_inner维
        y = y * F.silu(z)  # 应用门控
        y = self.out_proj(y)  # [B, H, W, c1]

        # 输出 [B, H, W, C] -> [B, C, H, W]
        y = rearrange(y, 'b h w c -> b c h w')

        # 残差连接
        return identity + y