# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2D旋转位置编码（2D Rotary Position Embeddings, RoPE）的实现
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 本模块提供了2D旋转位置编码的清晰实现，将原始的RoPE概念扩展到处理2D空间位置。
#
# 核心思想：
#   - 原始RoPE用于1D序列（如文本），通过旋转特征来编码位置信息
#   - 2D RoPE将此思想扩展到图像等2D数据，分别处理垂直和水平方向
#   - 相比传统的绝对/相对位置编码，RoPE能更好地保持距离关系

# 灵感来源：
#         https://github.com/meta-llama/codellama/blob/main/llama/model.py
#         https://github.com/naver-ai/rope-vit


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class PositionGetter:
    """生成并缓存2D网格中patch的空间位置坐标。
    
    这个类高效地管理2D网格中patch的空间坐标生成，通过缓存结果来避免重复计算。
    
    核心功能：
        - 为给定的高度和宽度生成笛卡尔坐标网格
        - 缓存已生成的位置，避免重复计算
        - 自动扩展到batch维度
    
    Attributes:
        position_cache: 字典，存储不同网格尺寸的预计算位置张量
                       键: (height, width) 元组
                       值: 对应的位置坐标张量
    """

    def __init__(self):
        """初始化位置生成器，创建空缓存。"""
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """为一批patch生成空间位置坐标。
        
        工作流程：
            1. 检查缓存中是否存在该尺寸的位置
            2. 如果不存在，生成并缓存新的位置坐标
            3. 返回扩展到batch_size的位置张量
        
        Args:
            batch_size: 批次中的样本数量
            height: patch网格的高度（patch行数）
            width: patch网格的宽度（patch列数）
            device: 目标设备（CPU/GPU）
        
        Returns:
            形状为 (batch_size, height*width, 2) 的张量
            包含每个位置的 (y, x) 坐标，为每个batch重复
            
        示例：
            height=2, width=3 会生成:
            [[0,0], [0,1], [0,2],
             [1,0], [1,1], [1,2]]
            然后扩展到batch_size
        """
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤1: 检查缓存，如果不存在则生成新的位置坐标
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if (height, width) not in self.position_cache:
            # 生成y坐标: [0, 1, 2, ..., height-1]
            y_coords = torch.arange(height, device=device)
            # 生成x坐标: [0, 1, 2, ..., width-1]
            x_coords = torch.arange(width, device=device)
            
            # 生成笛卡尔积：所有(y,x)坐标对
            # 例如: height=2, width=3 → 
            #   tensor([[0,0], [0,1], [0,2], [1,0], [1,1], [1,2]])
            positions = torch.cartesian_prod(y_coords, x_coords)
            
            # 缓存生成的位置
            self.position_cache[height, width] = positions

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤2: 从缓存获取位置并扩展到batch维度
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        cached_positions = self.position_cache[height, width]
        
        # 重塑并扩展：
        #   (H*W, 2) → (1, H*W, 2) → (B, H*W, 2)
        # clone()确保每个batch可以独立修改（虽然通常不会）
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """2D旋转位置编码实现。
    
    该模块根据token的2D空间位置应用旋转位置编码。它分别处理垂直和水平
    维度的位置相关特征旋转。
    
    核心原理：
        1. RoPE通过旋转特征向量来编码位置信息
        2. 对于2D情况，将特征分成两半：
           - 前半部分用于编码垂直位置（y坐标）
           - 后半部分用于编码水平位置（x坐标）
        3. 每个维度使用不同频率的正弦/余弦函数进行旋转
    
    数学原理：
        对于位置p和特征维度d，旋转角度 θ = p / (base_freq^(2i/d))
        应用旋转矩阵：
        [cos(θ)  -sin(θ)] [x1]
        [sin(θ)   cos(θ)] [x2]
    
    Args:
        frequency: 位置编码的基础频率，控制位置编码的周期。默认: 100.0
                  较大的值 → 更长的周期 → 更适合长序列
        scaling_factor: 频率计算的缩放因子。默认: 1.0
                       可用于调整不同分辨率的位置编码
    
    Attributes:
        base_frequency: 计算位置编码的基础频率
        scaling_factor: 用于缩放计算出的频率的因子
        frequency_cache: 缓存预计算的频率分量（cos和sin）
    """

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        """初始化2D RoPE模块。"""
        super().__init__()
        self.base_frequency = frequency        # 基础频率（默认100.0）
        self.scaling_factor = scaling_factor   # 缩放因子（默认1.0）
        # 缓存字典：存储不同配置下的cos和sin分量
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算旋转编码的频率分量（cos和sin）。
        
        这是RoPE的核心计算部分，生成位置相关的旋转角度。
        
        数学推导：
            1. 频率指数: exponents = [0, 2, 4, ..., dim-2] / dim
            2. 逆频率: inv_freq = 1 / (base_freq ^ exponents)
            3. 角度: angles = position × inv_freq
            4. 旋转分量: cos(angles), sin(angles)
        
        Args:
            dim: 特征维度（必须是偶数，因为成对旋转）
            seq_len: 最大序列长度（决定生成多少位置的编码）
            device: 计算目标设备
            dtype: 计算数据类型
        
        Returns:
            (cos_components, sin_components) 元组
            每个形状为 (seq_len, dim)
        """
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 缓存机制：避免重复计算相同配置的频率分量
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 步骤1: 计算频率带（frequency bands）
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 生成偶数索引: [0, 2, 4, ..., dim-2]
            # 例如 dim=8: [0, 2, 4, 6]
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            # 计算逆频率: 1 / (base_freq^exponents)
            # 不同维度使用不同频率，低维度用低频，高维度用高频
            inv_freq = 1.0 / (self.base_frequency**exponents)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 步骤2: 生成位置相关的频率
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 位置索引: [0, 1, 2, ..., seq_len-1]
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            
            # 计算所有位置和所有频率的外积
            # 形状: (seq_len, dim//2)
            # angles[p, i] = position[p] × inv_freq[i]
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 步骤3: 计算并缓存cos和sin分量
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            angles = angles.to(dtype)
            # 将angles复制一份拼接：(seq_len, dim//2) → (seq_len, dim)
            # 这是因为旋转是成对进行的
            angles = torch.cat((angles, angles), dim=-1)
            
            # 计算cos和sin分量
            cos_components = angles.cos().to(dtype)  # 形状: (seq_len, dim)
            sin_components = angles.sin().to(dtype)  # 形状: (seq_len, dim)
            
            # 缓存结果
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """通过分割和重组特征维度来执行特征旋转。
        
        这是RoPE的核心旋转操作，实现了复数乘法的实部和虚部。
        
        数学原理：
            将特征分成两半 [x1, x2]，旋转为 [-x2, x1]
            这对应于复数旋转中的虚部操作
            
        例如：
            输入: [a, b, c, d, e, f, g, h]  (dim=8)
            输出: [-e, -f, -g, -h, a, b, c, d]
        
        Args:
            x: 输入张量，任意形状 (..., dim)
        
        Returns:
            旋转后的特征张量，形状与输入相同
        """
        feature_dim = x.shape[-1]
        # 将特征维度分成两半
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        # 旋转：前半部分取负并移到后面，后半部分移到前面
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
        """沿一个维度应用1D旋转位置编码。
        
        这是应用RoPE的实际操作，将位置信息通过旋转注入到特征中。
        
        数学公式：
            output = tokens × cos(θ) + rotate(tokens) × sin(θ)
            其中 θ 由位置决定
        
        Args:
            tokens: 输入token特征，形状 (B, n_heads, n_tokens, dim)
            positions: 位置索引，形状 (B, n_tokens)，每个值表示该token的位置
            cos_comp: 余弦分量，形状 (max_pos, dim)
            sin_comp: 正弦分量，形状 (max_pos, dim)
        
        Returns:
            应用了旋转位置编码的token，形状与输入相同
        """
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤1: 根据位置索引提取对应的cos和sin值
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # F.embedding: 从cos_comp中查找positions对应的值
        # 输入: positions (B, n_tokens)
        # 输出: (B, n_tokens, dim)
        # 扩展维度: (B, n_tokens, dim) → (B, 1, n_tokens, dim)
        #           用于匹配tokens的n_heads维度
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤2: 应用旋转公式
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 公式: output = x*cos(θ) + rotate(x)*sin(θ)
        # 这等价于在复平面上旋转θ角度
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """对输入token应用2D旋转位置编码。
        
        这是主要的前向传播函数，将2D空间位置编码注入到token特征中。
        
        处理流程：
            1. 将特征维度分成两半：垂直特征 + 水平特征
            2. 使用y坐标对垂直特征应用RoPE
            3. 使用x坐标对水平特征应用RoPE
            4. 拼接处理后的特征
        
        Args:
            tokens: 输入张量，形状 (batch_size, n_heads, n_tokens, dim)
                   特征维度dim必须能被4整除（因为要分成2D的两个维度，每个维度又要成对旋转）
            positions: 位置张量，形状 (batch_size, n_tokens, 2)
                      包含每个token的(y, x)坐标
                      positions[..., 0] = y坐标（行）
                      positions[..., 1] = x坐标（列）
        
        Returns:
            应用了2D旋转位置编码的张量，形状与输入相同
        
        Raises:
            AssertionError: 如果输入维度无效或位置格式错误
            
        示例：
            对于一个14×14的patch网格：
            - tokens形状: (2, 8, 196, 768)  # 2样本, 8头, 196个patch, 768维
            - positions形状: (2, 196, 2)     # 每个patch的(y,x)坐标
            输出形状: (2, 8, 196, 768)
        """
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 验证输入
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        assert tokens.size(-1) % 2 == 0, "特征维度必须是偶数（用于成对旋转）"
        assert positions.ndim == 3 and positions.shape[-1] == 2, \
            "位置必须是形状为 (batch_size, n_tokens, 2) 的张量"

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 计算每个空间方向的特征维度
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 将总维度平分给垂直和水平方向
        # 例如: 总维度768 → 每个方向384维
        feature_dim = tokens.size(-1) // 2

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 获取频率分量（cos和sin）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 找到最大位置值，确定需要生成多少位置的编码
        max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, max_position, tokens.device, tokens.dtype
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 将特征分割为垂直和水平处理
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # chunk(2, dim=-1): 在最后一个维度上分成2块
        # 例如: (B, H, N, 768) → (B, H, N, 384) + (B, H, N, 384)
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 分别对每个维度应用RoPE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 垂直特征：使用y坐标 (positions[..., 0])
        vertical_features = self._apply_1d_rope(
            vertical_features, positions[..., 0], cos_comp, sin_comp
        )
        
        # 水平特征：使用x坐标 (positions[..., 1])
        horizontal_features = self._apply_1d_rope(
            horizontal_features, positions[..., 1], cos_comp, sin_comp
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 拼接处理后的特征
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 将垂直和水平特征重新组合
        # (B, H, N, 384) + (B, H, N, 384) → (B, H, N, 768)
        return torch.cat((vertical_features, horizontal_features), dim=-1)