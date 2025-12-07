# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.


    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. 在切换之前，每种注意力类型下需要划分多少个区块 If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """
    # 实际上就是dinov2
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        # 初始化位姿token
        # 位置编码
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        # 帧内注意力
        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        #全局注意力
        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        # 直接增加的可训练的参数层
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)
    """
    Todo hfx:
    处理图像patch
    """
    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }
            # 采用dinov2
            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std # 图像归一化

        # Reshape to [B*S, C, H, W] for patch embedding
        # 现在图片转token都是直接用dinov2了吗
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # ============================================================
        # 扩展特殊token以匹配批次大小和序列长度
        # ============================================================
        
        # 扩展相机token: (1, 2, X_cam, C) -> (B*S, X_cam, C)
        # 作用: 编码相机参数（如内参、外参、视角等）
        # - 第一帧使用专用的相机token（索引0）
        # - 后续帧共享另一个相机token（索引1）
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        
        # 扩展注册token: (1, 2, X_reg, C) -> (B*S, X_reg, C)
        # 作用: 类似于ViT的[CLS] token，用于全局信息聚合
        # - 提供可学习的"寄存器"空间，帮助模型存储全局特征
        # - 缓解注意力机制中的"token过度利用"问题
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # ============================================================
        # 拼接所有token形成完整的输入序列
        # ============================================================
        
        # 在特征维度（dim=1）上拼接：
        # [camera_token | register_token | patch_tokens]
        #  相机参数token | 注册token        | 图像patch token
        #  (X_cam个)    | (X_reg个)       | (H*W个)
        # 
        # 最终形状: (B*S, X_cam + X_reg + H*W, C)
        # 其中: 
        #   B*S: 总帧数（批次大小 × 每个样本的帧数）
        #   X_cam + X_reg + H*W: 每帧的总token数
        #   C: 特征维度（通道数）
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num): #分块划分的意义是什么？

            # 进行交替注意力机制
            # 其输入
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C] 把帧内和全局合并 为一个一维token
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx
    """
    帧间自注意力机制
    """
    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)
            # 含义：将 B 个样本的 S 个frames 展平成 B*S 个独立单元

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)
            # 含义：每个样本内的所有 frames 的所有 patches 连接成一个长序列

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates

# 用于区分第一个帧，因为VGGT是基于第一帧坐标系重建的。
# 不过token_tensor又是什么呢？
def slice_expand_and_flatten(token_tensor, B, S):
    """
    处理用于多帧处理的特殊token，输入形状为 (1, 2, X, C)：
    1) 使用第一个位置（索引0）的token仅用于第一帧
    2) 使用第二个位置（索引1）的token用于所有剩余帧（S-1帧）
    3) 将两者都扩展以匹配批次大小B
    4) 拼接形成 (B, S, X, C)，其中每个序列有1个首帧token，后跟(S-1)个其他帧token
    5) 展平为 (B*S, X, C) 以便处理
    
    参数:
        token_tensor: 输入tensor，形状 (1, 2, X, C)
        B: 批次大小 (Batch size)
        S: 序列长度/帧数 (Sequence length)
    
    返回:
        torch.Tensor: 处理后的token，形状 (B*S, X, C)
    """

    # 步骤1: 切片出"查询"token（第一个位置） => 形状 (1, 1, X, C)
    # 这些token专门用于序列的第一帧
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # 扩展后形状: (B, 1, X, C) - B个样本，每个样本1帧
    
    # 步骤2: 切片出"其他"token（第二个位置） => 形状 (1, 1, X, C)
    # 注意：token_tensor[:,1:,...] 实际上只取了索引1，因为输入维度1只有2个元素
    # 这些token将被用于所有后续帧
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # 扩展后形状: (B, S-1, X, C) - B个样本，每个样本S-1帧
    
    # 步骤3: 在时间维度（dim=1）上拼接 => 形状 (B, S, X, C)
    # 结果: 每个样本的序列为 [query_token, other_token, other_token, ..., other_token]
    #                          第1帧          第2帧         第3帧              第S帧
    combined = torch.cat([query, others], dim=1)
    
    # 步骤4: 将批次和时间维度展平 => 形状 (B*S, X, C)
    # 这样可以将所有帧作为独立样本进行批处理
    # 例如: (2, 3, X, C) -> (6, X, C)
    combined = combined.view(B * S, *combined.shape[2:])
    
    return combined