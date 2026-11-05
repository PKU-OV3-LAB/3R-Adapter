import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from huggingface_hub import PyTorchModelHubMixin
import os, numpy as np
import torchvision.utils as vutils

class Pi3(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
        ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        

        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features        # 1024
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #     Conf Decoder
        # ----------------------
        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)


    # def decode(self, hidden, N, H, W):
    #     BN, hw, _ = hidden.shape
    #     B = BN // N

    #     final_output = []
        
    #     hidden = hidden.reshape(B*N, hw, -1)

    #     register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

    #     # Concatenate special tokens with patch tokens
    #     hidden = torch.cat([register_token, hidden], dim=1)
    #     hw = hidden.shape[1]

    #     if self.pos_type.startswith('rope'):
    #         pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

    #     if self.patch_start_idx > 0:
    #         # do not use position embedding for special tokens (camera and register tokens)
    #         # so set pos to 0 for the special tokens
    #         pos = pos + 1
    #         pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
    #         pos = torch.cat([pos_special, pos], dim=1)
       
    #     for i in range(len(self.decoder)):
    #         blk = self.decoder[i]

    #         if i % 2 == 0:
    #             pos = pos.reshape(B*N, hw, -1)
    #             hidden = hidden.reshape(B*N, hw, -1)
    #         else:
    #             pos = pos.reshape(B, N*hw, -1)
    #             hidden = hidden.reshape(B, N*hw, -1)

    #         hidden = blk(hidden, xpos=pos)

    #         if i+1 in [len(self.decoder)-1, len(self.decoder)]:
    #             final_output.append(hidden.reshape(B*N, hw, -1))

    #     return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)
    
    # def forward(self, imgs):
    #     imgs = (imgs - self.image_mean) / self.image_std

    #     B, N, _, H, W = imgs.shape
    #     patch_h, patch_w = H // 14, W // 14
        
    #     # encode by dinov2
    #     imgs = imgs.reshape(B*N, _, H, W)
    #     hidden = self.encoder(imgs, is_training=True)

    #     if isinstance(hidden, dict):
    #         hidden = hidden["x_norm_patchtokens"]

    #     hidden, pos = self.decode(hidden, N, H, W)

    #     point_hidden = self.point_decoder(hidden, xpos=pos)
    #     conf_hidden = self.conf_decoder(hidden, xpos=pos)
    #     camera_hidden = self.camera_decoder(hidden, xpos=pos)

    #     with torch.amp.autocast(device_type='cuda', enabled=False):
    #         # local points
    #         point_hidden = point_hidden.float()
    #         ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
    #         xy, z = ret.split([2, 1], dim=-1)
    #         z = torch.exp(z)
    #         local_points = torch.cat([xy * z, z], dim=-1)

    #         # confidence
    #         conf_hidden = conf_hidden.float()
    #         conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)

    #         # camera
    #         camera_hidden = camera_hidden.float()
    #         camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

    #         # unproject local points using camera poses
    #         points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

    #     return dict(
    #         points=points,
    #         local_points=local_points,
    #         conf=conf,
    #         camera_poses=camera_poses,
    #     )
    
    def decode(self, hidden, N, H, W, save_vis_dir=None):
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
       
        # 寻找最后一层跨视角层的索引 (确保是奇数，表示 N 个视角进行了拼接交互)
        last_cross_view_idx = len(self.decoder) - 1
        if last_cross_view_idx % 2 == 0:
            last_cross_view_idx -= 1

        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)

            # ---------------------------------------------------------
            # [精准提取代码] 根据真实 Attention 逻辑，拦截并保存跨视角权重
            # ---------------------------------------------------------
            target_layers = [11, 23, 35] 
            
            if save_vis_dir is not None and i in target_layers:
                with torch.no_grad():
                    normed_hidden = blk.norm1(hidden) if hasattr(blk, 'norm1') else hidden
                    
                    qkv = blk.attn.qkv(normed_hidden)
                    B_cur, L_cur, C_cur = qkv.shape
                    
                    qkv = qkv.reshape(B_cur, L_cur, 3, blk.attn.num_heads, C_cur // 3 // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
                    
                    # 【核心显存优化】：强制转为 float16 计算，防止多层保存导致 OOM
                    q = (qkv[0] * blk.attn.scale).half() 
                    k = qkv[1].half()
                    
                    attn = q @ k.transpose(-2, -1)
                    attn = torch.softmax(attn, dim=-1)
                    
                    # 分别计算 Mean 和 Max
                    attn_map_mean = attn.mean(dim=1).cpu().numpy()    # (B, N*hw, N*hw)
                    attn_map_max = attn.max(dim=1)[0].cpu().numpy()   # (B, N*hw, N*hw)
                    
                    import os
                    import numpy as np
                    os.makedirs(save_vis_dir, exist_ok=True)
                    
                    # 落盘保存
                    save_path_mean = os.path.join(save_vis_dir, f'cross_attn_layer_{i}_mean.npy')
                    save_path_max = os.path.join(save_vis_dir, f'cross_attn_layer_{i}_max.npy')
                    np.save(save_path_mean, attn_map_mean)
                    np.save(save_path_max, attn_map_max)
                    print(f"✅ [Observation] 跨视角注意力 (Layer {i}) 已保存 | 包含 Mean 和 Max 矩阵")

            hidden = blk(hidden, xpos=pos)

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)
    
    # [修改] 增加 save_vis_dir 参数
    def forward(self, imgs, save_vis_dir=None):
        # ---------------------------------------------------------
        # [新增可视化代码] 在被标准化之前，将原图截取并保存下来
        # ---------------------------------------------------------
        # if save_vis_dir is not None:
        #     import os
        #     import torchvision.utils as vutils
        #     os.makedirs(save_vis_dir, exist_ok=True)
            
        #     B_orig, N_orig = imgs.shape[0], imgs.shape[1]
        #     patch_size = 14
            
            # for b_idx in range(B_orig):
            #     for n_idx in range(N_orig):
            #         # 1. 提取单张图像 (3, H, W) 并克隆，避免修改输入张量
            #         img_tensor = imgs[b_idx, n_idx].clone() 
                    
                    # # 2. 旋转 90 度 (k=1 为逆时针 90 度，k=3 为顺时针 90 度)
                    # # 假设图像需要竖屏展示，调整 dims=[1, 2] 表示在 H 和 W 维度旋转
                    # # img_rot = img_tensor
                    # img_rot = torch.rot90(img_tensor, k=3, dims=[1, 2])
                    # _, H_rot, W_rot = img_rot.shape
                    
                    # # 3. 绘制 Patch 网格线
                    # grid_color = torch.tensor([0.8, 0.8, 0.8]).view(3, 1, 1).to(img_rot.device)
                    
                    # # 画水平线
                    # for y in range(0, H_rot, patch_size):
                    #     # 画 1 像素宽的线，如果觉得太细可以改成 y:y+2
                    #     img_rot[:, y:y+1, :] = grid_color 
                    # # 画垂直线
                    # for x in range(0, W_rot, patch_size):
                    #     img_rot[:, :, x:x+1] = grid_color
                        
                    # # 4. 保存带网格的旋转图像
                    # img_path = os.path.join(save_vis_dir, f'batch_{b_idx}_view_{n_idx}_grid_rot90.jpg')
                    # vutils.save_image(img_rot, img_path)
        # ---------------------------------------------------------
        if save_vis_dir is not None:
            os.makedirs(save_vis_dir, exist_ok=True)
            B_orig, N_orig = imgs.shape[0], imgs.shape[1]
            for b_idx in range(B_orig):
                for n_idx in range(N_orig):
                    img_tensor = imgs[b_idx, n_idx] # (3, H, W)
                    img_path = os.path.join(save_vis_dir, f'batch_{b_idx}_view_{n_idx}.jpg')
                    vutils.save_image(img_tensor, img_path)
        # ---------------------------------------------------------

        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        # [修改] 将 save_vis_dir 透传给 decode
        hidden, pos = self.decode(hidden, N, H, W, save_vis_dir=save_vis_dir)

        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local points
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # confidence
            conf_hidden = conf_hidden.float()
            conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)

            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

            # unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
        )