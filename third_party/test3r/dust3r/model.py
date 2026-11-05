# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
import torch
import os
from packaging import version
import huggingface_hub
import math
from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed
from functools import reduce
import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # type: ignore # noqa
from operator import mul
import torch.nn as nn
inf = float('inf')
from torch.nn import Conv2d, Dropout
hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), ("Outdated huggingface_hub version, please reinstall requirements.txt")
from .modules import Adapter, GatedAdapter, CrossViewAdapter, DeformableAdapterHighRank, DeformableAdapterLowRank, Adapter3R, sliding_window_mask


def load_model(model_path, device, verbose=True, **kwargs):
    if verbose:
        print('Loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if kwargs:
        args = args[:-1]+', '+', '.join(f'{k}={v}' for k, v in kwargs.items())+')'
    if verbose:
        print(f"instantiating:\n{args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print('Missing keys:', s.missing_keys)
        print('Unexpected keys:', s.unexpected_keys)
    return net.to(device)


class AsymmetricCroCo3DStereo (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 use_prompt = False,
                 is_shallow = False,
                 prompt_size = 32,
                 use_adapter = False,
                 reduction_dim = 128,
                 peft_parts = ['enc'],
                 peft_layers = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],[0,1,2,3,4,5,6,7,8,9,10,11]],
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)
        
        self.peft_parts = peft_parts
        self.peft_layers = peft_layers
        # vpt
        self.use_prompt = use_prompt
        print(f"use prompt: {self.use_prompt}")
        if self.use_prompt:
            self.prompt_size = prompt_size
            self.is_shallow = is_shallow
            print(f">> Using {'shallow' if self.is_shallow else 'deep'} Prompt.")
            if self.is_shallow:
                self.prompt_embeddings = nn.Parameter(torch.zeros(1, 1, prompt_size, self.croco_args['enc_embed_dim']))
                self.patch_start_idx = self.prompt_embeddings.shape[2]
            else:
                if 'enc' in self.peft_parts:
                    self.prompt_embeddings = nn.Parameter(torch.zeros(len(self.peft_layers[0]), 1, prompt_size, self.croco_args['enc_embed_dim']))
                    self.patch_start_idx = self.prompt_embeddings.shape[2]
                    nn.init.xavier_uniform_(self.prompt_embeddings)
            if 'dec' in self.peft_parts:
                self.dec_prompt_embeddings = nn.Parameter(torch.zeros(2, len(self.peft_layers[1]), 1, prompt_size, self.croco_args['dec_embed_dim']))
                nn.init.xavier_uniform_(self.dec_prompt_embeddings)
        
        # adapter
        self.use_adapter = use_adapter
        if use_adapter:
            print(">> Using Adapter in each transformer block.")
            # add adapters to each transformer block
            if 'enc' in self.peft_parts:
                self.adapters = nn.ModuleList([Adapter3R(self.croco_args['enc_embed_dim'], reduction_dim, region_scale=0.2, fusion_factor=0.01) for i in range(len(self.peft_layers[0]))])
            if 'dec' in self.peft_parts:
                self.dec_adapters = nn.ModuleList([Adapter3R(self.croco_args['dec_embed_dim'], reduction_dim, region_scale=0.2, fusion_factor=0.01) for i in range(len(self.peft_layers[1]))])
                self.dec_adapters2 = nn.ModuleList([Adapter3R(self.croco_args['dec_embed_dim'], reduction_dim, region_scale=0.2, fusion_factor=0.01) for i in range(len(self.peft_layers[1]))])

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu', **kw)
        else:
            try:
                model = super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)
            except TypeError as e:
                raise Exception(f'tried to load {pretrained_model_name_or_path} from huggingface, but failed')
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none': [],
            'mask': [self.mask_token],
            'encoder': [self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size, **kw):
                            
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)
    
    def _encode_image(self, image, true_shape):
        '''
        image: torch.Size([2, 3, 160, 512])   
        '''
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, None, true_shape=true_shape)  # [2, Token, Dim]
        
        if self.use_prompt and 'enc' in self.peft_parts:
            # x = torch.cat([self.prompt_embeddings[0].expand(2, -1, -1), x], dim=1)
            pos_special = torch.zeros(2, self.patch_start_idx, 2).to(x.device).to(pos.dtype)  # [2,self.patch_start_idx, 2]
            # pos_final = torch.cat([pos_special, pos+1], dim=1)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        for i, blk in enumerate(self.enc_blocks):
            # if self.use_adapter and ('enc' in self.peft_parts and i in self.peft_layers[0]):
            #     a = x
                
            if self.use_prompt and ('enc' in self.peft_parts and i in self.peft_layers[0]):
                x = blk(torch.cat([self.prompt_embeddings[self.peft_layers[0].index(i)].expand(2,-1,-1), x], dim=1), torch.cat([pos_special, pos+1], dim=1))
                x = x[:, self.patch_start_idx:, :]
            else:
                x = blk(x, pos)  #x: [2, N, 1024] pos: [2, N, 2] pos1 [2, 288, 2]
            
            if self.use_adapter and ('enc' in self.peft_parts and i in self.peft_layers[0]):
                if type(self.adapters[self.peft_layers[0].index(i)]) in [Adapter3R, CrossViewAdapter, DeformableAdapterHighRank, DeformableAdapterLowRank]:
                    refs_mask = sliding_window_mask(x.shape[0], width=2, device=x.device)
                    x_refs = x.unsqueeze(0).expand(x.shape[0],-1,-1,-1)[refs_mask].view(x.shape[0], -1, *x.shape[1:])
                    x = x + self.adapters[self.peft_layers[0].index(i)](x, x_refs, img_shape=true_shape[0], patch_size=self.patch_embed.patch_size[0], scale=0.3)
                else:
                    x = x + self.adapters[self.peft_layers[0].index(i)](x, scale=0.3)
            
            # if i != 23 and self.use_prompt and not self.is_shallow and i != 0 and ('enc' in self.peft_parts and i in self.peft_layers[0]):
            #     x[:, :self.patch_start_idx, :] = self.prompt_embeddings[i].expand(2, -1, -1)

        x = self.enc_norm(x)
        
        return x, pos, None

    def update_prompt(self, x, pos1, i):
        x[:, pos1.shape[1]:, :] = self.prompt_embeddings[i].squeeze(0).flatten(2).transpose(1, 2)
        return x


    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0), torch.cat((true_shape1, true_shape2), dim=0))         
            out1, out2 = out.chunk(2, dim=0)
            pos1, pos2 = pos.chunk(2, dim=0)
        else:
            out1, pos1, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out1, out2, pos1, pos2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2, **kwargs):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)
        if self.use_prompt and 'dec' in self.peft_parts:
            prompt_pos = torch.zeros(1, self.patch_start_idx, 2).to(f1.device).to(pos1.dtype)  # [2,self.patch_start_idx, 2]

        final_output.append((f1, f2))
        for i, (blk1, blk2) in enumerate(zip(self.dec_blocks, self.dec_blocks2)):
            if self.use_prompt and ('dec' in self.peft_parts and i in self.peft_layers[1]):
                f1, _ = blk1(final_output[-1][0], torch.cat([self.dec_prompt_embeddings[0, self.peft_layers[1].index(i)], final_output[-1][1]], dim=1), pos1, torch.cat([prompt_pos, pos2+1], dim=1))
                f2, _ = blk2(final_output[-1][1], torch.cat([self.dec_prompt_embeddings[1, self.peft_layers[1].index(i)], final_output[-1][0]], dim=1), pos2, torch.cat([prompt_pos, pos1+1], dim=1))
            else:
                f1, _ = blk1(final_output[-1][0], final_output[-1][1], pos1, pos2) # img1 side
                f2, _ = blk2(final_output[-1][1], final_output[-1][0], pos2, pos1) # img2 side
            
            if self.use_adapter and ('dec' in self.peft_parts and i in self.peft_layers[1]):
                if type(self.dec_adapters[self.peft_layers[1].index(i)]) in [Adapter3R, CrossViewAdapter, DeformableAdapterHighRank, DeformableAdapterLowRank]:
                    x = torch.concat([f1,f2],dim=0)
                    refs_mask = sliding_window_mask(x.shape[0], width=2, device=x.device)
                    x_refs = x.unsqueeze(0).expand(x.shape[0],-1,-1,-1)[refs_mask].view(x.shape[0], -1, *x.shape[1:])
                    f1 = f1 + self.dec_adapters[self.peft_layers[1].index(i)](f1, x_refs[:1], img_shape=kwargs['img_shape'], patch_size=self.patch_embed.patch_size[0], scale=0.3)
                    f2 = f2 + self.dec_adapters2[self.peft_layers[1].index(i)](f2, x_refs[1:], img_shape=kwargs['img_shape'], patch_size=self.patch_embed.patch_size[0], scale=0.3)
                else:
                    f1 = f1 + self.dec_adapters[self.peft_layers[1].index(i)](final_output[-1][0], scale=0.3)
                    f2 = f2 + self.dec_adapters2[self.peft_layers[1].index(i)](final_output[-1][1], scale=0.3)
                # f1 = f1 + self.dec_adapters[self.peft_layers[1].index(i)](f1, scale=0.3)
                # f2 = f2 + self.dec_adapters2[self.peft_layers[1].index(i)](f2, scale=0.3)
            
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)
        img_shape = view1.get('true_shape')[0]
        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2, img_shape=img_shape)
        dec1 = [tok.float() for tok in dec1]
        dec2 = [tok.float() for tok in dec2]
        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, dec1, shape1)
            res2 = self._downstream_head(2, dec2, shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2
