# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed for the inference
# --------------------------------------------------------
from tqdm import tqdm
from tqdm import trange
import torch
from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.misc import invalid_to_nans
from dust3r.utils.geometry import depthmap_to_pts3d, geotrf
import croco.utils.misc as misc
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.functional.regression import pearson_corrcoef
from eval.mv_recon.utils import summary_parameters

def _interleave_imgs(img1, img2):
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor):
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res


def make_batch_symmetric(batch):
    view1, view2 = batch
    view1, view2 = (_interleave_imgs(view1, view2), _interleave_imgs(view2, view1))
    return view1, view2


def loss_of_one_batch(batch, model, criterion, device, symmetrize_batch=False, use_amp=False, ret=None):
    view1, view2 = batch
    ignore_keys = set(['depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'])
    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    if symmetrize_batch:
        view1, view2 = make_batch_symmetric(batch)

    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        pred1, pred2 = model(view1, view2)

        # loss is supposed to be symmetric
        with torch.cuda.amp.autocast(enabled=False):
            loss = criterion(view1, view2, pred1, pred2) if criterion is not None else None

    result = dict(view1=view1, view2=view2, pred1=pred1, pred2=pred2, loss=loss)
    return result[ret] if ret else result



@torch.no_grad()
def inference(pairs, model, device, batch_size=8, verbose=True):
    if verbose:
        print(f'>> Inference with model on {len(pairs)} image pairs')
    result = []

    # first, check if all images have the same size
    multiple_shapes = not (check_if_same_size(pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    for i in trange(0, len(pairs), batch_size, disable=not verbose):
        res = loss_of_one_batch(collate_with_cat(pairs[i:i + batch_size]), model, None, device)
        result.append(to_cpu(res))

    result = collate_with_cat(result, lists=multiple_shapes)

    return result

def inference_ttt(pairs, idx, model, device, batch_size=1, epoches = 2, lr = 0.00001, accum_iter = 4, verbose=True):


    if verbose:
        print(f'>> Inference TTT with model on {len(pairs)} image pairs')
    result = []

    model.train(True)

    for name, param in model.named_parameters():
        if "prompt_embeddings" in name or "adapters" in name: 
            param.requires_grad = True  
        else:
            param.requires_grad = False

    param_groups = misc.get_parameter_groups(model, weight_decay = 0)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    summary_parameters(model, detail=True)

    loss_scaler = NativeScaler()
    optimizer.zero_grad()

    for epoch in range(epoches):
        for i in tqdm(range(len(pairs)), desc=f'ttt {epoch} epoch:'):
            pair1 = [(pairs[i][0], pairs[i][1])]
            pair2 = [(pairs[i][0], pairs[i][2])]
            
            batch1 = collate_with_cat(pair1[0:])
            res1 = loss_of_one_batch(batch1, model, None, device)

            batch2 = collate_with_cat(pair2[0:])
            res2 = loss_of_one_batch(batch2, model, None, device)

            loss = train_criterion(res1["pred1"], res2["pred1"])

            loss = loss/accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(i+1) % accum_iter == 0)
            
            if (i+1) % accum_iter == 0:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = torch.norm(param.grad).item()
                optimizer.zero_grad()
    return model

def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)

def train_criterion_with_conf(res1, res2):
    # loss = ((res1 - res2)** 2).mean()
    conf1 = min_max_normalize(res1['conf']).squeeze(0).unsqueeze(-1)            
    conf2 = min_max_normalize(res2['conf']).squeeze(0).unsqueeze(-1)   
    loss = torch.abs((conf1*res1['pts3d'] - conf2*res2['pts3d'])).mean()
    # loss = torch.abs((res1['pts3d'] - res2['pts3d'])).mean()
    return loss 

def gradient_consistency_loss(res1, res2):
    Pts1 = res1['pts3d']  # [B, H, W, 3]
    Pts2 = res2['pts3d']
    # x direction gradient
    dPts1_dx = Pts1[:, :, 1:, :] - Pts1[:, :, :-1, :]
    dPts2_dx = Pts2[:, :, 1:, :] - Pts2[:, :, :-1, :]
    # y direction gradient
    dPts1_dy = Pts1[:, 1:, :, :] - Pts1[:, :-1, :, :]
    dPts2_dy = Pts2[:, 1:, :, :] - Pts2[:, :-1, :, :]
    # align tensors shape
    dPts1_dx = dPts1_dx[:, :-1, :, :]
    dPts2_dx = dPts2_dx[:, :-1, :, :]
    dPts1_dy = dPts1_dy[:, :, :-1, :]
    dPts2_dy = dPts2_dy[:, :, :-1, :]
    loss_dx = torch.abs(dPts1_dx - dPts2_dx).mean()
    loss_dy = torch.abs(dPts1_dy - dPts2_dy).mean()
    loss = loss_dx + loss_dy

    return loss

def compute_local_normals(Pts):
    dx = Pts[:, :, 1:, :] - Pts[:, :, :-1, :]
    dy = Pts[:, 1:, :, :] - Pts[:, :-1, :, :]
    dx = dx[:, :-1, :, :]
    dy = dy[:, :, :-1, :]
    normals = torch.cross(dx, dy, dim=-1)
    normals = torch.nn.functional.normalize(normals, dim=-1, eps=1e-6)

    return normals

def sobel_normals(pts):
    """
    pts: [B, H, W, 3]
    return: [B, H, W, 3]
    """

    B, H, W, C = pts.shape
    pts = pts.permute(0, 3, 1, 2)  # B,3,H,W
    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=pts.dtype, device=pts.device
    ).view(1, 1, 3, 3) / 8.0

    sobel_y = torch.tensor(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]],
        dtype=pts.dtype, device=pts.device
    ).view(1, 1, 3, 3) / 8.0

    grad_x = []
    grad_y = []

    for c in range(3):
        gx = torch.nn.functional.conv2d(pts[:, c:c+1], sobel_x, padding=1)
        gy = torch.nn.functional.conv2d(pts[:, c:c+1], sobel_y, padding=1)
        grad_x.append(gx)
        grad_y.append(gy)

    grad_x = torch.cat(grad_x, dim=1)  # B,3,H,W
    grad_y = torch.cat(grad_y, dim=1)

    grad_x = grad_x.permute(0,2,3,1)
    grad_y = grad_y.permute(0,2,3,1)

    normals = torch.cross(grad_x, grad_y, dim=-1)
    normals = torch.nn.functional.normalize(normals, dim=-1, eps=1e-6)

    return normals

def central_diff_normals(Pts):
    Pts_pad = torch.nn.functional.pad(Pts.permute(0,3,1,2), (1,1,1,1), mode='replicate')
    Px1 = Pts_pad[:,:,1:-1,2:]
    Px2 = Pts_pad[:,:,1:-1,:-2]
    Py1 = Pts_pad[:,:,2:,1:-1]
    Py2 = Pts_pad[:,:,:-2,1:-1]

    grad_x = (Px1 - Px2) / 2
    grad_y = (Py1 - Py2) / 2
    grad_x = grad_x.permute(0,2,3,1)
    grad_y = grad_y.permute(0,2,3,1)

    normals = torch.cross(grad_x, grad_y, dim=-1)
    normals = torch.nn.functional.normalize(normals, dim=-1, eps=1e-6)

    return normals

def pca_normals(pts, ksize=5):
    """
    pts: [B, H, W, 3]
    return: [B, H, W, 3]
    """
    B, H, W, _ = pts.shape
    pts = pts.permute(0,3,1,2)  # B,3,H,W

    unfold = torch.nn.functional.unfold(pts, kernel_size=ksize, padding=ksize//2)
    unfold = unfold.view(B, 3, ksize*ksize, H*W)
    unfold = unfold.permute(0,3,2,1)  # B,HW,K,3

    mean = unfold.mean(dim=2, keepdim=True)
    centered = unfold - mean
    cov = torch.matmul(centered.transpose(-1,-2), centered)  # B,HW,3,3

    # SVD
    _, _, V = torch.linalg.svd(cov)
    normals = V[:,:,-1,:]  # smallest singular vector
    normals = normals.view(B,H,W,3)
    normals = torch.nn.functional.normalize(normals, dim=-1, eps=1e-6)

    return normals


def local_plane_consistency_loss(res1, res2):
    Pts1 = res1['pts3d']
    Pts2 = res2['pts3d']
    n1 = compute_local_normals(Pts1)
    Pts1_c = Pts1[:, :-1, :-1, :]
    Pts2_c = Pts2[:, :-1, :-1, :]
    diff = Pts2_c - Pts1_c
    # project to the normal direction
    plane_error = torch.abs((n1 * diff).sum(dim=-1))

    return plane_error.mean()

def normal_consistency_loss(res1, res2):
    Pts1 = res1['pts3d']
    Pts2 = res2['pts3d']
    n1 = pca_normals(Pts1)
    n2 = pca_normals(Pts2)
    cos = (n1 * n2).sum(-1)
    loss = (1 - cos).mean()
    return loss

def self_depth_consistency_loss(res1, res2, with_conf=False):
    Pts1 = res1['pts3d']
    Pts2 = res2['pts3d']
    loss = torch.abs((Pts1 - Pts2))
    if with_conf:
        Conf1 = res1['conf']
        Conf2 = res2['conf']
        conf = (Conf1+Conf2)/2
        alpha = 0.2
        loss = loss * conf.unsqueeze(-1) -  alpha * torch.log(conf.unsqueeze(-1))
    loss = loss.mean()
    return loss 

def train_criterion(res1, res2):
    depth_loss = self_depth_consistency_loss(res1, res2, with_conf=True)
    # normal_loss = normal_consistency_loss(res1, res2)
    # loss = depth_loss
    # plane_loss = local_plane_consistency_loss(res1, res2)
    # gradient_loss = gradient_consistency_loss(res1, res2)
    # loss = depth_loss + 0.1*gradient_loss 
    # loss = depth_loss + 0.1*plane_loss
    loss = depth_loss
    return loss 

def train_criterion_with_gt(res_gt1, res_gt2, res1, res2):
    loss = torch.abs((res1 - res2)).mean() + 0.05*torch.abs((res_gt1 - res1)).mean() + 0.05*torch.abs((res_gt2 - res2)).mean()
    return loss 


def check_if_same_size(pairs):
    shapes1 = [img1['img'].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2['img'].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(shapes2[0] == s for s in shapes2)


def get_pred_pts3d(gt, pred, use_pose=False):
    if 'depth' in pred and 'pseudo_focal' in pred:
        try:
            pp = gt['camera_intrinsics'][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif 'pts3d' in pred:
        # pts3d from my camera
        pts3d = pred['pts3d']

    elif 'pts3d_in_other_view' in pred:
        # pts3d from the other camera, already transformed
        assert use_pose is True
        return pred['pts3d_in_other_view']  # return!

    if use_pose:
        camera_pose = pred.get('camera_pose')
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d


def find_opt_scaling(gt_pts1, gt_pts2, pr_pts1, pr_pts2=None, fit_mode='weiszfeld_stop_grad', valid1=None, valid2=None):
    assert gt_pts1.ndim == pr_pts1.ndim == 4
    assert gt_pts1.shape == pr_pts1.shape
    if gt_pts2 is not None:
        assert gt_pts2.ndim == pr_pts2.ndim == 4
        assert gt_pts2.shape == pr_pts2.shape

    # concat the pointcloud
    nan_gt_pts1 = invalid_to_nans(gt_pts1, valid1).flatten(1, 2)
    nan_gt_pts2 = invalid_to_nans(gt_pts2, valid2).flatten(1, 2) if gt_pts2 is not None else None

    pr_pts1 = invalid_to_nans(pr_pts1, valid1).flatten(1, 2)
    pr_pts2 = invalid_to_nans(pr_pts2, valid2).flatten(1, 2) if pr_pts2 is not None else None

    all_gt = torch.cat((nan_gt_pts1, nan_gt_pts2), dim=1) if gt_pts2 is not None else nan_gt_pts1
    all_pr = torch.cat((pr_pts1, pr_pts2), dim=1) if pr_pts2 is not None else pr_pts1

    dot_gt_pr = (all_pr * all_gt).sum(dim=-1)
    dot_gt_gt = all_gt.square().sum(dim=-1)

    if fit_mode.startswith('avg'):
        # scaling = (all_pr / all_gt).view(B, -1).mean(dim=1)
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
    elif fit_mode.startswith('median'):
        scaling = (dot_gt_pr / dot_gt_gt).nanmedian(dim=1).values
    elif fit_mode.startswith('weiszfeld'):
        # init scaling with l2 closed form
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (all_pr - scaling.view(-1, 1, 1) * all_gt).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip_(min=1e-8).reciprocal()
            # update the scaling with the new weights
            scaling = (w * dot_gt_pr).nanmean(dim=1) / (w * dot_gt_gt).nanmean(dim=1)
    else:
        raise ValueError(f'bad {fit_mode=}')

    if fit_mode.endswith('stop_grad'):
        scaling = scaling.detach()

    scaling = scaling.clip(min=1e-3)
    # assert scaling.isfinite().all(), bb()
    return scaling
