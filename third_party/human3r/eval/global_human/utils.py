from copy import deepcopy
import cv2
import os
import re
import numpy as np
import torch
import roma
import math
import tqdm
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.spatial.transform import Rotation
from eval.relpose.evo_utils import *
from PIL import Image
import imageio.v2 as iio
from matplotlib.figure import Figure
from itertools import product

def todevice(batch, device, callback=None, non_blocking=False):
    if callback:
        batch = callback(batch)
    if isinstance(batch, dict):
        return {k: todevice(v, device) for k, v in batch.items()}
    if isinstance(batch, (tuple, list)):
        return type(batch)((todevice(x, device) for x in batch))
    x = batch
    if device == 'numpy':
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x
to_device = todevice

def to_numpy(x):
    return todevice(x, 'numpy')

def compute_prf1(count, miss, fp):
    if count == 0:
        return (0, 0, 0)
    all_tp = count - miss
    all_fp = fp
    all_fn = miss
    if all_tp == 0:
        return (0.0, 0.0, 0.0)
    all_f1_score = round(all_tp / (all_tp + 0.5 * (all_fp + all_fn)), 2)
    all_recall = round(all_tp / (all_tp + all_fn), 2)
    all_precision = round(all_tp / (all_tp + all_fp), 2)
    return (100.0 * all_precision, 100.0 * all_recall, 100.0 * all_f1_score)

def match_2d_greedy(pred_kps, gtkp, valid_mask, imgPath=None, baseline=None, iou_thresh=0.05, valid=None, ind=-1):
    predList = np.arange(len(pred_kps))
    gtList = np.arange(len(gtkp))
    combs = list(product(predList, gtList))
    errors_per_pair = {}
    errors_per_pair_list = []
    for comb in combs:
        vmask = valid_mask[comb[1]]
        assert vmask.sum() > 0, print('no valid points')
        errors_per_pair[str(comb)] = np.linalg.norm(pred_kps[comb[0]][vmask, :2] - gtkp[comb[1]][vmask, :2], 2)
        errors_per_pair_list.append(errors_per_pair[str(comb)])
    gtAssigned = np.zeros((len(gtkp),), dtype=bool)
    opAssigned = np.zeros((len(pred_kps),), dtype=bool)
    errors_per_pair_list = np.array(errors_per_pair_list)
    bestMatch = []
    excludedGtBecauseInvalid = []
    falsePositiveCounter = 0
    while np.sum(gtAssigned) < len(gtAssigned) and np.sum(opAssigned) + falsePositiveCounter < len(pred_kps):
        found = False
        falsePositive = False
        while not found:
            if sum(np.inf == errors_per_pair_list) == len(errors_per_pair_list):
                print('something went wrong here')
            minIdx = np.argmin(errors_per_pair_list)
            minComb = combs[minIdx]
            iou = get_bbx_overlap(pred_kps[minComb[0]], gtkp[minComb[1]])
            if not opAssigned[minComb[0]] and (not gtAssigned[minComb[1]]) and (iou >= iou_thresh):
                found = True
                errors_per_pair_list[minIdx] = np.inf
            else:
                errors_per_pair_list[minIdx] = np.inf
                if iou < iou_thresh:
                    found = True
                    falsePositive = True
                    falsePositiveCounter += 1
        if not valid is None:
            if valid[minComb[1]]:
                if not falsePositive:
                    bestMatch.append(minComb)
                    opAssigned[minComb[0]] = True
                    gtAssigned[minComb[1]] = True
            else:
                gtAssigned[minComb[1]] = True
                excludedGtBecauseInvalid.append(minComb[1])
        elif not falsePositive:
            bestMatch.append(minComb)
            opAssigned[minComb[0]] = True
            gtAssigned[minComb[1]] = True
    bestMatch = np.array(bestMatch)
    opAssigned = []
    gtAssigned = []
    for pair in bestMatch:
        opAssigned.append(pair[0])
        gtAssigned.append(pair[1])
    opAssigned.sort()
    gtAssigned.sort()
    falsePositives = []
    misses = []
    opIds = np.arange(len(pred_kps))
    notAssignedIds = np.setdiff1d(opIds, opAssigned)
    for notAssignedId in notAssignedIds:
        falsePositives.append(notAssignedId)
    gtIds = np.arange(len(gtList))
    notAssignedIdsGt = np.setdiff1d(gtIds, gtAssigned)
    for notAssignedIdGt in notAssignedIdsGt:
        if not valid is None:
            if valid[notAssignedIdGt]:
                misses.append(notAssignedIdGt)
            else:
                excludedGtBecauseInvalid.append(notAssignedIdGt)
        else:
            misses.append(notAssignedIdGt)
    return (bestMatch, falsePositives, misses)

def get_bbx_overlap(p1, p2):
    min_p1 = np.min(p1, axis=0)
    min_p2 = np.min(p2, axis=0)
    max_p1 = np.max(p1, axis=0)
    max_p2 = np.max(p2, axis=0)
    bb1 = {}
    bb2 = {}
    bb1['x1'] = min_p1[0]
    bb1['x2'] = max_p1[0]
    bb1['y1'] = min_p1[1]
    bb1['y2'] = max_p1[1]
    bb2['x1'] = min_p2[0]
    bb2['x2'] = max_p2[0]
    bb2['y1'] = min_p2[1]
    bb2['y2'] = max_p2[1]
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)
    bb1_area = (bb1['x2'] - bb1['x1'] + 1) * (bb1['y2'] - bb1['y1'] + 1)
    bb2_area = (bb2['x2'] - bb2['x1'] + 1) * (bb2['y2'] - bb2['y1'] + 1)
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou

def avg_per_human(lst, default=0.0):
    values = [x for arr in lst for x in np.array(arr).flatten()]
    return np.mean(values) if values else default

def extract_metrics(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    matches = re.findall('n_human: (\\d+).*?PVE: ([\\d.]+), PA-PVE: ([\\d.]+), Metric-PVE: ([\\d.]+), MPJPE: ([\\d.]+), PA-MPJPE: ([\\d.]+), Metric-MPJPE: ([\\d.]+), RootError: ([\\d.]+), W-MPJPE: ([\\d.]+), WA-MPJPE: ([\\d.]+), RTE: ([\\d.]+), Scaled-RTE: ([\\d.]+), Jitter: ([\\d.]+), Foot-Sliding: ([\\d.]+), Precision: ([\\d.]+), Recall: ([\\d.]+), F1-Score: ([\\d.]+)', content)
    if not matches:
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total_humans = 0
    weighted_sums = [0.0] * 16
    for match in matches:
        n_human = int(match[0])
        metrics = [float(x) for x in match[1:]]
        total_humans += n_human
        for i, metric in enumerate(metrics):
            weighted_sums[i] += metric * n_human
    if total_humans == 0:
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    averages = [ws / total_humans for ws in weighted_sums]
    return (total_humans, *averages)

def process_directory(directory):
    results = []
    for root, _, files in os.walk(directory):
        for file in sorted(files):
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                result = extract_metrics(file_path)
                if result[0] > 0:
                    results.append(result)
    return results

def calculate_averages(results):
    if not results:
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total_humans = sum((r[0] for r in results))
    if total_humans == 0:
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    weighted_sums = [sum((r[i] * r[0] for r in results)) for i in range(1, 17)]
    averages = [ws / total_humans for ws in weighted_sums]
    return (total_humans, *averages)

def visualize(save_dir, img_path, view, gt_v3d_c, pred_v3d_c, K_to_proj, gt_K, bestMatch, smpl_face):
    from dust3r.utils import render_meshes, denormalize_rgb
    from matplotlib import colormaps
    def get_color(index):
        return np.asarray(colormaps["tab20"](int(index) % 20)[:3]) * 255
    from PIL import Image
    os.makedirs(save_dir, exist_ok=True)
    n_humans_i = pred_v3d_c.shape[0]
    img_array = denormalize_rgb(view.cpu().numpy())
    focal = K_to_proj[[0, 1], [0, 1]].cpu().numpy()
    princpt = K_to_proj[[0, 1], [-1, -1]].cpu().numpy()
    gt_focal = gt_K[[0, 1], [0, 1]].cpu().numpy()
    gt_princpt = gt_K[[0, 1], [-1, -1]].cpu().numpy()
    gt_color_indices = list(range(gt_v3d_c.shape[0]))
    pred_color_indices = [-1] * n_humans_i
    for pid, gid in bestMatch:
        pred_color_indices[pid] = gt_color_indices[gid]
    next_color_idx = max(gt_color_indices) + 1 if gt_color_indices else 0
    for i in range(len(pred_color_indices)):
        if pred_color_indices[i] == -1:
            pred_color_indices[i] = next_color_idx
            next_color_idx += 1
    gt_verts, gt_faces = ([], [])
    for j in range(gt_v3d_c.shape[0]):
        gt_verts.append(gt_v3d_c[j].cpu().numpy().reshape(-1, 3))
        gt_faces.append(smpl_face)
    gt_colors = [get_color(gt_color_indices[j]) / 255 for j in range(len(gt_verts))]
    gt_rend_array = render_meshes(img_array.copy(), gt_verts, gt_faces, {'focal': gt_focal, 'princpt': gt_princpt}, color=gt_colors)
    pred_verts, pred_faces = ([], [])
    for j in range(n_humans_i):
        pred_verts.append(pred_v3d_c[j].cpu().numpy().reshape(-1, 3))
        pred_faces.append(smpl_face)
    pred_colors = [get_color(pred_color_indices[j]) / 255 for j in range(len(pred_verts))]
    pred_rend_array = render_meshes(img_array.copy(), pred_verts, pred_faces, {'focal': focal, 'princpt': princpt}, color=pred_colors)
    img = np.concatenate([img_array, pred_rend_array, gt_rend_array], 1)
    Image.fromarray(img).save(os.path.join(f'{save_dir}/{os.path.splitext(os.path.basename(img_path))[0]}.jpg'))
import os
import numpy as np
from PIL import Image

def visualize3d(save_dir, img_path, view, gt_v3d_c, pred_v3d_c, K_to_proj, gt_K, bestMatch, smpl_face, pred_c2w=None, gt_c2w=None, pred_pts3d=None, gt_depthmap=None, msk=None, conf=None):
    from dust3r.utils import render_meshes, denormalize_rgb
    from matplotlib import colormaps
    def get_color(index):
        return np.asarray(colormaps["tab20"](int(index) % 20)[:3]) * 255
    os.makedirs(save_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    n_humans_i = pred_v3d_c.shape[0]
    img_array = denormalize_rgb(view.cpu().numpy())
    focal = K_to_proj[[0, 1], [0, 1]].cpu().numpy()
    princpt = K_to_proj[[0, 1], [-1, -1]].cpu().numpy()
    gt_focal = gt_K[[0, 1], [0, 1]].cpu().numpy()
    gt_princpt = gt_K[[0, 1], [-1, -1]].cpu().numpy()
    gt_color_indices = list(range(gt_v3d_c.shape[0]))
    pred_color_indices = [-1] * n_humans_i
    for pid, gid in bestMatch:
        pred_color_indices[pid] = gt_color_indices[gid]
    next_color_idx = max(gt_color_indices) + 1 if gt_color_indices else 0
    for i in range(len(pred_color_indices)):
        if pred_color_indices[i] == -1:
            pred_color_indices[i] = next_color_idx
            next_color_idx += 1
    gt_verts, gt_faces = ([], [])
    for j in range(gt_v3d_c.shape[0]):
        gt_verts.append(gt_v3d_c[j].cpu().numpy().reshape(-1, 3))
        gt_faces.append(smpl_face)
    gt_colors = [get_color(gt_color_indices[j]) / 255 for j in range(len(gt_verts))]
    gt_rend_array = render_meshes(img_array.copy(), gt_verts, gt_faces, {'focal': gt_focal, 'princpt': gt_princpt}, color=gt_colors)
    pred_verts, pred_faces = ([], [])
    for j in range(n_humans_i):
        pred_verts.append(pred_v3d_c[j].cpu().numpy().reshape(-1, 3))
        pred_faces.append(smpl_face)
    pred_colors = [get_color(pred_color_indices[j]) / 255 for j in range(len(pred_verts))]
    pred_rend_array = render_meshes(img_array.copy(), pred_verts, pred_faces, {'focal': focal, 'princpt': princpt}, color=pred_colors)
    img = np.concatenate([img_array, pred_rend_array, gt_rend_array], 1)
    Image.fromarray(img).save(os.path.join(save_dir, f'{base_filename}.jpg'))

    def to_np(data):
        if data is None:
            return None
        return data.detach().cpu().numpy() if hasattr(data, 'cpu') else data
    unified_data = {'image': img_array, 'pred_K': K_to_proj.cpu().numpy(), 'gt_K': gt_K.cpu().numpy(), 'smpl_faces': np.array(smpl_face)}
    unified_data['pred_verts'] = np.array(pred_verts) if len(pred_verts) > 0 else np.empty((0, 6890, 3))
    unified_data['gt_verts'] = np.array(gt_verts) if len(gt_verts) > 0 else np.empty((0, 6890, 3))
    unified_data['pred_colors'] = np.array(pred_colors)
    unified_data['gt_colors'] = np.array(gt_colors)
    if pred_c2w is not None:
        unified_data['pred_c2w'] = to_np(pred_c2w)
    if gt_c2w is not None:
        unified_data['gt_c2w'] = to_np(gt_c2w)
    if pred_pts3d is not None:
        unified_data['pred_pts3d'] = to_np(pred_pts3d)
    if gt_depthmap is not None:
        unified_data['gt_depthmap'] = to_np(gt_depthmap)
    if msk is not None:
        unified_data['msk'] = to_np(msk)
    if conf is not None:
        unified_data['conf'] = to_np(conf)
    np.savez_compressed(os.path.join(save_dir, f'{base_filename}.npz'), **unified_data)

def write_log(log_path, dataset, seq, counter, metrics):
    with open(log_path, 'a') as f:
        f.write(f"{dataset}-{seq: <16} |  n_human: {counter['n_human']:06d} | PVE: {avg_per_human(metrics['ca_pve']):.1f}, PA-PVE: {avg_per_human(metrics['pa_pve']):.1f}, Metric-PVE: {avg_per_human(metrics['me_pve']):.1f}, MPJPE: {avg_per_human(metrics['ca_mpjpe']):.1f}, PA-MPJPE: {avg_per_human(metrics['pa_mpjpe']):.1f}, Metric-MPJPE: {avg_per_human(metrics['me_mpjpe']):.1f}, RootError: {avg_per_human(metrics['rt_error']):.1f}, W-MPJPE: {avg_per_human(metrics['wa2_mpjpe']):.1f}, WA-MPJPE: {avg_per_human(metrics['waa_mpjpe']):.1f}, RTE: {avg_per_human(metrics['rte']):.1f}, Scaled-RTE: {avg_per_human(metrics['rte_scaled']):.1f}, Jitter: {avg_per_human(metrics['jitter']):.1f}, Foot-Sliding: {avg_per_human(metrics['fs']):.1f}, Precision: {metrics['precision']:.1f}, Recall: {metrics['recall']:.1f}, F1-Score: {metrics['f1_score']:.1f}\n")

def get_summary_log(summary, depth_metrics=None, reconstruction_metrics=None):
    log = f"EVALUATION SUMMARY\n{'=' * 7}EVALUATION SUMMARY{'=' * 7}\nTotal Humans: {summary[0]}\n\nCamera Coordinate Metrics (mm):\n  PVE:         {summary[1]:6.1f}\n  PA-PVE:      {summary[2]:6.1f}\n  Metric-PVE:  {summary[3]:6.1f}\n\n  MPJPE:       {summary[4]:6.1f}\n  PA-MPJPE:    {summary[5]:6.1f}\n  Metric-MPJPE: {summary[6]:6.1f}\n  Root-Error:  {summary[7]:6.1f}\n\nGlobal Coordinate Metrics (cm):\n  W-MPJPE:     {summary[8]:6.1f}\n  WA-MPJPE:    {summary[9]:6.1f}\n  RTE:         {summary[10]:6.1f}\n  Scaled-RTE:  {summary[11]:6.1f}\n  Jitter:      {summary[12]:6.1f}\n  Foot-Sliding: {summary[13]:6.1f}\n\nDetection Metrics (%):\n  Precision:   {summary[14]:6.1f}\n  Recall:      {summary[15]:6.1f}\n  F1-Score:    {summary[16]:6.1f}\n{'=' * 32}\n"
    if depth_metrics:
        log += f"\nDepth Estimation Metrics:\n  Abs Rel:   {depth_metrics['Abs Rel']:6.4f}\n  Sq Rel:    {depth_metrics['Sq Rel']:6.4f}\n  RMSE:      {depth_metrics['RMSE']:6.4f}\n  Log RMSE:  {depth_metrics['Log RMSE']:6.4f}\n  δ<1.25:    {depth_metrics['δ < 1.25']:6.4f}\n  δ<1.25^2:  {depth_metrics['δ < 1.25^2']:6.4f}\n  δ<1.25^3:  {depth_metrics['δ < 1.25^3']:6.4f}\n{'=' * 32}\n"
    if reconstruction_metrics:
        log += f"\nScene Reconstruction Metrics:\n  Acc:       {reconstruction_metrics['acc']:6.4f}\n  Comp:      {reconstruction_metrics['comp']:6.4f}\n  Acc_med:   {reconstruction_metrics['acc_med']:6.4f}\n  Comp_med:  {reconstruction_metrics['comp_med']:6.4f}\n  NC:        {reconstruction_metrics['nc']:6.4f}\n  NC1:       {reconstruction_metrics['nc1']:6.4f}\n  NC2:       {reconstruction_metrics['nc2']:6.4f}\n{'=' * 32}\n"
    return log

def compute_jpe(S1, S2):
    return torch.sqrt(((S1 - S2) ** 2).sum(dim=-1)).mean(dim=-1).numpy()

def compute_perjoint_jpe(S1, S2):
    return torch.sqrt(((S1 - S2) ** 2).sum(dim=-1)).numpy()

def batch_compute_similarity_transform_torch(S1, S2):
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    transposed = True
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2
    var1 = torch.sum(X1 ** 2, dim=1).sum(dim=1)
    K = X1.bmm(X2.permute(0, 2, 1))
    U, s, V = torch.svd(K)
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1
    t = mu2 - scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(mu1)
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t
    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)
    return S1_hat

def batch_align_by_pelvis(data_list, pelvis_idxs):
    pred_j3d, target_j3d, pred_verts, target_verts = data_list
    pred_pelvis = pred_j3d[:, pelvis_idxs].mean(dim=1, keepdims=True).clone()
    target_pelvis = target_j3d[:, pelvis_idxs].mean(dim=1, keepdims=True).clone()
    pred_j3d = pred_j3d - pred_pelvis
    target_j3d = target_j3d - target_pelvis
    pred_verts = pred_verts - pred_pelvis
    target_verts = target_verts - target_pelvis
    return (pred_j3d, target_j3d, pred_verts, target_verts, pred_pelvis, target_pelvis)

def align_pcl(Y, X, weight=None, fixed_scale=False):
    *dims, N, _ = Y.shape
    N = torch.ones(*dims, 1, 1) * N
    if weight is not None:
        Y = Y * weight
        X = X * weight
        N = weight.sum(dim=-2, keepdim=True)
    my = Y.sum(dim=-2) / N[..., 0]
    mx = X.sum(dim=-2) / N[..., 0]
    y0 = Y - my[..., None, :]
    x0 = X - mx[..., None, :]
    if weight is not None:
        y0 = y0 * weight
        x0 = x0 * weight
    C = torch.matmul(y0.transpose(-1, -2), x0) / N
    U, D, Vh = torch.linalg.svd(C)
    S = torch.eye(3).reshape(*(1,) * len(dims), 3, 3).repeat(*dims, 1, 1)
    neg = torch.det(U) * torch.det(Vh.transpose(-1, -2)) < 0
    S[neg, 2, 2] = -1
    R = torch.matmul(U, torch.matmul(S, Vh))
    D = torch.diag_embed(D)
    if fixed_scale:
        s = torch.ones(*dims, 1, device=Y.device, dtype=torch.float32)
    else:
        var = torch.sum(torch.square(x0), dim=(-1, -2), keepdim=True) / N
        s = torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) / var[..., 0]
    t = my - s * torch.matmul(R, mx[..., None])[..., 0]
    return (s, R, t)

def global_align_joints(gt_joints, pred_joints):
    s_glob, R_glob, t_glob = align_pcl(gt_joints.reshape(-1, 3), pred_joints.reshape(-1, 3))
    pred_glob = s_glob * torch.einsum('ij,tnj->tni', R_glob, pred_joints) + t_glob[None, None]
    return pred_glob

def first_align_joints(gt_joints, pred_joints):
    s_first, R_first, t_first = align_pcl(gt_joints[:2].reshape(1, -1, 3), pred_joints[:2].reshape(1, -1, 3))
    pred_first = s_first * torch.einsum('tij,tnj->tni', R_first, pred_joints) + t_first[:, None]
    return pred_first

def compute_rte(target_trans, pred_trans, fixed_scale=True):
    scale, rot, trans = align_pcl(target_trans[None, :], pred_trans[None, :], fixed_scale=fixed_scale)
    pred_trans_hat = (scale * torch.einsum('tij,tnj->tni', rot, pred_trans[None, :]) + trans[None, :])[0]
    disps, disp = ([], 0)
    for p1, p2 in zip(target_trans, target_trans[1:]):
        delta = (p2 - p1).norm(2, dim=-1)
        disp += delta
        disps.append(disp)
    rte = torch.norm(target_trans - pred_trans_hat, 2, dim=-1)
    return (rte / disp).numpy()

def compute_jitter(joints, fps=30):
    pred_jitter = torch.norm((joints[3:] - 3 * joints[2:-1] + 3 * joints[1:-2] - joints[:-3]) * fps ** 3, dim=2).mean(dim=-1)
    return pred_jitter.cpu().numpy() / 10.0

def compute_foot_sliding(target_verts, pred_verts, thr=0.01):
    assert target_verts.shape == pred_verts.shape
    assert target_verts.shape[-2] == 6890
    foot_idxs = [3216, 3387, 6617, 6787]
    foot_loc = target_verts[:, foot_idxs]
    foot_disp = (foot_loc[1:] - foot_loc[:-1]).norm(2, dim=-1)
    contact = foot_disp[:] < thr
    pred_feet_loc = pred_verts[:, foot_idxs]
    pred_disp = (pred_feet_loc[1:] - pred_feet_loc[:-1]).norm(2, dim=-1)
    error = pred_disp[contact]
    return error.cpu().numpy()

def eval_camcoord(batch, pelvis_idxs=[1, 2], fps=30):
    pred_j3d = batch['pred_j3d']
    target_j3d = batch['target_j3d']
    pred_verts = batch['pred_v3d']
    target_verts = batch['target_v3d']
    ca_pred_j3d, ca_target_j3d, ca_pred_verts, ca_target_verts, pred_pelvis, target_pelvis = batch_align_by_pelvis([pred_j3d, target_j3d, pred_verts, target_verts], pelvis_idxs)
    pa_pred_j3d = batch_compute_similarity_transform_torch(ca_pred_j3d, ca_target_j3d)
    pa_pred_verts = batch_compute_similarity_transform_torch(ca_pred_verts, ca_target_verts)
    m2mm = 1000
    rt_error = compute_jpe(pred_pelvis, target_pelvis) * m2mm
    me_mpjpe = compute_jpe(pred_j3d, target_j3d) * m2mm
    me_pve = compute_jpe(pred_verts, target_verts) * m2mm
    ca_mpjpe = compute_jpe(ca_pred_j3d, ca_target_j3d) * m2mm
    ca_pve = compute_jpe(ca_pred_verts, ca_target_verts) * m2mm
    pa_mpjpe = compute_jpe(pa_pred_j3d, ca_target_j3d) * m2mm
    pa_pve = compute_jpe(pa_pred_verts, ca_target_verts) * m2mm
    camcoord_metrics = {'me_mpjpe': me_mpjpe, 'ca_mpjpe': ca_mpjpe, 'pa_mpjpe': pa_mpjpe, 'me_pve': me_pve, 'ca_pve': ca_pve, 'pa_pve': pa_pve, 'rt_error': rt_error}
    return camcoord_metrics

def eval_global(batch, subsample=1):
    pred_j3d_glob = batch['pred_j3d']
    target_j3d_glob = batch['target_j3d']
    pred_verts_glob = batch['pred_v3d']
    target_verts_glob = batch['target_v3d']
    seq_length = pred_j3d_glob.shape[0]
    chunk_length = int(100 / subsample)
    wa2_mpjpe, waa_mpjpe = ([], [])
    for start in range(0, seq_length, chunk_length):
        end = min(seq_length, start + chunk_length)
        target_j3d = target_j3d_glob[start:end].clone().cpu()
        pred_j3d = pred_j3d_glob[start:end].clone().cpu()
        w_j3d = first_align_joints(target_j3d, pred_j3d)
        wa_j3d = global_align_joints(target_j3d, pred_j3d)
        wa2_mpjpe.append(compute_jpe(target_j3d, w_j3d))
        waa_mpjpe.append(compute_jpe(target_j3d, wa_j3d))
    m2mm = 1000
    wa2_mpjpe = np.concatenate(wa2_mpjpe) * m2mm
    waa_mpjpe = np.concatenate(waa_mpjpe) * m2mm
    rte = compute_rte(target_j3d_glob[:, 0].cpu(), pred_j3d_glob[:, 0].cpu()) * 100.0
    rte_scaled = compute_rte(target_j3d_glob[:, 0].cpu(), pred_j3d_glob[:, 0].cpu(), fixed_scale=False) * 100.0
    jitter = compute_jitter(pred_j3d_glob, fps=30)
    foot_sliding = compute_foot_sliding(target_verts_glob, pred_verts_glob) * m2mm
    global_metrics = {'wa2_mpjpe': wa2_mpjpe, 'waa_mpjpe': waa_mpjpe, 'rte': rte, 'rte_scaled': rte_scaled, 'jitter': jitter, 'fs': foot_sliding}
    return global_metrics

def absolute_error_loss(params, predicted_depth, ground_truth_depth):
    s, t = params
    predicted_aligned = s * predicted_depth + t
    abs_error = np.abs(predicted_aligned - ground_truth_depth)
    return np.sum(abs_error)

def absolute_value_scaling(predicted_depth, ground_truth_depth, s=1, t=0):
    predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1)
    ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1)
    initial_params = [s, t]
    from scipy.optimize import minimize
    result = minimize(absolute_error_loss, initial_params, args=(predicted_depth_np, ground_truth_depth_np))
    s, t = result.x
    return (s, t)

def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return (disparity, non_negtive_mask)
    else:
        return disparity

def absolute_value_scaling2(predicted_depth, ground_truth_depth, s_init=1.0, t_init=0.0, lr=0.0001, max_iters=1000, tol=1e-06):
    s = torch.tensor([s_init], requires_grad=True, device=predicted_depth.device, dtype=predicted_depth.dtype)
    t = torch.tensor([t_init], requires_grad=True, device=predicted_depth.device, dtype=predicted_depth.dtype)
    optimizer = torch.optim.Adam([s, t], lr=lr)
    prev_loss = None
    for i in range(max_iters):
        optimizer.zero_grad()
        predicted_aligned = s * predicted_depth + t
        abs_error = torch.abs(predicted_aligned - ground_truth_depth)
        loss = torch.sum(abs_error)
        loss.backward()
        optimizer.step()
        if prev_loss is not None and torch.abs(prev_loss - loss) < tol:
            break
        prev_loss = loss.item()
    return (s.detach().item(), t.detach().item())

def depth_evaluation(predicted_depth_original, ground_truth_depth_original, max_depth=80, custom_mask=None, post_clip_min=None, post_clip_max=None, pre_clip_min=None, pre_clip_max=None, align_with_lstsq=False, align_with_lad=False, align_with_lad2=False, metric_scale=False, lr=0.0001, max_iters=1000, use_gpu=False, align_with_scale=False, disp_input=False):
    if isinstance(predicted_depth_original, np.ndarray):
        predicted_depth_original = torch.from_numpy(predicted_depth_original)
    if isinstance(ground_truth_depth_original, np.ndarray):
        ground_truth_depth_original = torch.from_numpy(ground_truth_depth_original)
    if custom_mask is not None and isinstance(custom_mask, np.ndarray):
        custom_mask = torch.from_numpy(custom_mask)
    if predicted_depth_original.dim() == 3:
        _, h, w = predicted_depth_original.shape
        predicted_depth_original = predicted_depth_original.view(-1, w)
        ground_truth_depth_original = ground_truth_depth_original.view(-1, w)
        if custom_mask is not None:
            custom_mask = custom_mask.view(-1, w)
    if use_gpu:
        predicted_depth_original = predicted_depth_original.cuda()
        ground_truth_depth_original = ground_truth_depth_original.cuda()
    if max_depth is not None:
        mask = (ground_truth_depth_original > 0) & (ground_truth_depth_original < max_depth)
    else:
        mask = ground_truth_depth_original > 0
    predicted_depth = predicted_depth_original[mask]
    ground_truth_depth = ground_truth_depth_original[mask]
    if pre_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=pre_clip_min)
    if pre_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=pre_clip_max)
    if disp_input:
        real_gt = ground_truth_depth.clone()
        ground_truth_depth = 1 / (ground_truth_depth + 1e-08)
    if metric_scale:
        predicted_depth = predicted_depth
    elif align_with_lstsq:
        predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1, 1)
        ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1, 1)
        A = np.hstack([predicted_depth_np, np.ones_like(predicted_depth_np)])
        result = np.linalg.lstsq(A, ground_truth_depth_np, rcond=None)
        s, t = (result[0][0], result[0][1])
        s = torch.tensor(s, device=predicted_depth_original.device)
        t = torch.tensor(t, device=predicted_depth_original.device)
        predicted_depth = s * predicted_depth + t
    elif align_with_lad:
        s, t = absolute_value_scaling(predicted_depth, ground_truth_depth, s=torch.median(ground_truth_depth) / torch.median(predicted_depth))
        predicted_depth = s * predicted_depth + t
    elif align_with_lad2:
        s_init = (torch.median(ground_truth_depth) / torch.median(predicted_depth)).item()
        s, t = absolute_value_scaling2(predicted_depth, ground_truth_depth, s_init=s_init, lr=lr, max_iters=max_iters)
        predicted_depth = s * predicted_depth + t
    elif align_with_scale:
        dot_pred_gt = torch.nanmean(ground_truth_depth)
        dot_pred_pred = torch.nanmean(predicted_depth)
        s = dot_pred_gt / dot_pred_pred
        for _ in range(10):
            residuals = s * predicted_depth - ground_truth_depth
            abs_residuals = residuals.abs() + 1e-08
            weights = 1.0 / abs_residuals
            weighted_dot_pred_gt = torch.sum(weights * predicted_depth * ground_truth_depth)
            weighted_dot_pred_pred = torch.sum(weights * predicted_depth ** 2)
            s = weighted_dot_pred_gt / weighted_dot_pred_pred
        s = s.clamp(min=0.001)
        s = s.detach()
        predicted_depth = s * predicted_depth
    else:
        scale_factor = torch.median(ground_truth_depth) / torch.median(predicted_depth)
        predicted_depth *= scale_factor
    if disp_input:
        ground_truth_depth = real_gt
        predicted_depth = depth2disparity(predicted_depth)
    if post_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=post_clip_min)
    if post_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=post_clip_max)
    if custom_mask is not None:
        assert custom_mask.shape == ground_truth_depth_original.shape
        mask_within_mask = custom_mask.cpu()[mask]
        predicted_depth = predicted_depth[mask_within_mask]
        ground_truth_depth = ground_truth_depth[mask_within_mask]
    abs_rel = torch.mean(torch.abs(predicted_depth - ground_truth_depth) / ground_truth_depth).item()
    sq_rel = torch.mean((predicted_depth - ground_truth_depth) ** 2 / ground_truth_depth).item()
    rmse = torch.sqrt(torch.mean((predicted_depth - ground_truth_depth) ** 2)).item()
    predicted_depth = torch.clamp(predicted_depth, min=1e-05)
    log_rmse = torch.sqrt(torch.mean((torch.log(predicted_depth) - torch.log(ground_truth_depth)) ** 2)).item()
    max_ratio = torch.maximum(predicted_depth / ground_truth_depth, ground_truth_depth / predicted_depth)
    threshold_0 = torch.mean((max_ratio < 1.0).float()).item()
    threshold_1 = torch.mean((max_ratio < 1.25).float()).item()
    threshold_2 = torch.mean((max_ratio < 1.25 ** 2).float()).item()
    threshold_3 = torch.mean((max_ratio < 1.25 ** 3).float()).item()
    if metric_scale:
        predicted_depth_original = predicted_depth_original
        if disp_input:
            predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = torch.abs(predicted_depth_original - ground_truth_depth_original) / ground_truth_depth_original
    elif align_with_lstsq or align_with_lad or align_with_lad2:
        predicted_depth_original = predicted_depth_original * s + t
        if disp_input:
            predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = torch.abs(predicted_depth_original - ground_truth_depth_original) / ground_truth_depth_original
    elif align_with_scale:
        predicted_depth_original = predicted_depth_original * s
        if disp_input:
            predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = torch.abs(predicted_depth_original - ground_truth_depth_original) / ground_truth_depth_original
    else:
        predicted_depth_original = predicted_depth_original * scale_factor
        if disp_input:
            predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = torch.abs(predicted_depth_original - ground_truth_depth_original) / ground_truth_depth_original
    depth_error_parity_map_full = torch.zeros_like(ground_truth_depth_original)
    depth_error_parity_map_full = torch.where(mask, depth_error_parity_map, depth_error_parity_map_full)
    predict_depth_map_full = predicted_depth_original
    gt_depth_map_full = torch.zeros_like(ground_truth_depth_original)
    gt_depth_map_full = torch.where(mask, ground_truth_depth_original, gt_depth_map_full)
    num_valid_pixels = torch.sum(mask).item() if custom_mask is None else torch.sum(mask_within_mask).item()
    if num_valid_pixels == 0:
        abs_rel, sq_rel, rmse, log_rmse, threshold_0, threshold_1, threshold_2, threshold_3 = (0, 0, 0, 0, 0, 0, 0, 0)
    results = {'Abs Rel': abs_rel, 'Sq Rel': sq_rel, 'RMSE': rmse, 'Log RMSE': log_rmse, 'δ < 1.': threshold_0, 'δ < 1.25': threshold_1, 'δ < 1.25^2': threshold_2, 'δ < 1.25^3': threshold_3, 'valid_pixels': num_valid_pixels}
    return (results, depth_error_parity_map_full, predict_depth_map_full, gt_depth_map_full)

def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)
    valid_mask = depthmap > 0.0
    return (X_cam, valid_mask)

def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, **kw):
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)
    X_world = X_cam
    if camera_pose is not None:
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]
        X_world = np.einsum('ik, vuk -> vui', R_cam2world, X_cam) + t_cam2world[None, None, :]
    return (X_world, valid_mask)

def load_depth_and_pts_for_eval(folder_or_list, size, intrinsics, camera_poses, square_ok=False, verbose=True, crop=True):
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = (folder_or_list, sorted(os.listdir(folder_or_list)))
    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = ('', folder_or_list)
    else:
        raise ValueError(f'bad folder_or_list={folder_or_list!r} ({type(folder_or_list)})')
    depth_and_pts = []
    for i, path in enumerate(folder_content):
        if not path.lower().endswith('.npy'):
            continue
        depthmap = np.load(path.replace('/rgb/', '/depth/').replace('.png', '.npy'))
        H1, W1 = depthmap.shape
        import cv2
        depthmap = cv2.resize(depthmap, (size, int(size / W1 * H1)), interpolation=cv2.INTER_NEAREST)
        H, W = depthmap.shape
        cx, cy = (W // 2, H // 2)
        halfw, halfh = (2 * cx // 16 * 8, 2 * cy // 16 * 8)
        if not square_ok and W == H:
            halfh = 3 * halfw / 4
        if crop:
            depthmap = depthmap[cy - halfh:cy + halfh, cx - halfw:cx + halfw]
        else:
            depthmap = cv2.resize(depthmap, (2 * halfw, 2 * halfh), interpolation=cv2.INTER_NEAREST)
        W2, H2 = depthmap.shape
        depthmap[~np.isfinite(depthmap)] = 0
        depthmap[depthmap > 20.0] = 0.0
        pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(depthmap, intrinsics[i], camera_poses[i])
        valid_mask = valid_mask & np.isfinite(pts3d).all(axis=-1)
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        depth_and_pts.append({'depthmap': depthmap.astype(np.float32)[None], 'pts3d': pts3d.astype(np.float32)[None], 'valid_mask': valid_mask[None]})
    assert depth_and_pts, 'no images foud at ' + root
    if verbose:
        print(f' (Found {len(depth_and_pts)} images)')
    return depth_and_pts
