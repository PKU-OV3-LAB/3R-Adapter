import numpy as np
from scipy.spatial import cKDTree as KDTree
import torch


def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float32))
    return comp_ratio


def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(gt_points)
    distances, idx = gt_points_kd_tree.query(rec_points, workers=-1)
    acc = np.mean(distances)

    acc_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals[idx] * rec_normals, axis=-1)
        normal_dot = np.abs(normal_dot)

        return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)

    return acc, acc_median


def completion(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(rec_points)
    distances, idx = gt_points_kd_tree.query(gt_points, workers=-1)
    comp = np.mean(distances)
    comp_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals * rec_normals[idx], axis=-1)
        normal_dot = np.abs(normal_dot)

        return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)

    return comp, comp_median


def compute_iou(pred_vox, target_vox):
    # Get voxel indices
    v_pred_indices = [voxel.grid_index for voxel in pred_vox.get_voxels()]
    v_target_indices = [voxel.grid_index for voxel in target_vox.get_voxels()]

    # Convert to sets for set operations
    v_pred_filled = set(tuple(np.round(x, 4)) for x in v_pred_indices)
    v_target_filled = set(tuple(np.round(x, 4)) for x in v_target_indices)

    # Compute intersection and union
    intersection = v_pred_filled & v_target_filled
    union = v_pred_filled | v_target_filled

    # Compute IoU
    iou = len(intersection) / len(union)
    return iou

def summary_parameters(model, detail=False):
    if detail:
        print('>> Trainable Parameters:')
        trainable_paramters = [(str(n), str(v.dtype), str(tuple(v.shape)), str(v.numel())) for n, v in model.named_parameters() if v.requires_grad and not getattr(v, '_is_frozen', False)]    
        max_lens = [max([len(item) + 4 for item in col]) for col in zip(*trainable_paramters)]
        raw_format = '|' + '|'.join(['{{:{}s}}'.format(max_len) for max_len in max_lens]) + '|'
        raw_split = '-' * (sum(max_lens) + len(max_lens) + 1)
        print(raw_split)
        print(raw_format.format('Name', 'Dtype', 'Shape', '#Params'))
        print(raw_split)
        for name, dtype, shape, number in trainable_paramters:
            print(raw_format.format(name, dtype, shape, number))
            print(raw_split)

    num_trainable_params = sum([v.numel() for v in model.parameters() if v.requires_grad])
    num_trainable_fixed_params = sum([v.numel() for v in model.parameters() if v.requires_grad and getattr(v, '_is_frozen', False)])
    total_params = sum([v.numel() for v in model.parameters()])
    non_trainable_params = total_params - num_trainable_params
    print('>> {:25s}\t{:.2f}\tM  {:.2f}\tK'.format('# TrainingParams:', (num_trainable_params-num_trainable_fixed_params) / (1e6), (num_trainable_params-num_trainable_fixed_params) / (1e3)))
    print('>> {:25s}\t{:.2f}\tM  {:.2f}\tK'.format('# TrainableParams:', num_trainable_params / (1e6), num_trainable_params / (1e3)))
    print('>> {:25s}\t{:.2f}\tM'.format('# NonTrainableParams:', non_trainable_params / (1e6)))
    print('>> {:25s}\t{:.2f}\tM'.format('# TotalParams:', total_params / (1e6)))
    print('>> {:25s}\t{:.2f}\t%'.format('# TuningRatio:', (num_trainable_params-num_trainable_fixed_params) / total_params * 100.))
    print('\n')