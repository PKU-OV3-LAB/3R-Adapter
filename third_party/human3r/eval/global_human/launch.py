import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
import gc
sys.path.append('./src')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
import torch
import argparse
import shutil
from copy import deepcopy
from eval.global_human.metadata import dataset_metadata
from eval.global_human.utils import *
from accelerate import PartialState
from add_ckpt_path import add_path_to_dust3r
from collections import defaultdict, Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='path to the model weights', default='checkpoints/3RAdapter128-sm.o-l8t11-1.tran.1.shap-128.f-nm.f-h2.f2.r-r.img-depth15/2026.05.05-15:27:47/checkpoint-final.pth')
    parser.add_argument('--device', type=str, default='cuda', help='pytorch device')
    parser.add_argument('--output_dir', type=str, default='external/assets', help='value for outdir')
    parser.add_argument('--no_crop', type=bool, default=True, help='whether to crop input data')
    parser.add_argument('--eval_dataset', type=str, default='emdb1', choices=list(dataset_metadata.keys()))
    parser.add_argument('--crop_res', type=int, nargs=2, metavar=('W', 'H'), default=None)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--revisit', type=int, default=1)
    parser.add_argument('--freeze_state', action='store_true', default=False)
    parser.add_argument('--solve_pose', action='store_true', default=False)
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--is_naive', action='store_true', default=False)
    parser.add_argument('--use_ttt3r', action='store_true', default=True)
    parser.add_argument('--reset_interval', type=int, default=100)
    parser.add_argument('--use_fake_K', action='store_true', default=False)
    parser.add_argument('--eval_depth', action='store_true', default=False)
    parser.add_argument('--eval_reconstruction', action='store_true', default=False)
    return parser

def get_seq_list(metadata, img_path):
    get_seq_func = metadata.get('get_seq_func', None)
    split = metadata.get('split', '')
    if get_seq_func:
        annots = metadata['get_annot_func'](img_path, split)
        seq_list, seq_to_images = get_seq_func(img_path, split, annots)
        return (seq_list, seq_to_images, annots)
    if metadata.get('full_seq', False):
        seq_dir = f'{img_path}/{split}'
        seq_list = [seq for seq in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, seq))]
        return (sorted(seq_list), None, None)
    else:
        return (sorted(metadata.get('seq_list', [])), None, None)

def get_file_list(metadata, img_path, seq, seq_to_images=None):
    dir_path = metadata['dir_path_func'](img_path, seq)
    subsample = metadata.get('subsample', 1)
    max_frames = metadata.get('max_frames', None)
    if seq_to_images is not None:
        filelist = [os.path.join(dir_path, name) for name in seq_to_images[seq]]
    else:
        filelist = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
    filelist.sort()
    if max_frames is not None:
        filelist = filelist[:max_frames]
    sampled_indices = list(range(0, len(filelist), subsample))
    filelist = filelist[::subsample]
    return (filelist, sampled_indices)

def run_inference(device, args, metadata, filelist, sampled_indices, annots, model, smpl_model):
    from dust3r.inference import inference_recurrent_lighter
    get_view_func = metadata.get('get_view_func', None)
    mask_path_func = metadata.get('mask_path_func', None)
    mask_path_list = mask_path_func(filelist) if mask_path_func is not None else []
    views = prepare_input(filelist, [True for _ in filelist], msk_paths=mask_path_list, size=args.size, crop=not args.no_crop, revisit=args.revisit, update=not args.freeze_state, load_func=get_view_func, annots=annots, sampled_indices=sampled_indices, reset_interval=args.reset_interval, crop_res=args.crop_res, args=args)
    with torch.no_grad():
        smpl_model.update_smpl_gt_eval(views, args.eval_dataset)
    gt = prepare_gt(views, args=args)
    keep_keys = set(['img', 'img_mask', 'true_shape', 'img_mhmr', 'reset', 'update', 'K_mhmr', 'camera_intrinsics', 'camera_pose', 'msk'])
    for i, view in enumerate(views):
        views[i] = {key: view[key] for key in keep_keys if key in view}
    outputs, _ = inference_recurrent_lighter(views, model, device, verbose=False, is_naive=args.is_naive, use_ttt3r=args.use_ttt3r)
    pred = prepare_output(outputs, revisit=args.revisit, solve_pose=args.solve_pose, is_save=args.save, args=args)
    del outputs, views
    gc.collect()
    torch.cuda.empty_cache()
    return (gt, pred)

def get_pred_smpl(pred, gt, f_id, is_naive, smpl_layer, mhmr_img_res, K_to_proj):
    n_humans_i = pred['shape'][f_id].shape[0]
    expand = lambda x: x.expand(n_humans_i, -1, -1)
    with torch.no_grad():
        if is_naive:
            dist = pred['transl'][f_id][:, 0].unsqueeze(-1)
            dist = to_euclidean_dist(mhmr_img_res, dist, expand(gt['K_mhmr'][f_id]))
            smpl_out = smpl_layer(pred['rotvec'][f_id], pred['shape'][f_id], None, pred['loc'][f_id], dist, K=expand(gt['K_mhmr'][f_id]), expression=pred['expression'][f_id], K_to_proj=expand(gt['K'][f_id]))
            pred['transl'][f_id] = smpl_out['smpl_transl']
        else:
            smpl_out = smpl_layer(pred['rotvec'][f_id], pred['shape'][f_id], pred['transl'][f_id], None, None, K=expand(K_to_proj[f_id]), expression=pred['expression'][f_id])
    return smpl_out['smpl_v3d']

def match_2d(pr_j2d, gt_j2d):
    gt_j2d = gt_j2d.numpy()
    bestMatch, falsePositives, misses = match_2d_greedy(pr_j2d.numpy()[:, :gt_j2d.shape[1]], gt_j2d, np.ones_like(gt_j2d[..., 0]).astype(np.bool_))
    update = {'count': len(gt_j2d), 'miss': len(misses), 'fp': len(falsePositives)}
    return (bestMatch, update)

def eval_smpl_error(args, model, smpl_model, smpl_layer, save_dir=None):
    from dust3r.utils.geometry import perspective_projection, geotrf
    metadata = dataset_metadata.get(args.eval_dataset)
    img_path = metadata['img_path']
    mhmr_img_res = getattr(model, 'mhmr_img_res', None)
    subsample = metadata.get('subsample', 1)
    is_global = metadata['is_global'](metadata.get('split', ''))
    pelvis_idx = smpl_model.pelvis_idx
    seq_list, seq_to_images, annots = get_seq_list(metadata, img_path)
    if save_dir is None:
        save_dir = args.output_dir
    distributed_state = PartialState()
    model.to(distributed_state.device)
    device = distributed_state.device
    if args.eval_depth:
        gathered_depth_metrics = []
    if args.eval_reconstruction:
        from eval.global_human.criterion import Regr3D_t_ScaleShiftInv, L21
        from eval.global_human.criterion import accuracy, completion
        from eval.global_human.criterion import transpose_dict_list
        from dust3r.utils.geometry import geotrf
        from copy import deepcopy
        criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)
        gathered_reconstruction_metrics = []
    with distributed_state.split_between_processes(seq_list) as seqs:
        if len(seq_list) < distributed_state.num_processes:
            if distributed_state.process_index >= len(seq_list):
                seqs = []
        error_log_path = f'{save_dir}/_error_log_{distributed_state.process_index}.txt'
        for seq_idx, seq in enumerate(tqdm(seqs)):
            print(f'Evaluating sequence: {seq}')
            filelist, sampled_indices = get_file_list(metadata, img_path, seq, seq_to_images)
            gt, pred = run_inference(device, args, metadata, filelist, sampled_indices, annots, model, smpl_model)
            K_to_proj = gt['K']
            T_c2w = pred['T_c2w']
            global_batch = []
            metrics = defaultdict(list)
            counter = Counter()
            for f_id in range(len(filelist)):
                n_humans_i = pred['shape'][f_id].shape[0]
                if n_humans_i > 0:
                    pred_v3d_c = get_pred_smpl(pred, gt, f_id, args.is_naive, smpl_layer, mhmr_img_res, K_to_proj)
                    pred_v3d_c = smpl_model.smplx2smpl @ pred_v3d_c
                    pred_j3d_c = smpl_model.j_regressor @ pred_v3d_c
                    pr_j2d = perspective_projection(pred_j3d_c, K_to_proj[f_id].expand(n_humans_i, -1, -1))
                else:
                    pred_v3d_c = torch.empty(0, 6890, 3, dtype=torch.float32)
                    pr_j2d = torch.empty(0, 24, 2, dtype=torch.float32)
                bestMatch, update = match_2d(pr_j2d, gt['j2d'][f_id])
                counter.update(update)
                if len(bestMatch) > 0:
                    counter.update({'n_human': len(bestMatch)})
                    pid, gid = (bestMatch[:, 0], bestMatch[:, 1])
                    camcoord_batch = {'pred_j3d': pred_j3d_c[pid], 'target_j3d': gt['j3d_c'][f_id][gid], 'pred_v3d': pred_v3d_c[pid], 'target_v3d': gt['v3d_c'][f_id][gid]}
                    camcoord_metrics = eval_camcoord(camcoord_batch, pelvis_idx)
                    for k, v in camcoord_metrics.items():
                        metrics[k].append(v)
                    if is_global:
                        expand = lambda x: x.expand(len(bestMatch), -1, -1)
                        global_batch.append({'pred_j3d': geotrf(expand(T_c2w[f_id]), pred_j3d_c[pid]), 'target_j3d': gt['j3d_w'][f_id][gid], 'pred_v3d': geotrf(expand(T_c2w[f_id]), pred_v3d_c[pid]), 'target_v3d': gt['v3d_w'][f_id][gid]})
                if args.vis:
                    visualize3d(save_dir=f'{save_dir}/visualization/{seq}', img_path=filelist[f_id], view=gt['img'][f_id], gt_v3d_c=gt['v3d_c'][f_id], pred_v3d_c=pred_v3d_c, K_to_proj=K_to_proj[f_id], gt_K=gt['K'][f_id], bestMatch=bestMatch, smpl_face=smpl_model.smpl_faces['smpl'], pred_c2w=pred['T_c2w'][f_id], gt_c2w=gt['T_c2w'][f_id], pred_pts3d=pred['pts3d_self'][f_id] if 'pts3d_self' in pred else None, gt_depthmap=gt['depthmap'][f_id] if 'depthmap' in gt else None, msk=pred['msk'][f_id] if 'msk' in pred else None, conf=pred['conf_self'][f_id] if 'conf_self' in pred else None)
            metrics['precision'], metrics['recall'], metrics['f1_score'] = compute_prf1(counter['count'], counter['miss'], counter['fp'])
            if args.eval_depth:
                pred_depth = torch.cat(pred['pts3d_self'], 0)[..., -1]
                gt_depth = gt['depthmap']
                depth_results, error_map, depth_predict, depth_gt = depth_evaluation(pred_depth, gt_depth, max_depth=10, align_with_lad2=True, use_gpu=True, post_clip_max=70)
                gathered_depth_metrics.append(depth_results)
            if args.eval_reconstruction:
                reconstruction_keep_keys = ['camera_pose', 'pts3d', 'valid_mask', 'pts3d_in_self_view', 'pts3d_in_other_view', 'img', 'conf', 'msk']
                gt['camera_pose'], pred['camera_pose'] = (gt['T_c2w'], pred['T_c2w'])
                valid_length = pred['pts3d_in_self_view'].shape[0]
                gts = transpose_dict_list(gt, valid_length, keep_keys=reconstruction_keep_keys, squeeze=False)
                preds = transpose_dict_list(pred, valid_length, keep_keys=reconstruction_keep_keys, squeeze=False)
                gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = criterion.get_all_pts3d_t(gts, preds, use_pose=False)
                pred_scale, gt_scale, pred_shift_z, gt_shift_z = (monitoring['pred_scale'], monitoring['gt_scale'], monitoring['pred_shift_z'], monitoring['gt_shift_z'])
                in_camera1 = None
                pts_all = []
                pts_gt_all = []
                images_all = []
                masks_all = []
                conf_all = []
                for j, view in enumerate(gts):
                    if in_camera1 is None:
                        in_camera1 = view['camera_pose'][0].cpu()
                    image = view['img'].permute(0, 2, 3, 1).cpu().numpy()[0]
                    mask = view['valid_mask'].cpu().numpy()[0]
                    pts = pred_pts[j].cpu().numpy()[0]
                    conf = preds[j]['conf'].cpu().data.numpy()[0]
                    pts_gt = gt_pts[j].detach().cpu().numpy()[0]
                    pts[..., -1] += gt_shift_z.cpu().numpy().item()
                    pts = geotrf(in_camera1, pts)
                    pts_gt[..., -1] += gt_shift_z.cpu().numpy().item()
                    pts_gt = geotrf(in_camera1, pts_gt)
                    images_all.append((image[None, ...] + 1.0) / 2.0)
                    pts_all.append(pts[None, ...])
                    pts_gt_all.append(pts_gt[None, ...])
                    masks_all.append(mask[None, ...])
                    conf_all.append(conf[None, ...])
                images_all = np.concatenate(images_all, axis=0)
                pts_all = np.concatenate(pts_all, axis=0)
                pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                masks_all = np.concatenate(masks_all, axis=0)
                threshold = 0.1
                pts_all_masked = pts_all[masks_all > 0]
                pts_gt_all_masked = pts_gt_all[masks_all > 0]
                images_all_masked = images_all[masks_all > 0]
                import open3d as o3d
                pcd = o3d.geometry.PointCloud()
                pcd_dir = f'{save_dir}/visualization/{seq}'
                os.makedirs(f'{save_dir}/visualization', exist_ok=True)
                pcd.points = o3d.utility.Vector3dVector(pts_all_masked.reshape(-1, 3))
                pcd.colors = o3d.utility.Vector3dVector(images_all_masked.reshape(-1, 3))
                o3d.io.write_point_cloud(pcd_dir + '-pred.ply', pcd)
                pcd_gt = o3d.geometry.PointCloud()
                pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_all_masked.reshape(-1, 3))
                pcd_gt.colors = o3d.utility.Vector3dVector(images_all_masked.reshape(-1, 3))
                o3d.io.write_point_cloud(pcd_dir + '-gt.ply', pcd_gt)
                trans_init = np.eye(4)
                reg_p2p = o3d.pipelines.registration.registration_icp(pcd, pcd_gt, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
                transformation = reg_p2p.transformation
                pcd = pcd.transform(transformation)
                pcd.estimate_normals()
                pcd_gt.estimate_normals()
                gt_normal = np.asarray(pcd_gt.normals)
                pred_normal = np.asarray(pcd.normals)
                acc, acc_med, nc1, nc1_med = accuracy(pcd_gt.points, pcd.points, gt_normal, pred_normal)
                comp, comp_med, nc2, nc2_med = completion(pcd_gt.points, pcd.points, gt_normal, pred_normal)
                log_string = f'Idx: {seq_idx}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}'
                print(log_string, file=open(f'{save_dir}/visualization/0_all.txt', 'a'))
                print(log_string)
                gathered_reconstruction_metrics.append({'acc': acc, 'acc_med': acc_med, 'comp': comp, 'comp_med': comp_med, 'nc': (nc1 + nc2) / 2, 'nc1': nc1, 'nc2': nc2})
            if is_global:
                global_batch = {k: torch.cat([b[k] for b in global_batch]) for k in global_batch[0]}
                global_metrics = eval_global(global_batch, subsample)
                for k, v in global_metrics.items():
                    metrics[k].append(v)
            torch.cuda.empty_cache()
            os.makedirs(save_dir, exist_ok=True)
            write_log(error_log_path, args.eval_dataset, seq, counter, metrics)
            del gt, pred, filelist, sampled_indices
            del global_batch, metrics, counter
            if 'global_metrics' in locals():
                del global_metrics
            gc.collect()
    distributed_state.wait_for_everyone()
    torch.cuda.empty_cache()
    average_depth_metrics = None
    average_reconstruction_metrics = None
    if args.eval_depth:
        average_depth_metrics = {key: np.average([metric[key] for metric in gathered_depth_metrics], weights=[metric['valid_pixels'] for metric in gathered_depth_metrics]) for key in gathered_depth_metrics[0].keys() if key != 'valid_pixels'}
    if args.eval_reconstruction:
        average_reconstruction_metrics = {key: np.average([metric[key] for metric in gathered_reconstruction_metrics]) for key in gathered_reconstruction_metrics[0].keys()}
    results = process_directory(save_dir)
    summary = calculate_averages(results)
    if distributed_state.is_main_process:
        with open(f'{save_dir}/_error_log.txt', 'a') as f:
            for i in range(distributed_state.num_processes):
                if not os.path.exists(f'{save_dir}/_error_log_{i}.txt'):
                    break
                with open(f'{save_dir}/_error_log_{i}.txt', 'r') as f_sub:
                    f.write(f_sub.read())
            log = get_summary_log(summary, depth_metrics=average_depth_metrics, reconstruction_metrics=average_reconstruction_metrics)
            f.write(log)
        print(log.strip())
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    add_path_to_dust3r(args.weights)
    from dust3r.utils.image import load_images_for_eval as load_images
    from dust3r.utils.image import load_masks_for_eval as load_masks
    from dust3r.post_process import estimate_focal_knowing_depth
    from dust3r.model import ARCroco3DStereo
    from dust3r.utils.camera import pose_encoding_to_camera
    from dust3r.utils.geometry import weighted_procrustes, to_euclidean_dist, matrix_cumprod
    from dust3r.smpl_model import SMPLModel
    from dust3r.utils import SMPL_Layer
    from dust3r.utils.image import unpad_image
    args.no_crop = False

    def recover_cam_params(pts3ds_self, pts3ds_other, conf_self, conf_other):
        B, H, W, _ = pts3ds_self.shape
        pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1).reshape(B, 1, 2)
        focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode='weiszfeld')
        pts3ds_self = pts3ds_self.reshape(B, -1, 3)
        pts3ds_other = pts3ds_other.reshape(B, -1, 3)
        conf_self = conf_self.reshape(B, -1)
        conf_other = conf_other.reshape(B, -1)
        c2w = weighted_procrustes(pts3ds_self, pts3ds_other, torch.log(conf_self) * torch.log(conf_other), use_weights=True, return_T=True)
        return (c2w, focal, pp.reshape(B, 2))

    def _crop_resize(image, intrinsics, crop_res):
        import dust3r.datasets.utils.cropping as cropping
        from dust3r.utils.image import ImgNorm
        img_device = image.device if isinstance(image, torch.Tensor) else None
        had_batch_dim = False
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                assert image.shape[0] == 1, '_crop_resize expects a single image; got a batch'
                image = image.squeeze(0)
                had_batch_dim = True
            elif image.dim() != 3:
                raise RuntimeError(f'Unexpected image tensor shape {tuple(image.shape)}; expected CHW or 1xCHW')
            image_np = ((image.detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        else:
            image_np = image
        target_resolution = np.array(crop_res)
        image_pil, _, intrinsics = cropping.rescale_image_depthmap(image_np, None, intrinsics, target_resolution)
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image_pil.size, crop_res, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, crop_res)
        image_pil, _, intrinsics2 = cropping.crop_image_depthmap(image_pil, None, intrinsics, crop_bbox)
        image_arr = np.array(image_pil)
        if image_arr.ndim == 2:
            image_arr = np.repeat(image_arr[..., None], 3, axis=2)
        image_tensor = ImgNorm(image_pil)
        if had_batch_dim:
            image_tensor = image_tensor.unsqueeze(0)
        if img_device is not None:
            image_tensor = image_tensor.to(img_device)
        intrinsics_tensor = torch.from_numpy(intrinsics2).unsqueeze(0)
        return (image_tensor, intrinsics_tensor)

    def prepare_input(img_paths, img_mask, msk_paths, size, raymaps=None, raymap_mask=None, revisit=1, update=True, crop=True, load_func=None, annots=None, sampled_indices=None, reset_interval=100, crop_res=None, args=None):
        images = load_images(img_paths, size=size, verbose=False, crop=crop)
        images = load_func((img_paths, images, annots, sampled_indices))
        has_msk = len(msk_paths) > 0
        if has_msk:
            msks = load_masks(msk_paths, size=size, verbose=False, crop=crop)
        if args.eval_depth or args.eval_reconstruction:
            from eval.global_human.utils import load_depth_and_pts_for_eval
            depth_paths = [img_path.replace('/rgb/', '/depth/').replace('.png', '.npy') for img_path in img_paths]
            intrinsics = [image['intrinsics'] for image in images]
            camera_poses = [image['camera_pose'] for image in images]
            depth_and_pts = load_depth_and_pts_for_eval(depth_paths, size=size, intrinsics=intrinsics, camera_poses=camera_poses, verbose=False, crop=crop)
        views = []
        if raymaps is None and raymap_mask is None:
            num_views = len(images)
            for i in range(num_views):
                view = {'img': images[i]['img'], 'ray_map': torch.full((images[i]['img'].shape[0], 6, images[i]['img'].shape[-2], images[i]['img'].shape[-1]), torch.nan), 'true_shape': torch.from_numpy(images[i]['true_shape']), 'idx': i, 'instance': str(i), 'camera_pose': torch.from_numpy(images[i]['camera_pose']).unsqueeze(0), 'camera_intrinsics': torch.from_numpy(images[i]['intrinsics']).unsqueeze(0), 'img_mask': torch.tensor(True).unsqueeze(0), 'ray_mask': torch.tensor(False).unsqueeze(0), 'update': torch.tensor(True).unsqueeze(0), 'reset': torch.tensor((i + 1) % reset_interval == 0).unsqueeze(0)}
                if args.eval_depth:
                    view.update({'depthmap': torch.from_numpy(depth_and_pts[i]['depthmap'])})
                if args.eval_reconstruction:
                    view.update({'pts3d': torch.from_numpy(depth_and_pts[i]['pts3d']), 'valid_mask': torch.from_numpy(depth_and_pts[i]['valid_mask'])})
                if crop_res is not None:
                    view['img'], view['camera_intrinsics'] = _crop_resize(images[i]['img'], images[i]['intrinsics'], crop_res)
                    view['true_shape'] = torch.tensor([view['img'].shape[-2:]], dtype=torch.int32)
                if has_msk:
                    view['msk'] = msks[i]
                for key in images[i].keys():
                    if key.startswith(('smpl', 'T_w2c')):
                        view[key] = torch.tensor(images[i][key]).unsqueeze(0)
                views.append(view)
                if (i + 1) % reset_interval == 0:
                    overlap_view = deepcopy(view)
                    overlap_view['reset'] = torch.tensor(False).unsqueeze(0)
                    views.append(overlap_view)
        else:
            num_views = len(images) + len(raymaps)
            assert len(img_mask) == len(raymap_mask) == num_views
            assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)
            j = 0
            k = 0
            for i in range(num_views):
                view = {'img': images[j]['img'] if img_mask[i] else torch.full_like(images[0]['img'], torch.nan), 'ray_map': raymaps[k] if raymap_mask[i] else torch.full_like(raymaps[0], torch.nan), 'true_shape': torch.from_numpy(images[j]['true_shape']) if img_mask[i] else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]])), 'idx': i, 'instance': str(i), 'camera_pose': torch.from_numpy(images[i]['camera_pose']).unsqueeze(0), 'camera_intrinsics': torch.from_numpy(images[i]['intrinsics']).unsqueeze(0), 'img_mask': torch.tensor(img_mask[i]).unsqueeze(0), 'ray_mask': torch.tensor(raymap_mask[i]).unsqueeze(0), 'update': torch.tensor(img_mask[i]).unsqueeze(0), 'reset': torch.tensor((i + 1) % reset_interval == 0).unsqueeze(0)}
                if args.eval_depth:
                    view.update({'depthmap': torch.from_numpy(depth_and_pts[i]['depthmap'])})
                if args.eval_reconstruction:
                    view.update({'pts3d': torch.from_numpy(depth_and_pts[i]['pts3d']), 'valid_mask': torch.from_numpy(depth_and_pts[i]['valid_mask'])})
                if crop_res is not None:
                    view['img'], view['camera_intrinsics'] = _crop_resize(images[i]['img'], images[i]['intrinsics'], crop_res)
                    view['true_shape'] = torch.tensor([view['img'].shape[-2:]], dtype=torch.int32)
                if has_msk:
                    view['msk'] = msks[j] if img_mask[i] else torch.full_like(images[0]['img'], torch.nan)
                for key in images[i].keys():
                    if key.startswith(('smpl', 'T_w2c')):
                        view[key] = torch.tensor(images[i][key]).unsqueeze(0)
                if img_mask[i]:
                    j += 1
                if raymap_mask[i]:
                    k += 1
                views.append(view)
                if (i + 1) % reset_interval == 0:
                    overlap_view = deepcopy(view)
                    overlap_view['reset'] = torch.tensor(False).unsqueeze(0)
                    views.append(overlap_view)
            assert j == len(images) and k == len(raymaps)
        if revisit > 1:
            new_views = []
            for r in range(revisit):
                for i in range(len(views)):
                    new_view = deepcopy(views[i])
                    new_view['idx'] = r * len(views) + i
                    new_view['instance'] = str(r * len(views) + i)
                    if r > 0:
                        if not update:
                            new_view['update'] = torch.tensor(False).unsqueeze(0)
                    new_views.append(new_view)
            return new_views
        del images
        return views

    def prepare_gt(gts, revisit=1, args=None):
        target_out = defaultdict(list)
        valid_length = len(gts) // revisit
        gts = gts[-valid_length:]
        gts = [gt for gt in gts if not gt['reset']]
        intrinsics = [gt['camera_intrinsics'] for gt in gts]
        K_mhmr = [gt['K_mhmr'] for gt in gts]
        camera_pose = [gt['camera_pose'] for gt in gts]
        imgs = [gt['img'] for gt in gts]
        if args.eval_depth:
            depthmaps = [gt['depthmap'] for gt in gts]
        if args.eval_reconstruction:
            pts3ds = [gt['pts3d'] for gt in gts]
            valid_masks = [gt['valid_mask'] for gt in gts]
            msks = [gt['msk'] for gt in gts]
        target_out['K'] = torch.cat(intrinsics, 0)
        target_out['K_mhmr'] = torch.cat(K_mhmr, 0)
        target_out['T_c2w'] = torch.cat(camera_pose, 0)
        target_out['img'] = torch.cat(imgs, 0)
        if args.eval_depth:
            target_out['depthmap'] = torch.cat(depthmaps, 0)
        if args.eval_reconstruction:
            target_out['pts3d'] = torch.cat(pts3ds, 0)
            target_out['valid_mask'] = torch.cat(valid_masks, 0)
            target_out['msk'] = torch.cat(msks, 0)
        if 'T_w2c' in gts[0]:
            T_w2c_list = [gt['T_w2c'] for gt in gts]
            target_out['T_w2c'] = torch.cat(T_w2c_list, 0)
        smpl_mask_list = [gt['smpl_mask'] for gt in gts]
        for gt, smpl_mask in zip(gts, smpl_mask_list):
            target_out['v3d_c'].append(gt['smpl_v3d_c'][smpl_mask])
            target_out['j3d_c'].append(gt['smpl_j3d_c'][smpl_mask])
            target_out['v3d_w'].append(gt['smpl_v3d_w'][smpl_mask])
            target_out['j3d_w'].append(gt['smpl_j3d_w'][smpl_mask])
            target_out['v2d'].append(gt['smpl_v2d'][smpl_mask])
            target_out['j2d'].append(gt['smpl_j2d'][smpl_mask])
        del gts
        return target_out

    def prepare_output(outputs, revisit=1, solve_pose=False, is_save=False, args=None):
        pred_out = defaultdict(list)
        valid_length = len(outputs['pred']) // revisit
        outputs['pred'] = outputs['pred'][-valid_length:]
        outputs['views'] = outputs['views'][-valid_length:]
        reset_mask = torch.cat([view['reset'] for view in outputs['views']], 0)
        shifted_reset_mask = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0)
        outputs['pred'] = [pred for pred, mask in zip(outputs['pred'], shifted_reset_mask) if not mask]
        reset_mask = reset_mask[~shifted_reset_mask]
        if solve_pose:
            pts3ds_self_to_save = [output['pts3d_in_self_view'] for output in outputs['pred']]
            pts3ds_other = [output['pts3d_in_other_view'] for output in outputs['pred']]
            conf_self = [output['conf_self'] for output in outputs['pred']]
            conf_other = [output['conf'] for output in outputs['pred']]
            pr_poses, focal, pp = recover_cam_params(torch.cat(pts3ds_self_to_save, 0), torch.cat(pts3ds_other, 0), torch.cat(conf_self, 0), torch.cat(conf_other, 0))
            pts3ds_self = torch.cat(pts3ds_self_to_save, 0)
        else:
            pts3ds_self_to_save = [output['pts3d_in_self_view'] for output in outputs['pred']]
            pts3ds_other = [output['pts3d_in_other_view'] for output in outputs['pred']]
            conf_self = [output['conf_self'] for output in outputs['pred']]
            conf_other = [output['conf'] for output in outputs['pred']]
            pts3ds_self = torch.cat(pts3ds_self_to_save, 0)
            pr_poses = [pose_encoding_to_camera(pred['camera_pose'].clone()) for pred in outputs['pred']]
            pr_poses = torch.cat(pr_poses, 0)
            B, H, W, _ = pts3ds_self.shape
            pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1).reshape(B, 2)
            focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode='weiszfeld')
        has_mask = 'msk' in outputs['pred'][0]
        if has_mask:
            msks = [output['msk'][..., 0] for output in outputs['pred']]
            msks = [unpad_image(m, [H, W]) for m in msks]
        else:
            msks = [torch.zeros(1, H, W) for _ in range(B)]
        pred_out['pts3d_self'] = pts3ds_self_to_save
        pred_out['conf_self'] = conf_self
        pred_out['msk'] = msks
        if args.eval_reconstruction:
            pred_out['pts3d_in_self_view'] = torch.cat([output['pts3d_in_self_view'] for output in outputs['pred']], dim=0)
            pred_out['pts3d_in_other_view'] = torch.cat([output['pts3d_in_other_view'] for output in outputs['pred']], dim=0)
            pred_out['conf'] = torch.cat(conf_self, dim=0)
        if reset_mask.any():
            identity = torch.eye(4, device=pr_poses.device)
            reset_poses = torch.where(reset_mask.unsqueeze(-1).unsqueeze(-1), pr_poses, identity)
            cumulative_bases = matrix_cumprod(reset_poses)
            shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
            pr_poses = torch.einsum('bij,bjk->bik', shifted_bases, pr_poses)
        pred_out['T_c2w'] = pr_poses
        intrinsics = torch.eye(3, device=pp.device).unsqueeze(0).repeat(B, 1, 1)
        intrinsics[:, 0, 0] = focal
        intrinsics[:, 1, 1] = focal
        intrinsics[:, [0, 1], 2] = pp
        pred_out['K'] = intrinsics
        pred_out['shape'] = [output.get('smpl_shape', torch.empty(1, 0, 10))[0] for output in outputs['pred']]
        pred_out['rotvec'] = [roma.rotmat_to_rotvec(output.get('smpl_rotmat', torch.empty(1, 0, 53, 3, 3))[0]) for output in outputs['pred']]
        pred_out['transl'] = [output.get('smpl_transl', torch.empty(1, 0, 3))[0] for output in outputs['pred']]
        pred_out['expression'] = [output.get('smpl_expression', [None])[0] for output in outputs['pred']]
        pred_out['loc'] = [output.get('smpl_loc', torch.empty(1, 0, 2))[0] for output in outputs['pred']]
        del outputs
        return pred_out
    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copy(__file__, args.output_dir)
    model = ARCroco3DStereo.from_pretrained(args.weights)
    smpl_model = SMPLModel('cpu', model_args={'patch_size': model.croco_args['patch_size'], 'mhmr_img_res': model.mhmr_img_res, 'bb_patch_size': model.bb_patch_size}, eval_args={'dataset': args.eval_dataset, 'use_fake_K': args.use_fake_K})
    smpl_layer = SMPL_Layer(type='smplx', gender='neutral', num_betas=10, kid=False, person_center='head')
    eval_smpl_error(args, model, smpl_model, smpl_layer, save_dir=args.output_dir)
