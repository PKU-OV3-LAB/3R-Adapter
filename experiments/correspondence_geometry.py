"""Fixed Oxford geometry and crop protocol used for Camera Ready Table 4.

Projection and discretization intentionally preserve the measurement convention.
They are independent of model predictions and learned offsets.
"""
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def data_path(root, relative):
    root = Path(root).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"frame path must be dataset-relative: {relative}")
    return path


def load_view(ds, data_root, frame, resolution, rng):
    with Image.open(data_path(data_root, frame["img_path"])) as image:
        rgb = np.array(image.convert("RGB"))
    depth = np.load(data_path(data_root, frame["depth_path"])).astype(np.float32) / 1000.0
    depth[~np.isfinite(depth)] = 0.0
    depth[depth > 200.0] = 0.0
    intr = np.array(frame["intrinsic"], dtype=np.float32)
    ext = np.array(frame["extrinsic"], dtype=np.float32)
    pose = np.linalg.inv(ext).astype(np.float32)
    image, depth, intr = ds._crop_resize_if_necessary(
        rgb, depth, intr, resolution, rng=rng, info=0)
    return dict(img=np.array(image), depthmap=depth.astype(np.float32),
                camera_pose=pose, camera_intrinsics=intr.astype(np.float32))


def build_pairwise_gt_correspondences(views, patch_size=14, device="cpu"):
    from pi3.utils.geometry import se3_inverse, warp_kpts
    n = len(views)
    h, w = views[0]["depthmap"].shape[:2]
    ph, pw = h // patch_size, w // patch_size
    hw = ph * pw
    cc, rr = np.meshgrid(np.arange(pw), np.arange(ph))
    cx = (cc * patch_size + patch_size // 2).flatten().astype(np.float32)
    cy = (rr * patch_size + patch_size // 2).flatten().astype(np.float32)
    kx = (2 * cx / (w - 1) - 1).astype(np.float32)
    ky = (2 * cy / (h - 1) - 1).astype(np.float32)
    kpts = torch.tensor(np.stack([kx, ky], -1), device=device)[None]
    tokens = -np.ones((n, n, hw), dtype=np.int64)
    pixels = np.zeros((n, n, hw, 2), dtype=np.float32)
    valid = np.zeros((n, n, hw), dtype=bool)
    for i in range(n):
        depth0 = torch.tensor(views[i]["depthmap"], device=device, dtype=torch.float32)[None]
        k0 = torch.tensor(views[i]["camera_intrinsics"], device=device, dtype=torch.float32)[None]
        pose_i = torch.tensor(views[i]["camera_pose"], device=device, dtype=torch.float32)
        qvalid = np.array([views[i]["depthmap"][int(cy[t]), int(cx[t])] > 0
                          for t in range(hw)], dtype=bool)
        for j in range(n):
            if i == j:
                continue
            depth1 = torch.tensor(views[j]["depthmap"], device=device, dtype=torch.float32)[None]
            k1 = torch.tensor(views[j]["camera_intrinsics"], device=device, dtype=torch.float32)[None]
            pose_j = torch.tensor(views[j]["camera_pose"], device=device, dtype=torch.float32)
            transform = se3_inverse(pose_j[None]) @ pose_i[None]
            with torch.no_grad():
                mask, warped = warp_kpts(kpts, depth0, depth1, transform, k0, k1)
            mask = mask[0].cpu().numpy().astype(bool)
            warped = warped[0].cpu().numpy()
            wx = (warped[:, 0] + 1) / 2 * (w - 1)
            wy = (warped[:, 1] + 1) / 2 * (h - 1)
            inside = (wx > 0) & (wx < w - 1) & (wy > 0) & (wy < h - 1)
            col = np.clip(np.round(wx / patch_size - 0.5), 0, pw - 1).astype(int)
            row = np.clip(np.round(wy / patch_size - 0.5), 0, ph - 1).astype(int)
            keep = mask & qvalid & inside
            tokens[i, j, keep] = (row * pw + col)[keep]
            pixels[i, j, :, 0], pixels[i, j, :, 1] = wx, wy
            valid[i, j] = keep
    return dict(gt_ref_token=tokens, gt_ref_pixel=pixels, valid=valid,
                meta=dict(ph=ph, pw=pw, H=h, W=w, patch_size=patch_size, hw=hw, N=n))
