import os
import os.path as osp
import json
import numpy as np
import torch
import torchvision.transforms as tvf

from typing import Optional, Union, List
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from datasets.utils.cropping import resize_image, resize_image_depth_and_intrinsic

to_tensor = tvf.ToTensor()


class OxfordDayAndNight(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "test",   # day / night / day_night
        light_condition: str = "day_night",
        mode: str = "within_scene",
        load_img_size: int = 512,
        cache_file: str = "data/dataset_cache/oxford_dn_cache.npy",
    ):
        """
        data_dir:
            oxford-day-and-night/
                aria/
                    bodleian-library/
                        ns_processed/multi/undistorted_all_valid/day_night_44/
        """

        self.data_dir = data_dir
        self.light_condition = light_condition
        self.split = split
        self.mode = mode
        self.load_img_size = load_img_size

        with open(osp.join(self.data_dir, f'dataset_info_{self.light_condition}.json')) as f:
            self.dataset_info = json.load(f)
        
        file_split = {"train": f"{self.light_condition}_{self.mode}_train.json", "test": f"{self.light_condition}_{self.mode}_test.json"}[self.split]
        with open(osp.join(self.data_dir, file_split)) as f:
            scenes_sequences = json.load(f)
            self.sequence_list = []
            for scene in scenes_sequences.keys():
                for sequence in scenes_sequences[scene]:
                    self.sequence_list.append(f'{scene}/{sequence}')
        print(f"[OxfordDN] #sequences = {len(self.sequence_list)}")

    def __len__(self):
        return len(self.sequence_list)

    def get_seq_framenum(self, index=None, sequence_name=None):
        if sequence_name is None:
            sequence_name = self.sequence_list[index]
        return 80

    def __getitem__(self, idx_N):
        index, n_per_seq = idx_N
        sequence_name = self.sequence_list[index]
        scene, sequence = sequence_name.split('/')
        frames = self.dataset_info[scene][sequence]['frames']
        seq_len = len(frames)
        ids = np.random.choice(seq_len, n_per_seq, replace=False)

        return self.get_data(index=index, ids=ids)

    def get_data(
        self,
        index: Optional[int] = None,
        sequence_name: Optional[str] = None,
        ids: Union[List[int], np.ndarray, None] = None,
    ):
        if sequence_name is None:
            sequence_name = self.sequence_list[index]
        scene, sequence = sequence_name.split('/')
        frames = self.dataset_info[scene][sequence]['frames']
        seq_len = len(frames)

        if ids is None:
            ids = np.arange(seq_len)
        elif isinstance(ids, np.ndarray):
            ids = ids.tolist()

        image_paths: list      = [""] * len(ids)
        images: list           = [0]  * len(ids)
        depths: list           = [0] * len(ids)
        extrinsics = np.zeros((len(ids), 3, 4), dtype=np.float32)
        intrinsics = np.zeros((len(ids), 3, 3), dtype=np.float32)
        camera_poses = np.zeros((len(ids), 4, 4), dtype=np.float32)

        for i, fid in enumerate(ids):
            frame = frames[fid]

            img_path = osp.join(self.data_dir, frame["img_path"])
            rgb_image: Image.Image = Image.open(img_path)
            depthmap = np.load(osp.join(self.data_dir, frame["depth_path"])) / 1000.0
            rgb_image: Image.Image = resize_image(rgb_image, (depthmap.shape[1], depthmap.shape[0]))
            
            extrinsics[i] = np.array(frame["extrinsic"], dtype=np.float32)[:3, :]
            intrinsics[i] = np.array(frame["intrinsic"], dtype=np.float32)
            camera_poses[i] = np.linalg.inv(np.array(frame["extrinsic"], dtype=np.float32))
            
            rgb_image, depthmap, intrinsics[i] = resize_image_depth_and_intrinsic(
                image=rgb_image,
                depth_map=depthmap,
                intrinsic=intrinsics[i],
                output_width=self.load_img_size, # finally width = 518, height = 388
            )
            
            image_paths[i] = img_path
            images[i]      = to_tensor(rgb_image)
            depths[i]      = depthmap

        depths = np.array(depths)  # (S, H, W)
        pointclouds = unproject_depth_map_to_point_map(
            depth_map=depths[..., None],
            intrinsics_cam=intrinsics,
            extrinsics_cam=extrinsics
        )

        batch = {"seq_id": sequence_name, "seq_len": seq_len, "ind": torch.tensor(ids)}
        batch['image_paths'] = image_paths  # list of str
        batch['images']      = torch.stack(images, dim=0)
        batch['pointclouds'] = pointclouds  # in numpy
        batch['valid_mask']  = depths > 1e-4

        batch["extrs"] = torch.from_numpy(extrinsics)
        batch["camera_poses"] = torch.from_numpy(camera_poses)
        # batch["fl"] = torch.stack(focal_lengths)
        # batch["pp"] = torch.stack(principal_points)
        
        return batch