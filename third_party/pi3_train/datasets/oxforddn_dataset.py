import os.path as osp
import cv2
import numpy as np
import itertools
import os
import sys
import json
sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from tqdm import tqdm
from datasets.base.base_dataset import BaseDataset
from PIL import Image
from datasets.base.transforms import *


class OxfordDayAndNight(BaseDataset):
    def __init__(self, data_root=None, verbose=False, max_distance=16, light_condition='day_night', scene_split='within_scene', **kwargs):
        super().__init__(**kwargs)
        assert data_root is not None
        self.data_root = data_root
        self.split = self.mode
        self.verbose = verbose
        self.max_distance = max_distance
        self.video = True
        self.is_metric = True
        self.max_interval = 2
        self.light_condition = light_condition
        self.scene_split = scene_split
        self.loaded_data = self._load_data()

    def _load_data(self):
        
        with open(osp.join(self.data_root, f'dataset_info_{self.light_condition}.json')) as f:
            self.dataset_info = json.load(f)
        
        file_split = {"train": f"{self.light_condition}_{self.scene_split}_train.json", "test": f"{self.light_condition}_{self.scene_split}_test.json"}[self.split]
        with open(osp.join(self.data_root, file_split)) as f:
            scenes_sequences = json.load(f)
            self.scenes_sequences_list = []
            for scene in scenes_sequences.keys():
                for sequence in scenes_sequences[scene]:
                    self.scenes_sequences_list.append(f'{scene}/{sequence}')
        
        offset = 0
        sceneids = []
        frames = []
        scene_img_list = []
        start_img_ids = []
        loaded_sequences = []

        j = 0
        for scene_sequence in tqdm(self.scenes_sequences_list):
            scene, sequence = scene_sequence.split('/')
            
            num_imgs = len(self.dataset_info[scene][sequence]['frames'])
            img_ids = list(np.arange(num_imgs) + offset)
            cut_off = 3
            if num_imgs < cut_off:
                print(f"Skipping {scene}")
                continue
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]
            loaded_sequences.append(scene_sequence)

            start_img_ids.extend(start_img_ids_)
            sceneids.extend([j] * num_imgs)
            scene_img_list.append(img_ids)
            frames.extend(self.dataset_info[scene][sequence]['frames'])

            # offset groups
            offset += num_imgs
            j += 1

        self.sceneids = sceneids
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list
        self.frames = frames
        self.scenes_sequences_list = loaded_sequences

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.scene_img_list)

    def _get_views(self, index, resolution, rng):
        start_id = self.start_img_ids[index]
        all_image_ids = self.scene_img_list[self.sceneids[start_id]]
        num_imgs = len(all_image_ids)
        valid_idxs = [i for i in range(num_imgs)]
        num_imgs = len(valid_idxs)
        
        if self.frame_num > 16 and rng.random() < self.random_sample_thres:
            all_keys = valid_idxs
            should_replace = len(all_keys) < self.frame_num
            idxs = list(rng.choice(all_keys, size=self.frame_num, replace=should_replace))
        else:
            idxs = [rng.integers(0, num_imgs)]

            max_distance = int(self.max_distance / 8 * self.frame_num)
            start_idx = max(0, idxs[-1] - max_distance)
            end_idx = min(num_imgs-1, start_idx + 2*max_distance)
            start_idx = max(0, end_idx - 2*max_distance)
            valid_indices = np.arange(start_idx, end_idx + 1)

            if rng.random() < 0.5:
                should_replace = len(valid_indices) < self.frame_num - 1
                idxs.extend(list(rng.choice(valid_indices, self.frame_num-1, replace=should_replace)))
                idxs = [valid_idxs[i] for i in idxs]
            else:
                ref_frame_val = idxs[0]
                num_additional_to_select = self.frame_num - 1
                additional_selected_values = []
                pool_for_others_values = [val for val in valid_indices]
                pool_for_others_values.sort() 

                should_replace_for_others = len(pool_for_others_values) < num_additional_to_select

                if not pool_for_others_values: 
                    if should_replace_for_others:
                        additional_selected_values = [ref_frame_val] * num_additional_to_select
                else:
                    if not should_replace_for_others and len(pool_for_others_values) >= num_additional_to_select:
                        strata = np.array_split(pool_for_others_values, num_additional_to_select+1)
                        for stratum in strata:
                            if len(stratum) > 0 and ref_frame_val not in stratum: 
                                additional_selected_values.append(rng.choice(stratum))
                    else:
                        additional_selected_values = list(rng.choice(
                            pool_for_others_values,
                            num_additional_to_select,
                            replace=(should_replace_for_others or (len(pool_for_others_values) < num_additional_to_select))
                        ))

                idxs = [ref_frame_val, *additional_selected_values]
                idxs = [valid_idxs[idx] for idx in idxs]


        idxs = [all_image_ids[idx] for idx in idxs]
        self.this_views_info = dict(
            scene=self.scenes_sequences_list[self.sceneids[start_id]],
            idxs=idxs,
        )
        

        views = []
        for v, view_idx in enumerate(idxs):
            scene_id = self.sceneids[view_idx]
            scene, sequence = self.scenes_sequences_list[scene_id].split('/')

            basename = os.path.basename(self.frames[view_idx]['img_path']).split('.')[0]
            # Load RGB image
            rgb_image = np.array(Image.open(os.path.join(self.data_root, self.frames[view_idx]['img_path'])))
            # Load depthmap
            depthmap = np.load(os.path.join(self.data_root, self.frames[view_idx]['depth_path'])) / 1000.0
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            depthmap[depthmap > 200.0] = 0.0

            intrinsics = np.array(self.frames[view_idx]["intrinsic"]).astype(np.float32)
            extrinsic = np.array(self.frames[view_idx]["extrinsic"]).astype(np.float32)
            camera_pose = np.linalg.inv(extrinsic)
            
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx)

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset="OxfordDayAndNight",
                label=scene,
                instance=str(view_idx),
            ))
        
        return views
