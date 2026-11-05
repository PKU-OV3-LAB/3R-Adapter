import os.path as osp
import cv2
import numpy as np
import itertools
import os
import sys
import json
sys.path.append(osp.join(osp.dirname(__file__), '..', '..'))
from tqdm import tqdm
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2

class OxfordDayAndNight(BaseMultiViewDataset):

    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = 4
        super().__init__(*args, **kwargs)
        self.light_condition = 'day_night'
        self.mode = 'within_scene'
        self.loaded_data = self._load_data()

    def _load_data(self):
        with open(osp.join(self.ROOT, f'dataset_info_{self.light_condition}.json')) as f:
            self.dataset_info = json.load(f)
        file_split = {'train': f'{self.light_condition}_{self.mode}_train.json', 'test': f'{self.light_condition}_{self.mode}_test.json'}[self.split]
        with open(osp.join(self.ROOT, file_split)) as f:
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
        j = 0
        for scene_sequence in tqdm(self.scenes_sequences_list):
            scene, sequence = scene_sequence.split('/')
            num_imgs = len(self.dataset_info[scene][sequence]['frames'])
            img_ids = list(np.arange(num_imgs) + offset)
            cut_off = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            if num_imgs < cut_off:
                print(f'Skipping {scene}')
                continue
            start_img_ids_ = img_ids[:num_imgs - cut_off + 1]
            start_img_ids.extend(start_img_ids_)
            sceneids.extend([j] * num_imgs)
            scene_img_list.append(img_ids)
            frames.extend(self.dataset_info[scene][sequence]['frames'])
            offset += num_imgs
            j += 1
        self.sceneids = sceneids
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list
        self.frames = frames

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.scene_img_list)

    def _get_views(self, idx, resolution, rng, num_views):
        start_id = self.start_img_ids[idx]
        all_image_ids = self.scene_img_list[self.sceneids[start_id]]
        pos, ordered_video = self.get_seq_from_start_id(num_views, start_id, all_image_ids, rng, max_interval=self.max_interval, video_prob=1.0, fix_interval_prob=1.0)
        image_idxs = np.array(all_image_ids)[pos]
        views = []
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene, sequence = self.scenes_sequences_list[scene_id].split('/')
            basename = os.path.basename(self.frames[view_idx]['img_path']).split('.')[0]
            rgb_image = imread_cv2(os.path.join(self.ROOT, self.frames[view_idx]['img_path']))
            depthmap = np.load(os.path.join(self.ROOT, self.frames[view_idx]['depth_path'])) / 1000.0
            depthmap[~np.isfinite(depthmap)] = 0
            depthmap[depthmap > 200.0] = 0.0
            intrinsics = np.array(self.frames[view_idx]['intrinsic']).astype(np.float32)
            extrinsic = np.array(self.frames[view_idx]['extrinsic']).astype(np.float32)
            camera_pose = np.linalg.inv(extrinsic)
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx)
            img_mask, ray_mask = self.get_img_and_ray_masks(self.is_metric, v, rng, p=[0.85, 0.1, 0.05])
            views.append(dict(img=rgb_image, depthmap=depthmap.astype(np.float32), camera_pose=camera_pose.astype(np.float32), camera_intrinsics=intrinsics.astype(np.float32), dataset='OxfordDayAndNight', label=scene + '_' + basename, instance=self.frames[view_idx]['img_path'], is_metric=self.is_metric, is_video=ordered_video, quantile=np.array(1, dtype=np.float32), img_mask=img_mask, ray_mask=ray_mask, camera_only=False, depth_only=False, single_view=False, reset=False))
        assert len(views) == num_views
        return views
