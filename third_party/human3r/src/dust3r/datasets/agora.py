import os.path as osp
import numpy as np
import os
import sys
import sys
import pandas
sys.modules['pandas.core.indexes.numeric'] = pandas.core.indexes.base
import pickle
sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from tqdm import tqdm
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2
import torch
import copy
from dust3r.datasets.utils.transforms import ImgNorm, SeqColorJitter


invalid_seqs = []
hdri_scenes = []


class AGORA_Multi(BaseMultiViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.video = False
        self.is_metric = True
        self.max_interval = 4
        self.max_humans = 10
        self.smpl_key2shape = {
            'global_orient': (1, 3), 
            'body_pose': (23, 3), 
            'betas': (10,), 
            'transl': (3,), 
            'gender_id': ()
            }
        
        assert split in ['train','test','validation']
        self.split = split
        self.dataset_path = ROOT
        
        
        self.loaded_data = self._load_data(self.split)

    def _load_data(self, split):
        if self.split == 'train':
            annots_path = os.path.join(self.dataset_path,'smpl_neutral_annots','annots_smpl_{}_fit.npz'.format(split))
            self.annots = np.load(annots_path, allow_pickle=True)['annots'][()]
            self.img_names = list(self.annots.keys())
        elif self.split == 'validation':               
            annots_path = os.path.join(self.dataset_path,'smpl_neutral_annots','annots_smpl_{}.npz'.format(split))
            self.annots = np.load(annots_path, allow_pickle=True)['annots'][()]
            self.img_names = list(self.annots.keys())
        elif split == 'test':
            self.mode = 'infer'
            # no annotations are available for AGORA-test
            self.img_names = os.listdir(os.path.join(self.dataset_path, self.split, 'rgb'))
                
        self.scenes = self.img_names

        offset = 0
        scenes = []
        sceneids = []
        scene_img_list = []
        images = []
        start_img_ids = []

        j = 0
        for scene in tqdm(self.scenes):
            if "hdri" in scene or "closeup" in scene or not os.path.exists(os.path.join(self.dataset_path, self.split, 'smpl', scene.replace('.png', '.pkl'))):
                continue
            imagenames = [scene]
            num_imgs = 1
            img_ids = list(np.arange(num_imgs) + offset)
            cut_off = self.num_views
            if num_imgs < cut_off:
                print(f"Skipping {scene}")
                continue
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]
            start_img_ids.extend(start_img_ids_)
            sceneids.extend([j] * num_imgs)
            images.extend(imagenames)
            scenes.append(scene)
            scene_img_list.append(img_ids)

            # offset groups
            offset += num_imgs
            j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        start_id = self.start_img_ids[idx]
        all_image_ids = self.scene_img_list[self.sceneids[start_id]]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,
            max_interval=self.max_interval,
            video_prob=0,
            fix_interval_prob=1.0,
        )
        image_idxs = np.array(all_image_ids)[pos]

        views = []
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            # scene_dir = osp.join(self.ROOT, self.split, self.scenes[scene_id])
            rgb_dir = osp.join(self.ROOT, self.split, "rgb")
            depth_dir = osp.join(self.ROOT, self.split, "depth")
            mask_dir = osp.join(self.ROOT, self.split, "mask")
            cam_dir = osp.join(self.ROOT, self.split, "cam")
            smpl_dir = osp.join(self.ROOT, self.split, "smpl")

            imagename = self.images[view_idx]
            basename = os.path.splitext(imagename)[0]
            
            # Load RGB image
            rgb_image = imread_cv2(osp.join(rgb_dir, imagename))
            # Load mask image
            if os.path.exists(mask_dir):
                mask_image = imread_cv2(osp.join(mask_dir, basename+'.png'))
            else:
                mask_image = None
            # Load depthmap
            depthmap = np.load(osp.join(depth_dir, basename+".npy"))
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            depthmap[depthmap > 200.0] = 0.0
            
            cam = np.load(osp.join(cam_dir, basename + ".npz"))
            camera_pose = np.eye(4)
            intrinsics = cam["intrinsics"]

            annot_file = osp.join(smpl_dir, f"{basename}.pkl")
            annots = []
            smpl_mask = np.zeros(self.max_humans, dtype=np.bool_)
            if os.path.isfile(annot_file):
                with open(annot_file, 'rb') as f:
                    annots = pickle.load(f)
                humans = [hum for hum in annots if hum['transl'][-1] > 0.01] # the person should be in front of the camera
                if len(humans) > 0:
                    smpl_mask[:len(humans)] = 1.
                    l_dist = [hum['transl'][-1] for hum in humans]
                    indexed_lst = list(enumerate(l_dist))
                    sorted_indexed = sorted(indexed_lst, key=lambda x: x[1], reverse=False)
                    sorted_indices = [index for index, _ in sorted_indexed]
                    annots = [humans[h_idx] for h_idx in sorted_indices]

                    # Update smplx_gender - 0=neutral - 1=male - 2=female - kids?
                    for hum in annots:
                        hum['gender_id'] = np.asarray({'neutral': 0}[hum['gender']])

            if mask_image is not None:
                rgb_image, depthmap, mask_image, intrinsics = self._crop_resize_if_necessary_mask(rgb_image, depthmap, mask_image, intrinsics, resolution, rng=rng, info=view_idx)
            else:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx)

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(self.is_metric, v, rng, p=[0.85, 0.00, 0.15])
            # Reorganize the smpl annotations
            smpl_dict = {}
            for k in self.smpl_key2shape.keys():
                smpl_dict[k] = np.zeros((self.max_humans, *self.smpl_key2shape[k]), dtype=np.float32)
                if len(humans) > 0:
                    for h in range(len(humans)):
                        smpl_dict[k][h] = annots[h][k].astype(np.float32)
            
            views.append(
                dict(
                    img=rgb_image,
                    msk=False if mask_image is None else mask_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="AGORA",
                    label=imagename,
                    instance=osp.join(rgb_dir, imagename),
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(1, dtype=np.float32),
                    img_mask=img_mask,
                    ray_mask=ray_mask,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                    smpl_mask=smpl_mask,
                    **smpl_dict,
                )
            )

        assert len(views) == num_views
        return views


if __name__ == '__main__':
    dataset = AGORA_Multi(allow_repeat=True, split='train', ROOT="assets/SET_PATH",
                        aug_crop=16, resolution=[(288, 512)], transform=ImgNorm,
                        num_views=1, n_corres=0)
    sample = dataset[1]