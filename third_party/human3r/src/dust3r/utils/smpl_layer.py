import torch
from torch import nn
import smplx
import torch
from dust3r.utils.geometry import inverse_perspective_projection, perspective_projection
import roma
from dust3r.smpl_model import SMPLX_DIR, SMPLX2SMPL
from smplx.joint_names import JOINT_NAMES
import numpy as np
import pickle

class SMPL_Layer(nn.Module):

    def __init__(self, type='smplx', gender='neutral', num_betas=10, kid=False, person_center=None, *args, **kwargs):
        super().__init__()
        assert type == 'smplx'
        self.type = type
        self.kid = kid
        self.num_betas = num_betas
        self.bm_x = smplx.create(SMPLX_DIR, 'smplx', gender=gender, use_pca=False, flat_hand_mean=True, num_betas=num_betas)
        self.joint_names = JOINT_NAMES[:127]
        self.person_center = person_center
        self.person_center_idx = None
        if self.person_center is not None:
            self.person_center_idx = self.joint_names.index(self.person_center)
        with open(SMPLX2SMPL, 'rb') as f:
            self.smplx2smpl = torch.from_numpy(pickle.load(f)['matrix'].astype(np.float32)).to('cuda')
        self.j_regressor_smpl = smplx.create(SMPLX_DIR, 'smpl', gender='neutral').J_regressor[:24].to('cuda')

    def forward(self, pose, shape, transl, loc, dist, K, expression=None, K_to_proj=None, smplx_to_smpl=False):
        if loc is not None and dist is not None:
            assert pose.shape[0] == shape.shape[0] == loc.shape[0] == dist.shape[0]
            assert len(loc.shape) == 2 and list(loc.shape[1:]) == [2]
            assert len(dist.shape) == 2 and list(dist.shape[1:]) == [1]
        if self.type == 'smpl':
            assert len(pose.shape) == 3 and list(pose.shape[1:]) == [24, 3]
        elif self.type == 'smplx':
            assert len(pose.shape) == 3 and list(pose.shape[1:]) == [53, 3]
        else:
            raise NameError
        assert len(shape.shape) == 2 and (list(shape.shape[1:]) == [self.num_betas] or list(shape.shape[1:]) == [self.num_betas + 1])
        assert transl is not None or (loc is not None and dist is not None)
        bs = pose.shape[0]
        out = {}
        if bs == 0:
            return {}
        kwargs_pose = {'betas': shape}
        kwargs_pose['global_orient'] = self.bm_x.global_orient.repeat(bs, 1)
        kwargs_pose['body_pose'] = pose[:, 1:22].flatten(1)
        kwargs_pose['left_hand_pose'] = pose[:, 22:37].flatten(1)
        kwargs_pose['right_hand_pose'] = pose[:, 37:52].flatten(1)
        kwargs_pose['jaw_pose'] = pose[:, 52:53].flatten(1)
        if expression is not None:
            kwargs_pose['expression'] = expression.flatten(1)
        else:
            kwargs_pose['expression'] = self.bm_x.expression.repeat(bs, 1)
        kwargs_pose['leye_pose'] = self.bm_x.leye_pose.repeat(bs, 1)
        kwargs_pose['reye_pose'] = self.bm_x.reye_pose.repeat(bs, 1)
        output = self.bm_x(**kwargs_pose)
        verts = output.vertices
        j3d = output.joints
        R = roma.rotvec_to_rotmat(pose[:, 0])
        pelvis = j3d[:, [0]]
        j3d = (R.unsqueeze(1) @ (j3d - pelvis).unsqueeze(-1)).squeeze(-1)
        verts = (R.unsqueeze(1) @ (verts - pelvis).unsqueeze(-1)).squeeze(-1)
        if transl is None:
            if K.dtype == torch.float16:
                transl = inverse_perspective_projection(loc.unsqueeze(1).float(), K.float(), dist.unsqueeze(1).float())[:, 0]
                transl = transl.half()
            else:
                transl = inverse_perspective_projection(loc.unsqueeze(1), K, dist.unsqueeze(1))[:, 0]
        transl_up = transl.clone()
        if self.person_center_idx is None:
            transl_up = transl_up + pelvis[:, 0]
        else:
            person_center = j3d[:, [self.person_center_idx]]
            verts = verts - person_center
            j3d = j3d - person_center
        j3d_cam = j3d + transl_up.unsqueeze(1)
        verts_cam = verts + transl_up.unsqueeze(1)
        if smplx_to_smpl:
            verts_cam = self.smplx2smpl @ verts_cam
            j3d_cam = self.j_regressor_smpl @ verts_cam
        if K_to_proj is None:
            K_to_proj = K
        j2d = perspective_projection(j3d_cam, K_to_proj)
        v2d = perspective_projection(verts_cam, K_to_proj)
        out.update({'smpl_v3d': verts_cam, 'smpl_j3d': j3d_cam, 'smpl_j2d': j2d, 'smpl_v2d': v2d, 'smpl_transl': transl, 'smpl_transl_pelvis': j3d_cam[:, [0]]})
        return out
