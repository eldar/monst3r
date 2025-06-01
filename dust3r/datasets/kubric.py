import sys
sys.path.append('.')
import os
import torch
import numpy as np
import os.path as osp
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from torch._C import dtype, set_flush_denormal
import dust3r.utils.po_utils.basic
import dust3r.utils.po_utils.improc
from dust3r.utils.po_utils.misc import farthest_point_sample_py
from dust3r.utils.po_utils.geom import apply_4x4_py, apply_pix_T_cam_py
import glob
import cv2
from torchvision.transforms import ColorJitter, GaussianBlur
from functools import partial

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class Kubric(BaseStereoViewDataset):
    """ Dataset of outdoor street scenes, 5 images each time
    """

    def __init__(self, *args, mask_bg=True, erode_mask=True, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.mask_bg = mask_bg
        self.erode_mask = erode_mask
        self.data_dir = osp.join(self.ROOT, self.split)
        self.scene_names = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.scene_names)

    def get_stats(self):
        return f'{len(self)} pairs'

    def _get_views(self, idx, resolution, rng):
        is_validation = self.split == "validation"
        scene_name = self.scene_names[idx]
        scene_path = osp.join(self.data_dir, scene_name)

        # decide now if we mask the bg
        mask_bg = self.mask_bg or (self.mask_bg == 'rand' and rng.choice(2))

        n_views = 2
        views = []
        for frame_id in range(n_views):
            impath = osp.join(scene_path, f'rgb_v_{frame_id:01n}.png')
            rgb_image = imread_cv2(impath)

            pc_path_t0 = osp.join(scene_path, f'pc_v_{frame_id:01n}_t_{frame_id:01n}.png')
            pc_16bit_t0 = cv2.imread(pc_path_t0, cv2.IMREAD_UNCHANGED)
            metadata_path_t0 = osp.join(scene_path, f'meta_v_{frame_id:01n}_t_{frame_id:01n}.npz')
            input_metadata_t0 = np.load(metadata_path_t0)
            max_pc_t0 = input_metadata_t0['max_pc']
            min_pc_t0 = input_metadata_t0['min_pc']
            pc_W_t0 = (pc_16bit_t0.astype(np.float32) / 65535)
            pc_W_t0 = pc_W_t0 * (max_pc_t0-min_pc_t0) + min_pc_t0

            camera_pose = input_metadata_t0['camera_pose'].astype(np.float32)
            intrinsics = input_metadata_t0['camera_intrinsics'].astype(np.float32)

            if mask_bg:
                # load object mask
                mask_path_t0 = osp.join(scene_path, f'mask_v_{frame_id:01n}_t_{frame_id:01n}.png')
                maskmap_t0 = imread_cv2(mask_path_t0)
                if self.erode_mask:
                    # maskmap_t0 = cv2.erode(maskmap_t0, np.ones((15, 15), np.uint8), iterations=1)
                    maskmap_t0 = cv2.dilate(maskmap_t0, np.ones((5, 5), np.uint8), iterations=1)
                maskmap_t0 = (maskmap_t0.astype(np.float32) / 255.0) < 0.5
                # update the pc with mask
                pc_W_t0 *= maskmap_t0

            intrinsics_orig = intrinsics
            rgb_image, pc_W_t0, intrinsics = (
                self._crop_resize_if_necessary(
                    rgb_image, pc_W_t0, intrinsics, resolution, rng=rng, info=impath) #, flip_landscape=(not is_validation)
            )

            # sanity check
            pc_W_t0 = pc_W_t0[..., :3]

            view_dict = dict(
                img=rgb_image,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                pts3d=pc_W_t0,
                dataset='Kubric',
                label=scene_name,
                instance=osp.split(impath)[1],
            )
            views.append(view_dict)

        return views

