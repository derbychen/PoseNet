import torch
import numpy as np
import transforms3d.euler as txe
import transforms3d.quaternions as txq
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils.common import calc_poses_params, extract_translation
from PIL import Image
import os
import glob
import csv
import warnings
import pickle
import time

def process_poses(all_poses, pose_format='full-mat',
                  normalize_poses=True):
    # pose_format value here is the default(current) representation
    _, _, poses_mean, poses_std = calc_poses_params(all_poses, pose_format='full-mat')

    # print('poses_mean = {}'.format(poses_mean))
    # print('poses_std = {}'.format(poses_std))

    # Default pose format is full-mat
    new_poses = all_poses
    if pose_format == 'quat':
        # Convert to quaternions
        new_poses = np.zeros((len(all_poses), 7))
        for i in range(len(all_poses)):
            p = all_poses[i]
            R = p[:3, :3]
            t = p[:3, 3]
            q = txq.mat2quat(R)
            # Constrain rotations to one hemisphere
            q *= np.sign(q[0])
            new_poses[i, :3] = t
            new_poses[i, 3:] = q
        all_poses = new_poses

    if normalize_poses:
#         print('Poses Normalized! pose_format = {}'.format(pose_format))
        if pose_format == 'quat':
            all_poses[:, :3] -= poses_mean
            all_poses[:, :3] = np.divide(all_poses[:, :3], poses_std, where=poses_std!=0)
        else: # 'full-mat'
            all_poses[:, :3, 3] -= poses_mean
            all_poses[:, :3, 3] = np.divide(all_poses[:, :3, 3], poses_std, where=poses_std!=0)

    # print('all_poses samples = {}'.format(all_poses[:10]))

    return all_poses, poses_mean, poses_std

class KittiDset(Dataset):
    def __init__(self, img_path, txt_path, transform=None, 
                 stereo=False, normalize_poses=False):
        self.img_path = img_path
        self.txt_path = txt_path
        self.pose_format = 'quat'
        self.transform = transform
        self.stereo = stereo
        self.normalize_poses = normalize_poses

        pose_raw = self.read_poses_to_SE3(self.txt_path)
        ## pose_raw = SE3 at idx = 0

        self.label, poses_mean, poses_std = process_poses(pose_raw,
                                          pose_format=self.pose_format,
                                          normalize_poses=self.normalize_poses)
        self.poses_mean = poses_mean
        self.poses_std = poses_std

    def __getitem__(self, idx):
        pos = self.label[idx]
        pos = pos.astype(np.float32)
        pos = torch.from_numpy(pos)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pic_idx = str(idx)
        pic_idx = (6 - len(pic_idx)) * '0' + pic_idx + '.png'
        img_name = os.path.join(self.img_path, pic_idx)
        image = Image.open(img_name).convert('RGB') 
        if self.transform is not None:
            image = self.transform(image)
        image = np.asarray(image)
        image = torch.from_numpy(image)
        # image = torch.cat((image,image,image),dim=0) ## 3 channels
        return image, pos

    def __len__(self):
        return len(self.label)

    def read_poses_to_SE3(self, fname):
        """Reads poses from N x 12 format.
          Return SE3 format (Dict).
        """
        # poses = {}
        poses= []
        last = np.array([0, 0, 0, 1])
        with open(fname) as f:
            reader = csv.reader(f, delimiter=' ')
          
            for i, row in enumerate(reader):
                pose = np.asarray(row[:12], dtype=np.float).reshape(3, 4)
                pose = np.vstack((pose, last))
                poses.append(pose)
        return poses ## from N X 12 to SE(3)  