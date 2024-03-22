import os
import torch
import numpy as np
import imageio 
import json

from .utils import get_bbox3d_for_blenderobj

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_data(basedir, pivot_name=''):
    with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
        meta = json.load(fp)

    all_imgs = []
    all_masks = []
    all_poses = []
    all_fnames = []
    counts = [0]

    pivot_id = -1
    imgs = []
    masks = []
    poses = []
    fnames = []

    skip = 1
    for frame in meta['frames'][::skip]:
        fname = os.path.join(basedir, frame['file_path'] )
        fnames.append(fname)
        imgs.append(imageio.imread(fname))
        try:
            mname = os.path.join(basedir, frame['file_path'].replace('images', 'masks') )
            temp = imageio.imread(mname)
            if len(temp.shape) == 3:
                masks.append(np.array(temp)[:, :, 0])
            else:
                masks.append(temp)
        except:
            pass
        poses.append(np.array(frame['transform_matrix']))
        if pivot_name in fname:
            pivot_id = len(imgs) - 1

    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    try:
        masks = (np.array(masks) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    except:
        pass
    poses = np.array(poses).astype(np.float32)
    counts.append(counts[-1] + imgs.shape[0])
    all_imgs.append(imgs)
    try:
        all_masks.append(masks)
    except:
        pass
    all_poses.append(poses)
    all_fnames.append(fnames)
    
    i_train = np.arange(counts[0], counts[1])
    
    imgs = np.concatenate(all_imgs, 0)
    try:
        masks = np.concatenate(all_masks, 0)
    except:
        pass
    poses = np.concatenate(all_poses, 0)
    fnames_out = []
    for this_fnames in all_fnames:
        fnames_out = fnames_out + this_fnames
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    bounding_box = get_bbox3d_for_blenderobj(meta, H, W, near=0.1, far=10.0)

    return imgs, masks, poses, pivot_id, render_poses, [H, W, focal], i_train, bounding_box, fnames_out