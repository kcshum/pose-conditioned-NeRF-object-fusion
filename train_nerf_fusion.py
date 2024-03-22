from diffusion_utils.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline

from pathlib import Path
from PIL import Image
from numpy.linalg import inv, norm
from PIL import ImageDraw
import cv2

import os
import numpy as np
import random
import math
import time
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm, trange

from nerf_utils.run_nerf_helpers import *
from nerf_utils.radam import RAdam
from nerf_utils.loss import sigma_sparsity_loss, total_variation_loss

from nerf_utils.load_data import load_data
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------- #
# --------- nerf functions --------- #
# ---------------------------------- #

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded, keep_mask = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs_flat[~keep_mask, -1] = 0 # set sigma to 0 for invalid points
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'depth_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def render_path_train(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None):

    H, W, focal = hwf
    near, far = render_kwargs['near'], render_kwargs['far']

    rgbs = []
    depths = []

    rgb, depth, acc, _ = render(H, W, K, chunk=chunk, c2w=render_poses, **render_kwargs)
    rgbs.append(rgb)
    # normalize depth to [0,1]
    depth = (depth - near) / (far - near)
    depths.append(depth)

    return rgbs, depths

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args, i=args.i_embed)
    if args.i_embed==1:
        # hashed embedding table
        embedding_params = list(embed_fn.parameters())

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        # if using hashed for xyz, use SH for views
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args, i=args.i_embed_views)

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    if args.i_embed==1:
        model = NeRFSmall(num_layers=2,
                        hidden_dim=64,
                        geo_feat_dim=15,
                        num_layers_color=3,
                        hidden_dim_color=64,
                        input_ch=input_ch, input_ch_views=input_ch_views).to(device)
    else:
        model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None

    if args.N_importance > 0:
        if args.i_embed==1:
            model_fine = NeRFSmall(num_layers=2,
                        hidden_dim=64,
                        geo_feat_dim=15,
                        num_layers_color=3,
                        hidden_dim_color=64,
                        input_ch=input_ch, input_ch_views=input_ch_views).to(device)
        else:
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    if args.i_embed==1:
        optimizer = RAdam([
                            {'params': grad_vars, 'weight_decay': 1e-6},
                            {'params': embedding_params, 'eps': 1e-15}
                        ], lr=args.lrate, betas=(0.9, 0.99))
    else:
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    steps_kwargs = {
        'start' : 0,
    }
    basedir = args.basedir
    expname = args.expname

    filler = '@'
    to_expname_name_list = ['', '',
                            'box-',
                            'l', 'h',
                            ]

    to_expname_var_list = [str(args.prompt).replace(' ', '-'), str(os.path.basename(args.finetuned_model_path)),
                           str(args.box_name),
                           str(args.strength_lower_bound).zfill(3), str(args.strength_higher_bound).zfill(3),
                           ]

    for (namee, varr) in zip(to_expname_name_list, to_expname_var_list):
        expname = expname + filler + namee + varr

    ##########################
    # Load checkpoints
    ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        for ckpttt in ckpts:
            if ((args.ckpt_epoch_to_load).zfill(6) + '.tar') in ckpttt:
                ckpt_path = ckpttt

        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        steps_kwargs['start'] = ckpt['global_step']

        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        if args.i_embed==1:
            embed_fn.load_state_dict(ckpt['embed_fn_state_dict'])

    ##########################
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'embed_fn': embed_fn,
        'use_viewdirs' : args.use_viewdirs,
        'raw_noise_std' : args.raw_noise_std,
    }

    render_kwargs_train['ndc'] = False
    return render_kwargs_train, steps_kwargs, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    # sigma_loss = sigma_sparsity_loss(raw[...,3])
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1) / torch.sum(weights, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map)
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    # Calculate weights sparsity loss
    entropy = Categorical(probs = torch.cat([weights, 1.0-weights.sum(-1, keepdim=True)+1e-4], dim=-1)).entropy()

    sparsity_loss = entropy

    return rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                embed_fn=None,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, depth_map_0, acc_map_0, sparsity_loss_0 = rgb_map, depth_map, acc_map, sparsity_loss

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'depth_map' : depth_map, 'acc_map' : acc_map, 'sparsity_loss': sparsity_loss}
    ret['weights'] = weights
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['depth0'] = depth_map_0
        ret['acc0'] = acc_map_0
        ret['sparsity_loss0'] = sparsity_loss_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    return ret


# -------------------------------------------- #
# --------- all customizable configs --------- #
# -------------------------------------------- #

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()

    ### ~~~ nerf config ~~~ ###
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='logs',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='dataset/background/default',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4*2,
                        help='batch size of first stage nerf (number of random rays per gradient step)')
    parser.add_argument("--N_rand_mask", type=int, default=32*32*4*2,
                        help='batch size of masked area  (number of random rays per gradient step)')
    parser.add_argument("--N_rand_unmask", type=int, default=32*32*4*2,
                        help='batch size of unmasked area (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate of background nerf training stage')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--lrate_fusion", type=float, default=5e-4,
                        help='learning rate of object fusion stage')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=1,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--i_embed_views", type=int, default=2,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_image", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument(
        "--ckpt_epoch_to_load",
        type=str,
        default='0',
        help="which saved NeRF model to load, identified by epoch #.",
    )

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')
    parser.add_argument("--finest_res",   type=int, default=512,
                        help='finest resolultion for hashed embedding')
    parser.add_argument("--log2_hashmap_size",   type=int, default=19,
                        help='log2 of hashmap size')
    parser.add_argument("--sparse-loss-weight", type=float, default=1e-10,
                        help='learning rate')
    parser.add_argument("--tv-loss-weight", type=float, default=1e-6,
                        help='learning rate')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='every # of iters, print training info to screen.')
    parser.add_argument("--i_weights", type=int, default=5000,
                        help='every # of iters, save the NeRF model.')
    parser.add_argument("--i_trainset", type=int, default=5000,
                        help='every # of iters, render ALL training views from NeRF and updating dataset.')
    parser.add_argument("--i_visualization", type=int, default=100,
                        help='every # of iters, visualize ONLY the near views in the updating dataset.')

    ### ~~~ diffusion model and pose condition config ~~~ ###
    parser.add_argument("--finetuned_model_path", type=str, default='dream_outputs/default',
                        help='fine-tuned diffusion model to load')
    parser.add_argument(
        "--pivot_name",
        type=str,
        nargs="?",
        default="default.jpg",
        help="pivot file name, it is the first view to train for object fusion."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="",
        help="Diffusion model text prompt, used to produce images."
    )
    parser.add_argument(
        "--strength_lower_bound",
        type=int,
        default=35,
        help="Diffusion model noise strength lower bound, used in training. Should range from 0 to 100.",
    )
    parser.add_argument(
        "--strength_higher_bound",
        type=int,
        default=35,
        help="Diffusion model noise strength higher bound, used in training. Should range from 0 to 100.",
    )
    parser.add_argument(
        "--strength_test",
        type=int,
        default=35,
        help="Diffusion model noise strength, used in testing. Should range from 0 to 100.",
    )
    parser.add_argument(
        "--test_num_inference_steps",
        type=int,
        default=50,
        help="Diffusion model inference steps, used in testing.",
    )
    parser.add_argument(
        "--initial_iter",
        type=int,
        default=20000,
        help="# of iters to train on background before object fusion",
    )
    parser.add_argument(
        "--operation_iter",
        type=int,
        default=10,
        help="every # of iters, perform once the periodic dataset update for object fusion",
    )
    parser.add_argument(
        "--new_far_image_iters",
        type=int,
        default=500,
        help="every # of iters, perform once the pose-conditioned dataset update for object fusion "
             "(to include new near images)",
    )
    parser.add_argument(
        "--extra_train_on_pivot_iter",
        type=int,
        default=500,
        help="# of extra iters to train on the pivot only when object fusion starts",
    )
    parser.add_argument(
        "--num_Neighbors",
        type=int,
        default=2,
        help="# of new views to be included in pose-conditioned dataset updates",
    )
    parser.add_argument(
        "--box_name",
        type=str,
        default='default',
        help="which bounding box to use",
    )

    parser.add_argument("--no_vis_box", action='store_true',
                        help='do not visualize bounding box to save time')

    # for rendering videos
    parser.add_argument("--render_video", action='store_true',
                        help='to render video results (test only, no training)')

    parser.add_argument(
        "--num_Gaps",
        type=int,
        default=10,
        help="# of novel views to render between two images, for video rendering only",
    )

    parser.add_argument("--video_expname", type=str, default='default',
                        help='folder name to store the output images, lazy implementation')

    parser.add_argument("--video_frames", nargs='+',
                        help='a list of views of interest where a smooth trajectory renders through'
                             'note: the trajectory will complete a loop at the end (last view to first view)')

    parser.add_argument("--strength_video",
                        type=int,
                        default=35,
                        help="Diffusion model noise strength, used in video rendering. Should range from 0 to 100.",
    )

    ### ~~~ other config ~~~ ###
    # for background data rendering boundary: near and far
    parser.add_argument(
        "--data_near",
        type=float,
        default=0.1,
        help="background data, the nearest distance to render",
    )

    parser.add_argument(
        "--data_far",
        type=float,
        default=10.0,
        help="background data, the farthest distance to render",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed everything (for reproducible sampling)",
    )

    return parser



def train():
    parser = config_parser()
    args = parser.parse_args()

    # ---------------------------------------- #
    # --------- load diffusion model --------- #
    # ---------------------------------------- #
    model_path = Path(args.finetuned_model_path)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_path, local_files_only=False)

    print('loaded fine-tuned diffusion model from {} !'.format(model_path))

    # ---------------------------------------------- #
    # --------- load image and camera data --------- #
    # ---------------------------------------------- #
    import paddle
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        paddle.seed(args.seed)

    setup_seed(args.seed)

    K = None

    images, masks, poses, pivot_id, render_poses, hwf, i_train, bounding_box, fnames_out = load_data(args.datadir, args.pivot_name)
    args.bounding_box = bounding_box
    print('Number of Images Loaded: {} from {} !'.format(images.shape[0], args.datadir))
    print('TRAIN views are', i_train)
    print('pivot_id is: {} of image {} !'.format(pivot_id, fnames_out[pivot_id]))

    near = args.data_near
    far = args.data_far

    images = images[...,:3]

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    # --------------------------------------------- #
    # --------- save commands to txt file --------- #
    # --------------------------------------------- #
    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname

    filler = '@'
    to_expname_name_list = ['', '',
                            'box-',
                            'l', 'h',
                            ]

    to_expname_var_list = [str(args.prompt).replace(' ', '-'), str(os.path.basename(args.finetuned_model_path)),
                           str(args.box_name),
                           str(args.strength_lower_bound).zfill(3), str(args.strength_higher_bound).zfill(3),
                           ]

    for (namee, varr) in zip(to_expname_name_list, to_expname_var_list):
        expname = expname + filler + namee + varr

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # ----------------------------------- #
    # --------- load nerf model --------- #
    # ----------------------------------- #
    render_kwargs_train, steps_kwargs, grad_vars, optimizer = create_nerf(args)

    start = steps_kwargs['start']
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand

    # Move training data to GPU
    poses = torch.Tensor(poses).to(device)

    # ------------------------------------------------------------- #
    # --------- construct a bounding box and visualize it --------- #
    # Sorry for the lazy implementation.
    # It lacks a convenient UI or automatic way to construct a box.
    # Current approach is like:
    #    - determine a rectangle (adjust a point and two orthogonal vectors)
    #    - then determine a cube (adjust rectangle surface normal)
    # ------------------------------------------------------------- #
    # default bounding box
    if args.box_name == 'default':
        # you may rotate the points as [x, y, z]
        # the points are initialized around the center of the scene
        # rotate negative-positive the x:y:z axis = lie down right-left : lie down far-come : self-rotate right-left
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([1.0, 2.0, 0.0])

        # adjust rectangle (length and width)
        direct_01 = p1 - p0
        direct_02 = p2 - p1
        p0 = p0 + 0. * direct_01
        p1 = p1 + 0. * direct_01
        p2 = p2 + 0. * direct_01

        p0 = p0 + 0. * direct_02
        p1 = p1 + 0. * direct_02
        p2 = p2 + 0. * direct_02

        # adjust cube (height)
        norm_h = 0.5

        norm_vec = np.cross(p1-p0, p2-p0)
        normalze_val = norm(norm_vec)
        if normalze_val == 0:
            raise Exception
        norm_vec = norm_vec/normalze_val
        norm_vec = norm_vec * norm_h

        p0 = p0 + 0. * norm_vec
        p1 = p1 + 0. * norm_vec
        p2 = p2 + 0. * norm_vec

    elif args.box_name == 'wooden_table_01':
        p0 = np.array([1.0, -0.75, 0.0])
        p1 = np.array([0.5, -0.75, 0.75])
        p2 = np.array([0.5, 0.75, 0.75])

        direct_01 = p1 - p0
        direct_02 = p2 - p1
        p0 = p0 - 0.2 * direct_01
        p1 = p1 + 0.2 * direct_01
        p2 = p2 + 0.2 * direct_01

        p0 = p0 - 0.2 * direct_02
        p1 = p1 - 0.2 * direct_02
        p2 = p2 + 0.2 * direct_02

        norm_h = 0.5

        norm_vec = np.cross(p1 - p0, p2 - p0)
        normalze_val = norm(norm_vec)
        if normalze_val == 0:
            raise Exception
        norm_vec = norm_vec / normalze_val
        norm_vec = norm_vec * norm_h

    elif args.box_name == 'wooden_table_02':
        p0 = np.array([1.0, -0.75, 0.0])
        p1 = np.array([0.5, -0.75, 0.75])
        p2 = np.array([0.5, 0.75, 0.75])

        direct_01 = p1 - p0
        direct_02 = p2 - p1
        p0 = p0 - 0.15 * direct_01
        p1 = p1 + 0.05 * direct_01
        p2 = p2 + 0.05 * direct_01

        p0 = p0 - 0.1 * direct_02
        p1 = p1 - 0.1 * direct_02
        p2 = p2 + 0.05 * direct_02

        norm_h = 0.4

        norm_vec = np.cross(p1-p0, p2-p0)
        normalze_val = norm(norm_vec)
        if normalze_val == 0:
            raise Exception
        norm_vec = norm_vec/normalze_val
        norm_vec = norm_vec * norm_h

        p0 = p0 - 0.7 * norm_vec
        p1 = p1 - 0.7 * norm_vec
        p2 = p2 - 0.7 * norm_vec

    elif args.box_name == 'wooden_table_03':
        p0 = np.array([1.0, -0.75, 0.0])
        p1 = np.array([0.5, -0.75, 0.75])
        p2 = np.array([0.5, 0.75, 0.75])

        direct_01 = p1 - p0
        direct_02 = p2 - p1
        p0 = p0 - 0.3 * direct_01
        p1 = p1 + 0.2 * direct_01
        p2 = p2 + 0.2 * direct_01

        p0 = p0 - 0.2 * direct_02
        p1 = p1 - 0.2 * direct_02
        p2 = p2 + 0.2 * direct_02

        norm_h = 0.5

        norm_vec = np.cross(p1-p0, p2-p0)
        normalze_val = norm(norm_vec)
        if normalze_val == 0:
            raise Exception
        norm_vec = norm_vec/normalze_val
        norm_vec = norm_vec * norm_h

        p0 = p0 - 0.7 * norm_vec
        p1 = p1 - 0.7 * norm_vec
        p2 = p2 - 0.7 * norm_vec

    elif args.box_name == 'black_table_01':
        # rot degree : 0 : 38 : -12
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([0.77, -0.21, -0.60])
        p2 = np.array([1.08, 1.30, -0.62])

        translation = np.array([-0.3, -0.5, -0.3])
        p0 = p0 + translation
        p1 = p1 + translation
        p2 = p2 + translation

        direct_01 = p1 - p0
        direct_02 = p2 - p1
        p0 = p0 + -.4 * direct_01
        p1 = p1 + -.4 * direct_01
        p2 = p2 + -.4 * direct_01

        p0 = p0 + -.1 * direct_01
        p1 = p1 + .3 * direct_01
        p2 = p2 + .3 * direct_01

        p0 = p0 + -.0 * direct_02
        p1 = p1 + -.0 * direct_02
        p2 = p2 + -.0 * direct_02

        p0 = p0 + -.0 * direct_02
        p1 = p1 + -.0 * direct_02
        p2 = p2 + .2 * direct_02

        norm_h = 0.4

        norm_vec = np.cross(p1 - p0, p2 - p0)
        normalze_val = norm(norm_vec)
        if normalze_val == 0:
            raise Exception
        norm_vec = norm_vec / normalze_val
        norm_vec = norm_vec * norm_h

        p0 = p0 + .0 * norm_vec
        p1 = p1 + .0 * norm_vec
        p2 = p2 + .0 * norm_vec

    elif args.box_name == 'black_table_02':
        # rot degree : 0 : 38 : -12
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([0.77, -0.21, -0.60])
        p2 = np.array([1.08, 1.30, -0.62])

        translation = np.array([-0.3, -0.5, -0.3])
        p0 = p0 + translation
        p1 = p1 + translation
        p2 = p2 + translation

        direct_01 = p1 - p0
        direct_02 = p2 - p1
        p0 = p0 + -.0 * direct_01
        p1 = p1 + -.0 * direct_01
        p2 = p2 + -.0 * direct_01

        p0 = p0 + -.1 * direct_01
        p1 = p1 + .3 * direct_01
        p2 = p2 + .3 * direct_01

        p0 = p0 + -.0 * direct_02
        p1 = p1 + -.0 * direct_02
        p2 = p2 + -.0 * direct_02

        p0 = p0 + -.0 * direct_02
        p1 = p1 + -.0 * direct_02
        p2 = p2 + .2 * direct_02

        norm_h = 0.4

        norm_vec = np.cross(p1 - p0, p2 - p0)
        normalze_val = norm(norm_vec)
        if normalze_val == 0:
            raise Exception
        norm_vec = norm_vec / normalze_val
        norm_vec = norm_vec * norm_h

        p0 = p0 + .0 * norm_vec
        p1 = p1 + .0 * norm_vec
        p2 = p2 + .0 * norm_vec

    elif args.box_name == 'sofa_01':
        # rot degree : 2 : 40 : 15
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([0.74,0.20,-0.64])
        p2 = np.array([0.27,2.14,-0.59])

        direct_01 = p1 - p0
        direct_02 = p2 - p1
        p0 = p0 + -0.3 * direct_01
        p1 = p1 + 0.2 * direct_01
        p2 = p2 + 0.2 * direct_01

        p0 = p0 - 0.5 * direct_02
        p1 = p1 - 0.5 * direct_02
        p2 = p2 - 0.5 * direct_02

        norm_h = 0.7

        norm_vec = np.cross(p1-p0, p2-p0)
        normalze_val = norm(norm_vec)
        if normalze_val == 0:
            raise Exception
        norm_vec = norm_vec/normalze_val
        norm_vec = norm_vec * norm_h

        p0 = p0 + 0.8 * norm_vec
        p1 = p1 + 0.8 * norm_vec
        p2 = p2 + 0.8 * norm_vec

    elif args.box_name == 'bed_01':
        # rot degree : -5 : 48 : -35
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([0.55,-0.38,-0.74])
        p2 = np.array([1.58,1.32,-0.86])

        direct_01 = p1 - p0
        direct_02 = p2 - p1
        p0 = p0 + 0.1 * direct_01
        p1 = p1 + 0.45 * direct_01
        p2 = p2 + 0.45 * direct_01

        p0 = p0 - 0.5 * direct_02
        p1 = p1 - 0.5 * direct_02
        p2 = p2 - 0.4 * direct_02

        norm_h = 0.5

        norm_vec = np.cross(p1-p0, p2-p0)
        normalze_val = norm(norm_vec)
        if normalze_val == 0:
            raise Exception
        norm_vec = norm_vec/normalze_val
        norm_vec = norm_vec * norm_h

        p0 = p0 + 0.6 * norm_vec
        p1 = p1 + 0.6 * norm_vec
        p2 = p2 + 0.6 * norm_vec

    elif args.box_name == 'bed_02':
        # rot degree : -5 : 48 : -35
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([0.55,-0.38,-0.74])
        p2 = np.array([1.58,1.32,-0.86])

        direct_01 = p1 - p0
        direct_02 = p2 - p1
        p0 = p0 + 0.1 * direct_01
        p1 = p1 + 0.45 * direct_01
        p2 = p2 + 0.45 * direct_01

        p0 = p0 - 0.5 * direct_02
        p1 = p1 - 0.5 * direct_02
        p2 = p2 - 0.4 * direct_02

        norm_h = 0.7

        norm_vec = np.cross(p1-p0, p2-p0)
        normalze_val = norm(norm_vec)
        if normalze_val == 0:
            raise Exception
        norm_vec = norm_vec/normalze_val
        norm_vec = norm_vec * norm_h

        p0 = p0 + 0.6 * norm_vec
        p1 = p1 + 0.6 * norm_vec
        p2 = p2 + 0.6 * norm_vec

    elif args.box_name == 'floor_01':
        # rot degree : 2 : 45 : -5
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([0.70,-0.06,-0.71])
        p2 = np.array([0.93,1.93,-0.66])

        direct_01 = p1 - p0
        direct_02 = p2 - p1
        p0 = p0 + -0.4 * direct_01
        p1 = p1 + 0.2 * direct_01
        p2 = p2 + 0.2 * direct_01

        p0 = p0 - 0.6 * direct_02
        p1 = p1 - 0.6 * direct_02
        p2 = p2 - 0.4 * direct_02

        norm_h = 0.7

        norm_vec = np.cross(p1-p0, p2-p0)
        normalze_val = norm(norm_vec)
        if normalze_val == 0:
            raise Exception
        norm_vec = norm_vec/normalze_val
        norm_vec = norm_vec * norm_h

        p0 = p0 + 0.8 * norm_vec
        p1 = p1 + 0.8 * norm_vec
        p2 = p2 + 0.8 * norm_vec

    elif args.box_name == 'wall_01':
        # rot degree : 3 : 8 : 43
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([0.72,0.68,-0.14])
        p2 = np.array([-0.29,1.78,-0.06])

        translation = np.array([0., 0., -.3])
        p0 = p0 + translation
        p1 = p1 + translation
        p2 = p2 + translation

        direct_01 = p1 - p0
        direct_02 = p2 - p1
        p0 = p0 + .2 * direct_01
        p1 = p1 + .2 * direct_01
        p2 = p2 + .2 * direct_01

        p0 = p0 + .0 * direct_01
        p1 = p1 + -.8 * direct_01
        p2 = p2 + -.8 * direct_01

        p0 = p0 + -.45 * direct_02
        p1 = p1 + -.45 * direct_02
        p2 = p2 + -.45 * direct_02

        p0 = p0 + -.05 * direct_02
        p1 = p1 + -.05 * direct_02
        p2 = p2 + -.2 * direct_02

        norm_h = 1.4

        norm_vec = np.cross(p1-p0, p2-p0)
        normalze_val = norm(norm_vec)
        if normalze_val == 0:
            raise Exception
        norm_vec = norm_vec/normalze_val
        norm_vec = norm_vec * norm_h

        p0 = p0 + -.2 * norm_vec
        p1 = p1 + -.2 * norm_vec
        p2 = p2 + -.2 * norm_vec

    elif args.box_name == 'wooden_floor_01':
        # rot degree : 10 : 15 : 25
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([0.88,0.41,-0.26])
        p2 = np.array([0.12,2.23,0.08])

        direct_01 = p1 - p0
        direct_02 = p2 - p1
        p0 = p0 - 0.1 * direct_01
        p1 = p1 + 1.2 * direct_01
        p2 = p2 + 1.2 * direct_01

        p0 = p0 - 0.6 * direct_02
        p1 = p1 - 0.6 * direct_02
        p2 = p2 - 0.6 * direct_02

        norm_h = 0.7

        norm_vec = np.cross(p1-p0, p2-p0)
        normalze_val = norm(norm_vec)
        if normalze_val == 0:
            raise Exception
        norm_vec = norm_vec/normalze_val
        norm_vec = norm_vec * norm_h

        p0 = p0 + 0.4 * norm_vec
        p1 = p1 + 0.4 * norm_vec
        p2 = p2 + 0.4 * norm_vec

    elif args.box_name == 'road_01':
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.3, 0.0, 0.0])
        p2 = np.array([1.3, 1.4, 0.0])

        norm_h = 0.5

        translation = np.array([0.4, -0.4, -0.33])
        p0 = p0 + translation
        p1 = p1 + translation
        p2 = p2 + translation

        norm_vec = np.cross(p1 - p0, p2 - p0)
        normalze_val = norm(norm_vec)
        if normalze_val == 0:
            raise Exception
        norm_vec = norm_vec / normalze_val
        norm_vec = norm_vec * norm_h

    else:
        raise Exception('no bounding box of a scene named like this')

    p3 = p0 + p2 - p1
    p4 = p0 + norm_vec
    p5 = p1 + norm_vec
    p6 = p2 + norm_vec
    p7 = p3 + norm_vec

    ##### visualize bounding box #####
    world_xyz_list = [np.transpose(np.array([p0])), np.transpose(np.array([p1])),
                      np.transpose(np.array([p2])), np.transpose(np.array([p3])),
                      np.transpose(np.array([p4])), np.transpose(np.array([p5])),
                      np.transpose(np.array([p6])), np.transpose(np.array([p7]))]
    line_index_list = [(0, 1), (1, 2), (2, 3), (3, 0),
             (0, 4), (1, 5), (2, 6), (3, 7),
             (4, 5), (5, 6), (6, 7), (7, 4)]
    face_index_list = [(0, 1, 2, 3), (4, 5, 6, 7),
                        (0, 1, 5, 4), (3, 2, 6, 1),
                        (0, 3, 7, 4), (1, 2, 6, 5)]
    line_color_list = ['red', 'red', 'red', 'red',
                        'red', 'red', 'red', 'red',
                        'red', 'red', 'red', 'red']

    def world2frame(world_xyz_list, K, c2w):
        frame_wh_list = []
        for world_xyz in world_xyz_list:
            R = c2w[:3, :3].cpu().numpy()
            t = c2w[:3, 3].cpu().numpy()
            t = np.transpose(np.array([t]))

            camera_xyz = np.matmul(inv(R), world_xyz - t)
            camera_xyz = -1. * camera_xyz/camera_xyz[-1]

            frame_w = camera_xyz[0] * K[0][0] + K[0][2]
            frame_h = -1 * (camera_xyz[1] * K[1][1]) + K[1][2]
            frame_wh = (int(frame_w[0]), int(frame_h[0]))
            frame_wh_list.append(frame_wh)
        return frame_wh_list

    if not args.no_vis_box:
        for i in trange(len(images)):
            image = np.array(images[i])
            pose = poses[i]
            frame_wh_list = world2frame(world_xyz_list=world_xyz_list, K=K, c2w=pose)

            image = 255. * image
            boundingboxsavedir = os.path.join(basedir, expname, 'boundingbox')
            os.makedirs(boundingboxsavedir, exist_ok=True)
            fname = fnames_out[i]
            fname = os.path.basename(fname)
            fname = str(os.path.splitext(fname)[0])

            image = Image.fromarray(image.astype(np.uint8))
            img1 = ImageDraw.Draw(image)
            for line_index, line_color in zip(line_index_list, line_color_list):
                line_coor = [frame_wh_list[line_index[0]], frame_wh_list[line_index[1]]]

                img1.line(line_coor, fill=line_color, width=6)
            image.save(os.path.join(boundingboxsavedir, 'boxed_' + fname + '.png'))

    # --------------------------------------------------------- #
    # --------- prepare for a near2far updating graph --------- #
    # --------------------------------------------------------- #
    with torch.no_grad():
        all_dist = {}
        for j in i_train:
            for k in i_train:
                if j == k:
                    continue
                translation_j = poses[j, :3, 3:].cpu().numpy()
                translation_k = poses[k, :3, 3:].cpu().numpy()
                dist = np.linalg.norm(translation_j - translation_k)

                key = str(j) + '_' + str(k)
                all_dist[key] = dist

        all_dist = sorted(all_dist.items(), key=lambda x: x[1])
        all_dist = dict(all_dist)

    done_nodes_for_pseudo = [np.random.choice(i_train) if pivot_id == -1 else pivot_id]

    # get a near2far order for the dataset
    this_to_nodes_for_pseudo = []
    while len(done_nodes_for_pseudo) != len(i_train):
        done_nodes_for_pseudo = done_nodes_for_pseudo + this_to_nodes_for_pseudo

        going_pool_for_pseudo = []
        this_to_nodes_for_pseudo = []
        for key in all_dist:
            pairs = key.split('_')
            from_node = int(pairs[0])
            to_node = int(pairs[1])

            if (from_node in done_nodes_for_pseudo) and (to_node not in done_nodes_for_pseudo) and (to_node not in this_to_nodes_for_pseudo):
                this_to_nodes_for_pseudo.append(to_node)

                going_pool_for_pseudo.append([from_node, to_node])
                if len(going_pool_for_pseudo) == args.num_Neighbors:
                    break

    # ------------------------------------------------------------------------- #
    # --------- the dataset update function, for training and testing --------- #
    # ------------------------------------------------------------------------- #

    def generate_one_data_point(data_point_id, diffusion_strength, num_inference_steps=50):
        with torch.no_grad():
            pose = poses[data_point_id, :3, :4]

            frame_wh_list = world2frame(world_xyz_list=world_xyz_list, K=K, c2w=pose)
            mask_image = np.zeros_like(images[0])
            for face_index in face_index_list:
                contour = [np.array(
                    [
                        frame_wh_list[face_index[0]],
                        frame_wh_list[face_index[1]],
                        frame_wh_list[face_index[2]],
                        frame_wh_list[face_index[3]]
                    ]
                    , dtype=np.int32)]
                cv2.drawContours(mask_image, contour, 0, (255, 255, 255), -1)

            mask_image = mask_image / 255.
            new_going_mask = torch.Tensor(mask_image).to(device)

            if diffusion_strength == 1.0:
                init_image, _ = render_path_train(torch.Tensor(pose).to(device), hwf, K, args.chunk, render_kwargs_train)
                init_image = init_image[0]
                init_image = init_image.cpu().numpy()

                init_image = np.random.rand(init_image.shape[0], init_image.shape[1], init_image.shape[2])
            else:
                init_image, _ = render_path_train(torch.Tensor(pose).to(device), hwf, K, args.chunk, render_kwargs_train)
                init_image = init_image[0]
                init_image = init_image.cpu().numpy()

            GT_BG = images[data_point_id]

            instance_image = np.multiply(init_image, mask_image) + np.multiply(GT_BG, 1. - mask_image)
            nerf_image_to_save = torch.Tensor(init_image)
            instance_image = 255. * instance_image
            instance_image = Image.fromarray(instance_image.astype(np.uint8))

            x_samples = pipe(prompt=args.prompt, image=instance_image,
                             mask_image=Image.fromarray((255. * mask_image).astype(np.uint8)), strength=diffusion_strength,
                             num_inference_steps=num_inference_steps, output_type='',
                             guidance_scale=7.5).images[0]

            new_going_image = torch.from_numpy(x_samples).to(device)

            new_going_masked_image = np.multiply(x_samples, mask_image) + np.multiply(GT_BG, 1. - mask_image)
            new_going_masked_image = torch.from_numpy(new_going_masked_image).to(device)

        return new_going_image, new_going_masked_image, nerf_image_to_save, new_going_mask

    # ---------------------------------------------- #
    # ------------ render and test only ------------ #
    # ---------------------------------------------- #
    # Short circuit if only rendering out from trained model
    if args.render_image:
        rendersavedir = os.path.join(basedir, expname, 'renderings')
        os.makedirs(rendersavedir, exist_ok=True)

        with torch.no_grad():
            for img_i in tqdm(i_train, desc='rendering training views'):
                _, going_masked_image, _, _ \
                    = generate_one_data_point(data_point_id=img_i,
                                                     diffusion_strength=0.01 * float(args.strength_test),
                                                     num_inference_steps=args.test_num_inference_steps)

                fname = fnames_out[img_i]
                fname = os.path.basename(fname)
                fname = str(os.path.splitext(fname)[0])

                going_masked_image = 255. * going_masked_image.cpu().numpy()
                Image.fromarray(going_masked_image.astype(np.uint8)).save(
                    os.path.join(rendersavedir, fname + '.png'))

    if args.render_video:
        def interp_t(pose_first, pose_second, num_Gaps, t):
            if t > num_Gaps:
                raise Exception('t cannot be larger than num_Gaps !')
            pose_first = pose_first
            rotation_first = pose_first[:, :3]
            translation_first = pose_first[:, 3:]

            pose_second = pose_second
            rotation_second = pose_second[:, :3]
            translation_second = pose_second[:, 3:]

            key_rots = R.from_matrix([rotation_first, rotation_second])
            key_times = [0, num_Gaps]
            slerp = Slerp(key_times, key_rots)

            times = list(range(0, num_Gaps + 1))
            interp_rots = slerp(times)
            interp_rots = interp_rots.as_matrix()

            interp_trans = []
            for this_time in times:
                interp_trans.append(
                    translation_first + (translation_second - translation_first) * this_time / num_Gaps)

            interp_matirces = np.concatenate((interp_rots, interp_trans), axis=2)
            pose_mid = interp_matirces[t]

            return pose_mid

        def cal_bezier_curve(pose_start, pose_mid, pose_end, num_Gaps, t):
            startmid_term = interp_t(pose_start, pose_mid, num_Gaps, t)
            midend_term = interp_t(pose_mid, pose_end, num_Gaps, t)
            pose_cur = interp_t(startmid_term, midend_term, num_Gaps, t)
            return pose_cur

        poses_video = []
        for i, pose_video_fname in enumerate(args.video_frames):
            for j, fname_out in enumerate(fnames_out):
                if pose_video_fname in fname_out:
                    poses_video.append(poses[j, :3, :4].cpu().numpy())
                    break

        num_Gaps = int(args.num_Gaps // 2 * 2)

        poses_interpolate = []
        for i in range(len(poses_video)):
            poses_interpolate.append(interp_t(poses_video[i], poses_video[(i+1) % len(poses_video)], num_Gaps, int(num_Gaps/2)))

        videooutsavedir = os.path.join(basedir, expname, 'videos', args.video_expname)
        os.makedirs(videooutsavedir, exist_ok=True)

        count = 0
        videotheseframesdir = []
        for i in tqdm(range(len(poses_video)), desc='rendering video'):
            for t in tqdm(range(num_Gaps), leave=False):
                pose_cur = cal_bezier_curve(poses_interpolate[i],
                                 poses_video[(i+1) % len(poses_video)],
                                 poses_interpolate[(i+1) % len(poses_video)],
                                 num_Gaps, t)

                with torch.no_grad():
                    init_image, _ = render_path_train(torch.Tensor(pose_cur).to(device), hwf, K, args.chunk,
                                                      render_kwargs_train)
                    init_image = init_image[0]
                    init_image = init_image.cpu().numpy()

                    mask_image = np.zeros_like(init_image)
                    frame_wh_list = world2frame(world_xyz_list=world_xyz_list, K=K,c2w=torch.tensor(pose_cur))
                    for face_index in face_index_list:
                        contour = [np.array(
                            [
                                frame_wh_list[face_index[0]],
                                frame_wh_list[face_index[1]],
                                frame_wh_list[face_index[2]],
                                frame_wh_list[face_index[3]]
                            ]
                            , dtype=np.int32)]
                        cv2.drawContours(mask_image, contour, 0, (255, 255, 255), -1)

                    mask_image = Image.fromarray(mask_image.astype(np.uint8))

                    init_image = 255. * init_image
                    init_image = Image.fromarray(init_image.astype(np.uint8))

                    strength_video = args.strength_video / 100.
                    init_image = pipe(prompt=args.prompt, image=init_image,
                                     mask_image=mask_image, strength=strength_video,
                                     num_inference_steps=50, output_type='',
                                     guidance_scale=7.5).images[0]

                    init_image = 255. * init_image
                    videothisframedir = os.path.join(videooutsavedir, str(count).zfill(3) + '.png')
                    Image.fromarray(init_image.astype(np.uint8)).save(videothisframedir)

                    videotheseframesdir.append(videothisframedir)
                    count += 1

        frames = []
        for i in range(len(videotheseframesdir)):
            frames.append(cv2.imread(videotheseframesdir[i]))

        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(os.path.join(videooutsavedir, 'video.mp4'), cv2.VideoWriter_fourcc(*'MP4V'), 11,
                              (width, height))
        for frame in frames:
            out.write(frame)
        out.release()

    if args.render_image or args.render_video:
        return

    # ------------------------------------------------- #
    # --------- prepare to start the training --------- #
    # ------------------------------------------------- #
    N_iters = args.initial_iter \
              + args.extra_train_on_pivot_iter \
              + (1 + (math.ceil((len(i_train)-1)/2))) * args.new_far_image_iters
    print('Begin')

    loss_list = []
    time_list = []
    start = start + 1
    time0 = time.time()

    # initialize the going dataset
    going_images_pool = [torch.Tensor(image).to(device) for image in images]  # for visualization
    going_masked_images_pool = [torch.Tensor(image).to(device) for image in images]  # for optimization
    going_nerf_image_pool = [torch.zeros_like(torch.Tensor(image)).to(device) for image in images]  # for visualization, black if not updated
    going_masks_pool = [torch.zeros_like(torch.Tensor(image)).to(device) for image in images]  # for visualization, red if not updated
    for i in range(len(going_masks_pool)):
        going_masks_pool[i][:, :, 0] += 1

    going_near2far_until_index = -args.num_Neighbors  # use random.randint to select node

    # learning scheduler, about training stage; training num of steps; training diffusion strength; etc.
    cur_stage = 'initial'
    remaining_iter = 0
    remaining_operation_iter = 0

    for i in trange(start, N_iters):
        # -------------------------------------- #
        # --------- update the dataset --------- #
        # -------------------------------------- #
        if i > args.initial_iter:
            if remaining_operation_iter <= 0 and going_near2far_until_index > 0:
                remaining_operation_iter = args.operation_iter
                random_data_point_id = done_nodes_for_pseudo[random.randint(0, going_near2far_until_index)]

                cur_strength = 0.01 * float(random.randint(args.strength_lower_bound, args.strength_higher_bound))

                new_going_image, new_going_masked_image, \
                nerf_image_to_save, new_going_mask \
                    = generate_one_data_point(data_point_id=random_data_point_id,
                                                     diffusion_strength=cur_strength)

                going_images_pool[random_data_point_id] = new_going_image
                going_masked_images_pool[random_data_point_id] = new_going_masked_image
                going_nerf_image_pool[random_data_point_id] = nerf_image_to_save
                going_masks_pool[random_data_point_id] = new_going_mask

            if remaining_iter <= 0:
                cur_stage = 'fusion'

                if going_near2far_until_index < 0:
                    cur_strength = 1.0
                else:
                    cur_strength = 0.01 * float(random.randint(args.strength_lower_bound, args.strength_higher_bound))

                going_near2far_until_index_last = going_near2far_until_index
                going_near2far_until_index = min(going_near2far_until_index + args.num_Neighbors, len(i_train)-1)

                for sdlkf in range(max(0, going_near2far_until_index_last+1), going_near2far_until_index+1):
                    data_point_id = done_nodes_for_pseudo[sdlkf]

                    new_going_image, new_going_masked_image, \
                    nerf_image_to_save, new_going_mask \
                        = generate_one_data_point(data_point_id=data_point_id,
                                                         diffusion_strength=cur_strength)

                    going_images_pool[data_point_id] = new_going_image
                    going_masked_images_pool[data_point_id] = new_going_masked_image
                    going_nerf_image_pool[data_point_id] = nerf_image_to_save
                    going_masks_pool[data_point_id] = new_going_mask

                remaining_iter = args.new_far_image_iters + args.extra_train_on_pivot_iter
                args.extra_train_on_pivot_iter = 0

                print("remaining_iter of this round: ", remaining_iter)

        # ----------------------------------------- #
        # --------- pretrained on BG only --------- #
        # ----------------------------------------- #
        if cur_stage == 'initial':
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3, :4]

            rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

            if i < args.precrop_iters:
                dH = int(H//2 * args.precrop_frac)
                dW = int(W//2 * args.precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                        torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                    ), -1)
                if i == start:
                    print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
            else:
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            #####  Core optimization loop  #####
            rgb, depth, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                    verbose=i < 10, retraw=True,
                                                    **render_kwargs_train)

            optimizer.zero_grad()
            img_loss = img2mse(rgb, target_s)
            loss = img_loss

            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss = loss + img_loss0

            sparsity_loss = args.sparse_loss_weight * (extras["sparsity_loss"].sum() + extras["sparsity_loss0"].sum())
            loss = loss + sparsity_loss

            # add Total Variation loss
            if args.i_embed == 1:
                n_levels = render_kwargs_train["embed_fn"].n_levels
                min_res = render_kwargs_train["embed_fn"].base_resolution
                max_res = render_kwargs_train["embed_fn"].finest_resolution
                log2_hashmap_size = render_kwargs_train["embed_fn"].log2_hashmap_size
                TV_loss = sum(total_variation_loss(render_kwargs_train["embed_fn"].embeddings[i],
                                                   min_res, max_res,
                                                   i, log2_hashmap_size,
                                                   n_levels=n_levels) for i in range(n_levels))
                loss = loss + args.tv_loss_weight * TV_loss
                if i > 1000:
                    args.tv_loss_weight = 0.0

            loss.backward()
            optimizer.step()

            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
            ################################

        # ---------------------------------------------------- #
        # --------- train on diffusion modifications --------- #
        # ---------------------------------------------------- #
        elif cur_stage == 'fusion':
            remaining_iter = remaining_iter - 1
            remaining_operation_iter = remaining_operation_iter - 1

            N_rand = args.N_rand

            img_i = done_nodes_for_pseudo[random.randint(0, going_near2far_until_index)]

            pose = poses[img_i, :3, :4]
            target = going_masked_images_pool[img_i]

            rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

            if i < args.precrop_iters:
                dH = int(H//2 * args.precrop_frac)
                dW = int(W//2 * args.precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                        torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                    ), -1)
                if i == start:
                    print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
            else:
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            #####  Core optimization loop  #####
            rgb, depth, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                    verbose=i < 10, retraw=True,
                                                    **render_kwargs_train)

            optimizer.zero_grad()
            loss = 0

            img_loss = img2mse(rgb, target_s)
            loss = loss + img_loss

            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss = loss + img_loss0

            sparsity_loss = args.sparse_loss_weight * (extras["sparsity_loss"].sum() + extras["sparsity_loss0"].sum())
            loss = loss + sparsity_loss

            # add Total Variation loss
            if args.i_embed == 1:
                n_levels = render_kwargs_train["embed_fn"].n_levels
                min_res = render_kwargs_train["embed_fn"].base_resolution
                max_res = render_kwargs_train["embed_fn"].finest_resolution
                log2_hashmap_size = render_kwargs_train["embed_fn"].log2_hashmap_size
                TV_loss = sum(total_variation_loss(render_kwargs_train["embed_fn"].embeddings[i],
                                                   min_res, max_res,
                                                   i, log2_hashmap_size,
                                                   n_levels=n_levels) for i in range(n_levels))
                loss = loss + args.tv_loss_weight * TV_loss

            loss.backward()
            optimizer.step()

            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            new_lrate = args.lrate_fusion
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
            ################################

        t = time.time()-time0
        #####           end            #####

        # ----------------------------------------------------- #
        # --------- various visualization and logging --------- #
        # ----------------------------------------------------- #
        if i % args.i_weights==0 or i == N_iters-1:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            if args.i_embed==1:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'embed_fn_state_dict': render_kwargs_train['embed_fn'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            print('Saved checkpoints at', path)

        if i % args.i_trainset == 0 or i == N_iters-1:
            if abs(start - i) < (args.i_trainset - 10):
                pass
            else:
                trainsavedir = os.path.join(basedir, expname, '{:06d}_{}'.format(i, cur_stage))
                print('Saving train set')
                os.makedirs(trainsavedir, exist_ok=True)

                with torch.no_grad():
                    for img_i in i_train:
                        pose = poses[img_i, :3, :4]
                        out, _ = render_path_train(torch.Tensor(pose).to(device), hwf, K, args.chunk, render_kwargs_train)

                        out = out[0].cpu().numpy()
                        out = out * 255.

                        fname = fnames_out[img_i]
                        fname = os.path.basename(fname)
                        fname = str(os.path.splitext(fname)[0])

                        Image.fromarray(out.astype(np.uint8)).save(
                            os.path.join(trainsavedir, 'current-nerf_' + fname + '.png'))

                if cur_stage == 'fusion':
                    for img_i in i_train:
                        masked_going_image = going_masked_images_pool[img_i]
                        masked_going_image = 255. * masked_going_image.cpu().numpy()

                        fname = fnames_out[img_i]
                        fname = os.path.basename(fname)
                        fname = str(os.path.splitext(fname)[0])

                        Image.fromarray(masked_going_image.astype(np.uint8)).save(
                            os.path.join(trainsavedir, 'current-dataset_' + fname + '.png'))

        if (i - 1) % args.i_visualization==0 and cur_stage != 'initial':
            vissavedir = os.path.join(basedir, expname, 'visualization')
            os.makedirs(vissavedir, exist_ok=True)

            # save dataset
            final_images = []
            for img_i in done_nodes_for_pseudo[0:going_near2far_until_index+1]:
                going_image = going_images_pool[img_i]
                masked_going_image = going_masked_images_pool[img_i]
                going_BG_image = torch.Tensor(images[img_i]).to(device)
                nerf_image_to_save = going_nerf_image_pool[img_i]
                going_mask = going_masks_pool[img_i]

                going_image = 255. * going_image.cpu().numpy()
                masked_going_image = 255. * masked_going_image.cpu().numpy()
                going_BG_image = 255. * going_BG_image.cpu().numpy()
                nerf_image_to_save = 255. * nerf_image_to_save.cpu().numpy()
                going_mask = 255. * going_mask.cpu().numpy()

                final_image = np.concatenate((going_BG_image, going_mask, nerf_image_to_save, going_image, masked_going_image), axis=0)
                final_images.append(final_image)

            final_final_image = final_images[0]
            for img in final_images[1:]:
                final_final_image = np.concatenate((final_final_image, img), axis=1)

            pil_concat_to_save = Image.fromarray(final_final_image.astype(np.uint8))
            width, height = pil_concat_to_save.size
            if going_near2far_until_index < 20:
                down_size_factor = 1
            elif going_near2far_until_index < 40:
                down_size_factor = 2
            elif going_near2far_until_index < 60:
                down_size_factor = 3
            else:
                down_size_factor = 4
            pil_concat_to_save = pil_concat_to_save.resize((int(width/down_size_factor), int(height/down_size_factor)))
            pil_concat_to_save.save(os.path.join(vissavedir, 'vis_{:06d}'.format(i) + '.png'))

        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Stage: {cur_stage} Iter: {i} Loss: {loss.item()}")
            loss_list.append(loss.item())
            time_list.append(t)

        global_step += 1

    print('Training ends!')


if __name__ == '__main__':
    train()
