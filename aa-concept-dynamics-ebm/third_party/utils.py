# Code from https://github.com/yilundu/ebm_code_release_pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import random
import cv2
import subprocess
import numpy as np

from PIL import Image

from torch.nn.utils import spectral_norm
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import math

def swish(x):
    return x * torch.sigmoid(x)

def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)

def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


class ReplayBuffer(object):
    def __init__(self, size, transform, FLAGS):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

        # def get_color_distortion(s=1.0):
        #     # s is the strength of color distortion.
        #     color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.4*s)
        #     rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        #     rnd_gray = transforms.RandomGrayscale(p=0.2)
        #     color_distort = transforms.Compose([
        #         rnd_color_jitter,
        #         rnd_gray])
        #     return color_distort

        # color_transform = get_color_distortion()

        feat_size = FLAGS.num_timesteps

        self.dataset = FLAGS.dataset
        if transform:
            self.transform = None
        else:
            self.transform = None

    def __len__(self):
        return len(self._storage)

    def add(self, ims):
        batch_size = ims.shape[0]
        if self._next_idx >= len(self._storage):
            self._storage.extend(list(ims))
        else:
            if batch_size + self._next_idx < self._maxsize:
                self._storage[self._next_idx:self._next_idx +
                                             batch_size] = list(ims)
            else:
                split_idx = self._maxsize - self._next_idx
                self._storage[self._next_idx:] = list(ims)[:split_idx]
                self._storage[:batch_size - split_idx] = list(ims)[split_idx:]
        self._next_idx = (self._next_idx + ims.shape[0]) % self._maxsize

    def _encode_sample(self, idxes, no_transform=False, downsample=False):
        ims = []
        for i in idxes:
            im = self._storage[i]
            ims.append(im)
        return np.array(ims)

    def sample(self, batch_size, no_transform=False, downsample=False):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes, no_transform=no_transform, downsample=downsample), idxes

    def set_elms(self, data, idxes):
        if len(self._storage) < self._maxsize:
            self.add(data)
        else:
            for i, ix in enumerate(idxes):
                self._storage[ix] = data[i]

def align_replayed_batch(feat, rep_feat):
    # BS, N, T, F = rep_feat.shape
    ff_loc_0, ff_vel_0  = feat[..., :1, :2],     feat[..., :1, 2:4]
    bf_loc_0, bf_vel_0 =  rep_feat[..., :1, :2], rep_feat[..., :1, 2:4]

    vel_ratio = ff_vel_0 / bf_vel_0
    rep_feat[..., :2] -= bf_loc_0
    rep_feat[..., :4] *= torch.cat([vel_ratio, vel_ratio], dim=-1)
    rep_feat[..., :2] += ff_loc_0

    return rep_feat

def compress_x_mod(x_mod):
    x_mod = (np.clip(x_mod, -1, 1)).astype(np.float16)
    return x_mod

def decompress_x_mod(x_mod):
    x_mod = x_mod  #+ np.random.uniform(-1, 1 / 5, x_mod.shape) # Note: We do not add noise in the first attempt
    return x_mod

def linear_annealing(device, step, start_step, end_step, start_value, end_value):
    """
    Linear annealing

    :param x: original value. Only for getting device
    :param step: current global step
    :param start_step: when to start changing value
    :param end_step: when to stop changing value
    :param start_value: initial value
    :param end_value: final value
    :return:
    """
    if device is not None:
        if step <= start_step:
            x = torch.tensor(start_value, device=device)
        elif start_step < step < end_step:
            slope = (end_value - start_value) / (end_step - start_step)
            x = torch.tensor(start_value + slope * (step - start_step), device=device)
        else:
            x = torch.tensor(end_value, device=device)
    else:
        if step <= start_step:
            x = start_value
        elif start_step < step < end_step:
            slope = (end_value - start_value) / (end_step - start_step)
            x = start_value + slope * (step - start_step)
        else:
            x = end_value

    return x

class GaussianBlur(object):
    def __init__(self, min=0.1, max=2.0, kernel_size=9):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

def init_distributed_mode(params):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - n_nodes
        - node_id
        - local_rank
        - global_rank
        - world_size
    """
    SLURM_VARIABLES = [
        'SLURM_JOB_ID',
        'SLURM_JOB_NODELIST', 'SLURM_JOB_NUM_NODES', 'SLURM_NTASKS', 'SLURM_TASKS_PER_NODE',
        'SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU',
        'SLURM_NODEID', 'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_TASK_PID'
    ]
    PREFIX = "%i - " % int(os.environ['SLURM_PROCID'])
    for name in SLURM_VARIABLES:
        value = os.environ.get(name, None)
        print(PREFIX + "%s: %s" % (name, str(value)))
    # number of nodes / node ID
    params.nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    params.node_rank = int(os.environ['SLURM_NODEID'])
    # define master address and master port
    hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
    params.master_addr = hostnames.split()[0].decode('utf-8')

def visualize_trajectories(state, state_gen, edges, savedir=None, b_idx=0):

    print(edges[b_idx].cpu().detach().numpy())

    # loc_gen = torch.chunk(loc_gen, loc_gen.shape[0], dim=0)
    # vel_gen = torch.chunk(vel_gen, vel_gen.shape[0], dim=0)
    # loc_gen, vel_gen = [[loc.squeeze(0), vel.squeeze(0)] for loc, vel in zip(loc_gen, vel_gen)]

    # TODO: Here

    # lims = [-0.25, -0.25]
    if state is not None:
        plt, fig = get_trajectory_figure(state, b_idx)
        if not savedir:
            plt.show()
        else: fig.savefig(savedir + '_gt.png', dpi=fig.dpi); plt.close()

    if state_gen is not None:
        plt2, fig2 = get_trajectory_figure(state_gen, b_idx)
        if not savedir:
            plt2.show()
        else: fig2.savefig(savedir + '_gen.png', dpi=fig.dpi); plt2.close()

    # if state_gen is not None:
    #     fig2 = plt.figure()
    #     axes = plt.gca()
    #     axes.set_xlim([-1., 1.])
    #     axes.set_ylim([-1., 1.])
    #     state_gen = state_gen[b_idx].permute(0, 2, 3, 1).cpu().detach().numpy()
    #     loc_gen, vel_gen = np.split(state_gen[:, :, :2], state_gen.shape[0], 0), \
    #                        np.split(state_gen[:, :, 2:], state_gen.shape[0], 0)
    #     for i in range(loc.shape[-1]):
    #         plt.plot(loc_gen[-1][0, :, 0, i], loc_gen[-1][0, :, 1, i])
    #         plt.plot(loc_gen[-1][0, 0, 0, i], loc_gen[-1][0, 0, 1, i], 'd')
    #     if not savedir:
    #         plt.show()
    #     else: fig2.savefig(savedir + '_gen.png', dpi=fig.dpi); plt.close()

    # energies = [sim._energy(loc[i, :, :], vel[i, :, :], edges) for i in
    #             range(loc.shape[0])]
    # plt.plot(energies)
    # plt.show()

def gaussian_kernel(a, b):
    # Modification for allowing batch
    bs = a.shape[0]
    dim1_1, dim1_2 = a.shape[1], b.shape[1]
    depth = a.shape[2]
    a = a.reshape(bs, dim1_1, 1, depth)
    b = b.reshape(bs, 1, dim1_2, depth)
    a_core = a.expand(bs, dim1_1, dim1_2, depth)
    b_core = b.expand(bs, dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(3).mean(3)/depth
    return torch.exp(-numerator)

# Implemented from: https://github.com/Saswatm123/MMD-VAE
def batch_MMD(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()

def MMD(latent):
    ls = latent.shape[1]
    randperm = (torch.arange(ls) + torch.randint(1, ls, (1,))) % ls
    latent = latent.permute(1, 0, 2)
    return batch_MMD(latent[randperm], latent)

def augment_trajectories(locvel, rotation=None):
    if rotation is not None:
        if rotation == 'random':
            rotation = random.random() * math.pi * 2
        else: raise NotImplementedError
        locvel = rotate_with_vel(points=locvel, angle=rotation)
    return locvel

def rotate(origin, points, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = points[..., 0:1], points[..., 1:2]

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return torch.cat([qx, qy], dim=-1)

def rotate_with_vel(points, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """

    locs = torch.view_as_complex(points[..., :2])
    locpol = torch.polar(locs.abs(), locs.angle() + angle)
    locs = torch.view_as_real(locpol)

    if points.shape[-1] > 2:
        vels = torch.view_as_complex(points[..., 2:])
        velpol = torch.polar(vels.abs(), vels.angle() + angle)
        vels = torch.view_as_real(velpol)
        feat = torch.cat([locs, vels], dim=-1)
    else: feat = locs
    return feat

def normalize_trajectories(state, augment=False, normalize=True):
    '''
    state: [BS, NO, T, XY VxVy]
    '''

    if augment:
        state = augment_trajectories(state, 'random')

    # loc, vel = state[..., :2], state[..., 2:]
    # loc_mean = torch.mean(loc, dim=(1,2,3), keepdim=True)
    # state = torch.cat([loc - loc_mean, vel], dim=-1)

    if normalize:
        loc, vel = state[..., :2], state[..., 2:]
        ## Instance normalization
        loc_max = torch.amax(loc, dim=(1,2,3), keepdim=True)
        loc_min = torch.amin(loc, dim=(1,2,3), keepdim=True)
        # vel_max = torch.amax(vel, dim=(1,2,3), keepdim=True)
        # vel_min = torch.amin(vel, dim=(1,2,3), keepdim=True)

        ## Batch normalization
        # loc_max = loc.max()
        # loc_min = loc.min()
        # vel_max = vel.max()
        # vel_min = vel.min() #(dim=-2, keepdims=True)[0]

        # Normalize to [-1, 1]
        loc = (loc - loc_min) * 2 / (loc_max - loc_min) - 1

        if state.shape[-1] > 2:
            vel = vel * 2 / (loc_max - loc_min)
            state = torch.cat([loc, vel], dim=-1)
        else: state = loc
    return state

def get_trajectory_figure(state, b_idx, lims=None, plot_type ='loc', highlight_nodes = None):
    fig = plt.figure()
    axes = plt.gca()
    lw = 1.5
    sz_pt = 20
    maps = ['afmhot', 'cool', 'Wistia', 'YlGnBu'] #https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
    cmap = maps[2]
    alpha = 0.8
    if lims is not None:
        axes.set_xlim([lims[0], lims[1]])
        axes.set_ylim([lims[0], lims[1]])
    state = state[b_idx].permute(1, 2, 0).cpu().detach().numpy()
    loc, vel = state[:, :2][None], state[:, 2:][None]
    # vel_norm = np.sqrt((vel ** 2).sum(axis=1))
    colors = ['b', 'r', 'c', 'y', 'k', 'm', 'g']
    if highlight_nodes is not None:
        modes = ['-' if node == 0 else '--' for node in highlight_nodes]
        assert len(modes) == loc.shape[-1]
    else: modes = ['-']*loc.shape[-1]
    if plot_type == 'loc' or plot_type == 'both':
        for i in range(loc.shape[-1]):
            plt.plot(loc[0, :, 0, i], loc[0, :, 1, i], modes[i], c=colors[i], linewidth=lw)
            # plt.plot(loc[0, 0, 0, i], loc[0, 0, 1, i], 'd')
            plt.scatter(loc[0, :, 0, i], loc[0, :, 1, i], s=sz_pt, c=np.arange(loc.shape[1]), cmap=cmap, alpha=alpha)
        pass
    if plot_type == 'vel' or plot_type == 'both':
        for i in range(loc.shape[-1]):
            acc_pos = loc[0, 0:1, 0, i], loc[0, 0:1, 1, i]
            vels = vel[0, :, 0, i], vel[0, :, 1, i]
            for t in range(loc.shape[1] - 1):
                acc_pos = np.concatenate([acc_pos[0], acc_pos[0][t:t+1]+vels[0][t:t+1]]), \
                          np.concatenate([acc_pos[1], acc_pos[1][t:t+1]+vels[1][t:t+1]])
            plt.plot(acc_pos[0], acc_pos[1], modes[i], c=colors[i], linewidth=lw)
            # plt.plot(loc[0, 0, 0, i], loc[0, 0, 1, i], 'd')
            plt.scatter(loc[0, :, 0, i], loc[0, :, 1, i], s=sz_pt, c=np.arange(loc.shape[1]), cmap=cmap, alpha=alpha)
    return plt, fig

def accumulate_traj(states):
    loc, vel = states[..., :2], states[..., 2:]
    acc_pos = loc[:, :, 0:1]
    for t in range(loc.shape[2] - 1):
        acc_pos = torch.cat([acc_pos, acc_pos[:, :, t:t+1] + vel[:, :, t:t+1]], dim=2)
    return torch.cat([acc_pos, vel], dim=-1)

def get_rel_pairs(rel_send, rel_rec):
    # o = torch.arange(3).to(inputs.device)[None, :, None].type(torch.cuda.FloatTensor) ### To test node relations
    # edges = self.node2edge(o, rel_rec, rel_send)
    # print(rel_send + rel_rec)
    ne, nn = rel_send.shape[1:]
    a = np.arange(ne)
    rel = (rel_send + rel_rec).detach().cpu().numpy()[0]
    unique_values = np.unique(rel, axis=0)
    group_list = []
    for value in unique_values:
        this_group = []
        for i in range(ne):
            if all(rel[i] == value):
                this_group.append(a[i])
        group_list.append(this_group)
    return group_list

def save_rel_matrices(model, rel_rec, rel_send):
    if model.rel_rec is None and model.rel_send is None:
        model.rel_rec, model.rel_send = rel_rec[0:1], rel_send[0:1]

def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return (w/w.sum())#.double()

gkern = None
def smooth_trajectory(x, kernel_size, std, interp_size = 100):

    x = x.permute(0,1,3,2)
    traj_shape = x.shape
    x = x.flatten(0,2)
    x_in = F.interpolate(x[:, None, :], interp_size, mode='linear')
    global gkern
    if gkern is None:
        gkern = gaussian_fn(kernel_size, std).to(x_in.device)
    x_in_sm = F.conv1d(x_in, weight=gkern[None, None], padding=kernel_size//2) # padding? double?
    x_sm = F.interpolate(x_in_sm, traj_shape[-1], mode='linear')

    # visualize
    # import matplotlib.pyplot as plt
    # plt.close();plt.plot(x_sm[0,0].cpu().detach().numpy());plt.plot(x[0].cpu().detach().numpy()); plt.show()
    return x_sm.reshape(traj_shape).permute(0,1,3,2)

def get_model_grad_norm(models):
    parameters = [p for p in models[0].parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        total_norm = 0.0
    else:
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(device) for p in parameters]), 2.0).item()
    return total_norm

def get_model_grad_max(models):
    parameters = [abs(p.grad.detach()).max() for p in models[0].parameters() if p.grad is not None and p.requires_grad]
    return max(parameters)

def create_masks(FLAGS, dev):
    if FLAGS.masking_type == 'random':
        mask = torch.randint(2, (FLAGS.batch_size, FLAGS.components)).to(dev)
    elif FLAGS.masking_type == 'ones':
        mask = torch.ones((FLAGS.batch_size, FLAGS.components)).to(dev)
    elif FLAGS.masking_type == 'by_receiver':
        mask = torch.ones((FLAGS.batch_size, FLAGS.components)).to(dev)
        node_ids = torch.randint(FLAGS.n_objects, (FLAGS.batch_size,))
        sel_edges = (FLAGS.n_objects - 1)*node_ids
        for n in range(FLAGS.n_objects-1):
            mask[torch.arange(0, FLAGS.batch_size), sel_edges+n] = 0
    else: raise NotImplementedError
    return mask
# class Downsample(nn.Module):
#     def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
#         super(Downsample, self).__init__()
#         self.filt_size = filt_size
#         self.pad_off = pad_off
#         self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
#         self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
#         self.stride = stride
#         self.off = int((self.stride-1)/2.)
#         self.channels = channels
#
#         if self.filt_size == 1:
#             a = np.array([1., ])
#         elif self.filt_size == 2:
#             a = np.array([1., 1.])
#         elif self.filt_size == 3:
#             a = np.array([1., 2., 1.])
#         elif self.filt_size == 4:
#             a = np.array([1., 3., 3., 1.])
#         elif self.filt_size == 5:
#             a = np.array([1., 4., 6., 4., 1.])
#         elif self.filt_size == 6:
#             a = np.array([1., 5., 10., 10., 5., 1.])
#         elif self.filt_size == 7:
#             a = np.array([1., 6., 15., 20., 15., 6., 1.])
#         else:
#             raise ValueError(f'invalid filt size: {self.filt_size}')
#
#         filt = torch.Tensor(a[:, None] * a[None, :])
#         filt = filt/torch.sum(filt)
#         self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels,1,1,1)))
#
#         self.pad = get_pad_layer(pad_type)(self.pad_sizes)
#
#     def forward(self, inp):
#         if self.filt_size == 1:
#             if self.pad_off == 0:
#                 return inp[:, :, ::self.stride, ::self.stride]
#             else:
#                 return self.pad(inp)[:, :, ::self.stride, ::self.stride]
#         else:
#             return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])
#
#
# def get_pad_layer(pad_type):
#     if pad_type in ['refl', 'reflect']:
#         PadLayer = nn.ReflectionPad2d
#     elif pad_type in ['repl', 'replicate']:
#         PadLayer = nn.ReplicationPad2d
#     elif pad_type == 'zero':
#         PadLayer = nn.ZeroPad2d
#     else:
#         raise ValueError('Pad type [%s] not recognized' % pad_type)
#     return PadLayer
#
#
# class Downsample1D(nn.Module):
#     def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
#         super(Downsample1D, self).__init__()
#         self.filt_size = filt_size
#         self.pad_off = pad_off
#         self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
#         self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
#         self.stride = stride
#         self.off = int((self.stride - 1) / 2.)
#         self.channels = channels
#
#         # print('Filter size [%i]' % filt_size)
#         if self.filt_size == 1:
#             a = np.array([1., ])
#         elif self.filt_size == 2:
#             a = np.array([1., 1.])
#         elif self.filt_size == 3:
#             a = np.array([1., 2., 1.])
#         elif self.filt_size == 4:
#             a = np.array([1., 3., 3., 1.])
#         elif self.filt_size == 5:
#             a = np.array([1., 4., 6., 4., 1.])
#         elif self.filt_size == 6:
#             a = np.array([1., 5., 10., 10., 5., 1.])
#         elif self.filt_size == 7:
#             a = np.array([1., 6., 15., 20., 15., 6., 1.])
#         else:
#             raise ValueError(f'invalid filt size: {self.filt_size}')
#
#         filt = torch.Tensor(a)
#         filt = filt / torch.sum(filt)
#         self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))
#
#         self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)
#
#     def forward(self, inp):
#         if self.filt_size == 1:
#             if self.pad_off == 0:
#                 return inp[:, :, ::self.stride]
#             else:
#                 return self.pad(inp)[:, :, ::self.stride]
#         else:
#             return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])
#
#
# def get_pad_layer_1d(pad_type):
#     if pad_type in ['refl', 'reflect']:
#         PadLayer = nn.ReflectionPad1d
#     elif pad_type in ['repl', 'replicate']:
#         PadLayer = nn.ReplicationPad1d
#     elif pad_type == 'zero':
#         PadLayer = nn.ZeroPad1d
#     else:
#         raise ValueError('Pad type [%s] not recognized' % pad_type)
#     return PadLayer
#
#
# class Self_Attn(nn.Module):
#     """ Self attention Layer"""
#     def __init__(self,in_dim,activation):
#         super(Self_Attn,self).__init__()
#         self.chanel_in = in_dim
#         self.activation = activation
#
#         self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
#         self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
#         self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#         self.softmax  = nn.Softmax(dim=-1) #
#
#     def forward(self,x):
#         """
#             inputs :
#                 x : input feature maps( B X C X W X H)
#             returns :
#                 out : self attention value + input feature
#                 attention: B X N X N (N is Width*Height)
#         """
#         m_batchsize,C,width ,height = x.size()
#         proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
#         proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
#         energy =  torch.bmm(proj_query,proj_key) # transpose check
#         attention = self.softmax(energy) # BX (N) X (N)
#         proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
#
#         out = torch.bmm(proj_value,attention.permute(0,2,1) )
#         out = out.view(m_batchsize,C,width,height)
#
#         out = self.gamma*out + x
#         return out,attention
#
#
# class CondResBlock(nn.Module):
#     def __init__(self, args, downsample=True, rescale=True, filters=64, latent_dim=64, im_size=64, classes=512, norm=True, spec_norm=False, no_res=False):
#         super(CondResBlock, self).__init__()
#
#         self.filters = filters
#         self.latent_dim = latent_dim
#         self.im_size = im_size
#         self.downsample = downsample
#         self.no_res = no_res
#
#         if filters <= 128:
#             # self.bn1 = nn.InstanceNorm2d(filters, affine=True)
#             self.bn1 = nn.GroupNorm(int(filters / 128 * 32), filters, affine=True)
#         else:
#             self.bn1 = nn.GroupNorm(32, filters)
#
#         if not norm:
#             self.bn1 = None
#
#         self.args = args
#
#         if spec_norm:
#             self.conv1 = spectral_norm(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1))
#         else:
#             self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
#
#         if filters <= 128:
#             self.bn2 = nn.GroupNorm(int(filters / 128 * 32), filters, affine=True)
#         else:
#             self.bn2 = nn.GroupNorm(32, filters, affine=True)
#
#         if not norm:
#             self.bn2 = None
#
#         if spec_norm:
#             self.conv2 = spectral_norm(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1))
#         else:
#             self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
#
#         self.dropout = nn.Dropout(0.2)
#
#         # Upscale to an mask of image
#         self.latent_map = nn.Linear(classes, 2*filters)
#         self.latent_map_2 = nn.Linear(classes, 2*filters)
#
#         self.relu = torch.nn.ReLU(inplace=True)
#         self.act = swish
#
#         # Upscale to mask of image
#         if downsample:
#             if rescale:
#                 self.conv_downsample = nn.Conv2d(filters, 2 * filters, kernel_size=3, stride=1, padding=1)
#
#                 if args.alias:
#                     self.avg_pool = Downsample(channels=2*filters)
#                 else:
#                     self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
#             else:
#                 self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
#
#                 if args.alias:
#                     self.avg_pool = Downsample(channels=filters)
#                 else:
#                     self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
#
#     def forward(self, x, y):
#         x_orig = x
#
#         if y is not None:
#             latent_map = self.latent_map(y).view(-1, 2*self.filters, 1, 1)
#
#             gain = latent_map[:, :self.filters]
#             bias = latent_map[:, self.filters:]
#
#         x = self.conv1(x)
#
#         if self.bn1 is not None:
#             x = self.bn1(x)
#
#         if y is not None:
#             x = gain * x + bias
#
#         x = self.act(x)
#         # x = self.dropout(x)
#
#         if y is not None:
#             latent_map = self.latent_map_2(y).view(-1, 2*self.filters, 1, 1)
#             gain = latent_map[:, :self.filters]
#             bias = latent_map[:, self.filters:]
#
#         x = self.conv2(x)
#
#         if self.bn2 is not None:
#             x = self.bn2(x)
#
#         if y is not None:
#             x = gain * x + bias
#
#         x = self.act(x)
#
#         if not self.no_res:
#             x_out = x_orig + x
#         else:
#             x_out = x
#
#         if self.downsample:
#             x_out = self.conv_downsample(x_out)
#             x_out = self.act(self.avg_pool(x_out))
#
#         return x_out
