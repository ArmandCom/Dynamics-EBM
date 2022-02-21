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
from torchvision import transforms
import matplotlib.pyplot as plt


def swish(x):
    return x * torch.sigmoid(x)

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


class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        if self.filt_size == 1:
            a = np.array([1., ])
        elif self.filt_size == 2:
            a = np.array([1., 1.])
        elif self.filt_size == 3:
            a = np.array([1., 2., 1.])
        elif self.filt_size == 4:
            a = np.array([1., 3., 3., 1.])
        elif self.filt_size == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif self.filt_size == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif self.filt_size == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
        else:
            raise ValueError(f'invalid filt size: {self.filt_size}')

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer(pad_type):
    if pad_type in ['refl', 'reflect']:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ['repl', 'replicate']:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == 'zero':
        PadLayer = nn.ZeroPad2d
    else:
        raise ValueError('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class Downsample1D(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        # print('Filter size [%i]' % filt_size)
        if self.filt_size == 1:
            a = np.array([1., ])
        elif self.filt_size == 2:
            a = np.array([1., 1.])
        elif self.filt_size == 3:
            a = np.array([1., 2., 1.])
        elif self.filt_size == 4:
            a = np.array([1., 3., 3., 1.])
        elif self.filt_size == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif self.filt_size == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif self.filt_size == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
        else:
            raise ValueError(f'invalid filt size: {self.filt_size}')

        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))

        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride]
        else:
            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer_1d(pad_type):
    if pad_type in ['refl', 'reflect']:
        PadLayer = nn.ReflectionPad1d
    elif pad_type in ['repl', 'replicate']:
        PadLayer = nn.ReplicationPad1d
    elif pad_type == 'zero':
        PadLayer = nn.ZeroPad1d
    else:
        raise ValueError('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out,attention


class CondResBlock(nn.Module):
    def __init__(self, args, downsample=True, rescale=True, filters=64, latent_dim=64, im_size=64, classes=512, norm=True, spec_norm=False, no_res=False):
        super(CondResBlock, self).__init__()

        self.filters = filters
        self.latent_dim = latent_dim
        self.im_size = im_size
        self.downsample = downsample
        self.no_res = no_res

        if filters <= 128:
            # self.bn1 = nn.InstanceNorm2d(filters, affine=True)
            self.bn1 = nn.GroupNorm(int(filters / 128 * 32), filters, affine=True)
        else:
            self.bn1 = nn.GroupNorm(32, filters)

        if not norm:
            self.bn1 = None

        self.args = args

        if spec_norm:
            self.conv1 = spectral_norm(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1))
        else:
            self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        if filters <= 128:
            self.bn2 = nn.GroupNorm(int(filters / 128 * 32), filters, affine=True)
        else:
            self.bn2 = nn.GroupNorm(32, filters, affine=True)

        if not norm:
            self.bn2 = None

        if spec_norm:
            self.conv2 = spectral_norm(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1))
        else:
            self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout(0.2)

        # Upscale to an mask of image
        self.latent_map = nn.Linear(classes, 2*filters)
        self.latent_map_2 = nn.Linear(classes, 2*filters)

        self.relu = torch.nn.ReLU(inplace=True)
        self.act = swish

        # Upscale to mask of image
        if downsample:
            if rescale:
                self.conv_downsample = nn.Conv2d(filters, 2 * filters, kernel_size=3, stride=1, padding=1)

                if args.alias:
                    self.avg_pool = Downsample(channels=2*filters)
                else:
                    self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
            else:
                self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

                if args.alias:
                    self.avg_pool = Downsample(channels=filters)
                else:
                    self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x, y):
        x_orig = x

        if y is not None:
            latent_map = self.latent_map(y).view(-1, 2*self.filters, 1, 1)

            gain = latent_map[:, :self.filters]
            bias = latent_map[:, self.filters:]

        x = self.conv1(x)

        if self.bn1 is not None:
            x = self.bn1(x)

        if y is not None:
            x = gain * x + bias

        x = self.act(x)
        # x = self.dropout(x)

        if y is not None:
            latent_map = self.latent_map_2(y).view(-1, 2*self.filters, 1, 1)
            gain = latent_map[:, :self.filters]
            bias = latent_map[:, self.filters:]

        x = self.conv2(x)

        if self.bn2 is not None:
            x = self.bn2(x)

        if y is not None:
            x = gain * x + bias

        x = self.act(x)

        if not self.no_res:
            x_out = x_orig + x
        else:
            x_out = x

        if self.downsample:
            x_out = self.conv_downsample(x_out)
            x_out = self.act(self.avg_pool(x_out))

        return x_out


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

def normalize_trajectories(state):
    '''
    state: [BS, NO, T, XY VxVy]
    '''
    loc, vel = state[..., :2], state[..., 2:]
    loc_max = loc.max()
    loc_min = loc.min()
    vel_max = vel.max()
    vel_min = vel.min() #(dim=-2, keepdims=True)[0]

    # Normalize to [-1, 1]
    loc = (loc - loc_min) * 2 / (loc_max - loc_min) - 1
    vel = (vel - vel_min) * 2 / (vel_max - vel_min) - 1

    state = torch.cat([loc, vel], dim=-1)
    return state

def get_trajectory_figure(state, b_idx, lims=None, plot_type ='loc'):
    fig = plt.figure()
    axes = plt.gca()
    if lims is not None:
        axes.set_xlim([lims[0], lims[1]])
        axes.set_ylim([lims[0], lims[1]])
    state = state[b_idx].permute(1, 2, 0).cpu().detach().numpy()
    loc, vel = state[:, :2][None], state[:, 2:][None]
    vel_norm = np.sqrt((vel ** 2).sum(axis=1))
    if plot_type == 'loc' or plot_type == 'both':
        for i in range(loc.shape[-1]):
            plt.plot(loc[0, :, 0, i], loc[0, :, 1, i])
            plt.plot(loc[0, 0, 0, i], loc[0, 0, 1, i], 'd')
        pass
    if plot_type == 'vel' or plot_type == 'both':
        for i in range(loc.shape[-1]):
            acc_pos = loc[0, 0:1, 0, i], loc[0, 0:1, 1, i]
            vels = vel[0, :, 0, i], vel[0, :, 1, i]
            for t in range(loc.shape[1] - 1):
                acc_pos = np.concatenate([acc_pos[0], acc_pos[0][t:t+1]+vels[0][t:t+1]]), \
                          np.concatenate([acc_pos[1], acc_pos[1][t:t+1]+vels[1][t:t+1]])
            plt.plot(acc_pos[0], acc_pos[1])
            plt.plot(loc[0, 0, 0, i], loc[0, 0, 1, i], 'd')

    return plt, fig

def accumulate_traj(states):
    loc, vel = states[..., :2], states[..., 2:]
    acc_pos = loc[:, :, 0:1]
    for t in range(loc.shape[2] - 1):
        acc_pos = torch.cat([acc_pos, acc_pos[:, :, t:t+1] + vel[:, :, t:t+1]], dim=2)
    return torch.cat([acc_pos, vel], dim=-1)