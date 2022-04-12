import torch
import time
from models import EdgeGraphEBM_OneStep, EdgeGraphEBM_CNNOneStep, EdgeGraphEBM_LateFusion, EdgeGraphEBM_CNN_OS_noF # TrajGraphEBM, EdgeGraphEBM, LatentEBM, ToyEBM, BetaVAE_H, LatentEBM128
from scipy.linalg import toeplitz
# from tensorflow.python.platform import flags
import numpy as np
import torch.nn.functional as F
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from dataset import ChargedParticlesSim, ChargedParticles, SpringsParticles, TrajectoryDataset
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# from easydict import EasyDict
import os.path as osp
# from torch.nn.utils import clip_grad_norm
# import numpy as np
# from imageio import imwrite
# import cv2
import argparse
# import pdb
# from torchvision.datasets import ImageFolder
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torch.optim as optim
# from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import random
# from torchvision.utils import make_grid
# from imageio import get_writer
from third_party.utils import visualize_trajectories, get_trajectory_figure, \
    linear_annealing, ReplayBuffer, compress_x_mod, decompress_x_mod, accumulate_traj, \
    normalize_trajectories, augment_trajectories, MMD, align_replayed_batch, get_rel_pairs
from pathlib import Path

homedir = '/data/Armand/EBM/'
# port = 6021
sample_diff = False

"""Parse input arguments"""
parser = argparse.ArgumentParser(description='Train EBM model')
parser.add_argument('--train', action='store_true', help='whether or not to train')
parser.add_argument('--test_manipulate', action='store_true', help='whether or not to train')
parser.add_argument('--cuda', default=True, action='store_true', help='whether to use cuda or not')
parser.add_argument('--port', default=6010, type=int, help='Port for distributed')

parser.add_argument('--single', action='store_true', help='test overfitting of the dataset')


parser.add_argument('--dataset', nargs='+', type=str, help='Dataset to use (intphys or others or imagenet or cubes)')
parser.add_argument('--logdir', default='/data/Armand/EBM/cachedir', type=str, help='location where log of experiments will be stored')
parser.add_argument('--logname', default='', type=str, help='name of logs')
parser.add_argument('--exp', default='default', type=str, help='name of experiments')

# arguments for NRI springs/charged dataset
parser.add_argument('--n_objects', default=3, type=int, help='Dataset to use (intphys or others or imagenet or cubes)')
parser.add_argument('--sequence_length', type=int, default=5000, help='Length of trajectory.')
parser.add_argument('--sample_freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--noise_var', type=float, default=0.0, help='Variance of the noise if present.')
parser.add_argument('--interaction_strength', type=float, default=1., help='Size of the box')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

# training
parser.add_argument('--resume_iter', nargs='+', type=int, help='iteration to resume training')
parser.add_argument('--resume_name',  nargs='+', type=str, help='name of the model to resume')
parser.add_argument('--batch_size', default=64, type=int, help='size of batch of input to use')
parser.add_argument('--num_epoch', default=10000, type=int, help='number of epochs of training to run')
parser.add_argument('--lr', default=3e-4, type=float, help='learning rate for training')
parser.add_argument('--log_interval', default=100, type=int, help='log outputs every so many batches')
parser.add_argument('--save_interval', default=2000, type=int, help='save outputs every so many batches')
parser.add_argument('--autoencode', action='store_true', help='if set to True we use L2 loss instead of Contrastive Divergence')
# parser.add_argument('--forecast', action='store_true', help='if set to True we use L2 loss instead of Contrastive Divergence')
parser.add_argument('--forecast', default=-1, type=int, help='forecast N steps in the future (the encoder only sees the previous). -1 invalidates forecasting')
parser.add_argument('--cd_and_ae', action='store_true', help='if set to True we use L2 loss and Contrastive Divergence')
parser.add_argument('--cd_mode', default='', type=str, help='chooses between options for energy-based objectives. zeros: zeroes out some of the latents. mix: mixes latents of other samples in the batch.')

# data
parser.add_argument('--data_workers', default=4, type=int, help='Number of different data workers to load data in parallel')
parser.add_argument('--ensembles', default=1, type=int, help='use an ensemble of models')

# EBM specific settings

# Model specific settings
parser.add_argument('--filter_dim', default=64, type=int, help='number of filters to use')
# parser.add_argument('--components', default=2, type=int, help='number of components to explain an image with')
parser.add_argument('--component_weight', action='store_true', help='optimize for weights of the components also')
parser.add_argument('--tie_weight', action='store_true', help='tie the weights between seperate models')
parser.add_argument('--optimize_mask', action='store_true', help='also optimize a segmentation mask over image')
parser.add_argument('--pos_embed', action='store_true', help='add a positional embedding to model')
parser.add_argument('--spatial_feat', action='store_true', help='use spatial latents for object segmentation')
parser.add_argument('--dropout', default=0.1, type=float, help='use spatial latents for object segmentation')
parser.add_argument('--factor_encoder', action='store_true', help='if we use message passing in the encoder')
parser.add_argument('--normalize_data_latent', action='store_true', help='if we normalize data before encoding the latents')
parser.add_argument('--obj_id_embedding', action='store_true', help='add object identifier')

# Model specific - TrajEBM
parser.add_argument('--input_dim', default=4, type=int, help='dimension of an object')
parser.add_argument('--latent_hidden_dim', default=64, type=int, help='hidden dimension of the latent')
parser.add_argument('--latent_dim', default=64, type=int, help='dimension of the latent')
parser.add_argument('--obj_id_dim', default=6, type=int, help='size of the object id embedding')
parser.add_argument('--num_fixed_timesteps', default=5, type=int, help='constraints')
parser.add_argument('--num_timesteps', default=19, type=int, help='constraints')

parser.add_argument('--num_steps', default=10, type=int, help='Steps of gradient descent for training')
parser.add_argument('--num_steps_test', default=40, type=int, help='Steps of gradient descent for training')
parser.add_argument('--num_visuals', default=16, type=int, help='Number of visuals')
parser.add_argument('--num_additional', default=0, type=int, help='Number of additional components to add')
parser.add_argument('--plot_attr', default='loc', type=str, help='number of gpus per nodes')

## Options
parser.add_argument('--kl', action='store_true', help='Whether we compute the KL component of the CD loss')
parser.add_argument('--kl_coeff', default=0.2, type=float, help='Coefficient multiplying the KL term in the loss')
parser.add_argument('--sm', action='store_true', help='Whether we compute the smoothness component of the CD loss')
parser.add_argument('--sm_coeff', default=1, type=float, help='Coefficient multiplying the Smoothness term in the loss')
parser.add_argument('--spectral_norm', action='store_true', help='Spectral normalization in ebm')
parser.add_argument('--momentum', default=0.0, type=float, help='Momenum update for Langevin Dynamics')
parser.add_argument('--sample_ema', action='store_true', help='If set to True, we sample from the ema model')
parser.add_argument('--replay_batch', action='store_true', help='If set to True, we initialize generation from a buffer.')
parser.add_argument('--buffer_size', default=10000, type=int, help='Size of the buffer')
parser.add_argument('--entropy_nn', action='store_true', help='If set to True, we add an entropy component to the loss')
parser.add_argument('--mmd', action='store_true', help='If set to True, we add a pairwise mmd component to the loss')

parser.add_argument('--step_lr', default=500.0, type=float, help='step size of latents')
parser.add_argument('--step_lr_decay_factor', default=1.0, type=float, help='step size of latents')
parser.add_argument('--noise_coef', default=0.0, type=float, help='step size of latents')
parser.add_argument('--noise_decay_factor', default=1.0, type=float, help='step size of latents')
parser.add_argument('--ns_iteration_end', default=200000, type=int, help='training iteration where the number of sampling steps reach their max.')
parser.add_argument('--num_steps_end', default=-1, type=int, help='number of sampling steps')

parser.add_argument('--sample', action='store_true', help='generate negative samples through Langevin')
parser.add_argument('--decoder', action='store_true', help='decoder for model')

# Distributed training hyperparameters
parser.add_argument('--nodes', default=1, type=int, help='number of nodes for training')
parser.add_argument('--gpus', default=1, type=int, help='number of gpus per nodes')
parser.add_argument('--node_rank', default=0, type=int, help='rank of node')
parser.add_argument('--gpu_rank', default=0, type=int, help='number of gpus per nodes')


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

def ema_model(models, models_ema, mu=0.99):
    for model, model_ema in zip(models, models_ema):
        for param, param_ema in zip(model.parameters(), model_ema.parameters()):
            param_ema.data = mu * param_ema.data + (1 - mu) * param.data

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

def average_gradients(models):
    size = float(dist.get_world_size())

    for model in models:
        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size

def gen_recursive(latent, FLAGS, models, feat_neg, feat, ini_timesteps=None, stride=1, stage_steps=[10, 3]):
    """
    Samples trajectory auto-regressively in test.

    ini_timesteps: number of timesteps sampled at once the first time
    stride: how many more samples add at each iteration
    stage_steps: How many steps does the sampling take at each iteration. If list, it should be of length 2 and contain:
        The initial number of steps and the number of steps at each iteration after.
    Note: We might not sample all trajectory if max_len is not divisible by stride
    """

    max_len = feat.shape[2]
    num_fixed_timesteps_old = FLAGS.num_fixed_timesteps
    if ini_timesteps is None:
        ini_timesteps = num_fixed_timesteps_old + 1

    feat_in = feat[:, :, :num_fixed_timesteps_old]
    feat_negs_all =[]
    for t in range(ini_timesteps, max_len, stride):
        feat_neg_in = feat_neg[:, :, :t]

        if isinstance(stage_steps, list):
            # assert len(stage_steps) == 3
            if t == ini_timesteps:
                stage_step = stage_steps[0]
            elif t >= max_len - stride:
                stage_step = stage_steps[2]
                feat_neg_in = feat_neg_out
                FLAGS.num_fixed_timesteps = num_fixed_timesteps_old
            else: stage_step = stage_steps[1]
        elif isinstance(stage_steps, int): stage_step = stage_steps
        else: raise NotImplementedError

        feat_neg_out, feat_negs, feat_neg_kl, feat_grad = gen_trajectories(latent, FLAGS, models, feat_neg_in, feat_in, stage_step, FLAGS.sample,
                                                                       create_graph=False)
        feat_in = feat_neg_out
        feat_negs_all.extend(feat_negs)
        FLAGS.num_fixed_timesteps = feat_in.shape[2]

    FLAGS.num_fixed_timesteps = num_fixed_timesteps_old
    return feat_neg, feat_negs_all, feat_neg_kl, feat_grad

def  gen_trajectories(latent, FLAGS, models, feat_neg, feat, num_steps, sample=False, create_graph=True, idx=None, training_step=0):
    feat_noise = torch.randn_like(feat_neg).detach()
    feat_negs_samples = []

    num_fixed_timesteps = FLAGS.num_fixed_timesteps

    ## Step
    step_lr = FLAGS.step_lr
    ini_step_lr = step_lr

    ## Noise
    noise_coef = FLAGS.noise_coef
    ini_noise_coef = noise_coef

    ## Momentum parameters
    momentum = FLAGS.momentum
    old_update = 0.0

    feat_negs = [feat_neg]

    feat_neg.requires_grad_(requires_grad=True) # noise image [b, n_o, T, f]
    feat_fixed = feat.clone() #.requires_grad_(requires_grad=False)

    # TODO: Sample from buffer.
    s = feat.size()

    # Num steps linear annealing
    if not sample:
        if FLAGS.num_steps_end != -1 and training_step > 0:
            num_steps = int(linear_annealing(None, training_step, start_step=100000, end_step=FLAGS.ns_iteration_end, start_value=num_steps, end_value=FLAGS.num_steps_end))

        #### FEATURE TEST ##### varying fixed steps
        # if random.random() < 0.5:
        #     num_fixed_timesteps = random.randint(3, num_fixed_timesteps)
        #### FEATURE TEST #####

    if FLAGS.num_fixed_timesteps > 0:
        feat_neg = torch.cat([feat_fixed[:, :, :num_fixed_timesteps],
                              feat_neg[:, :,  num_fixed_timesteps:]], dim=2)
    feat_neg_kl = None

    for i in range(num_steps):
        feat_noise.normal_()

        ## Step - LR
        if FLAGS.step_lr_decay_factor != 1.0:
            step_lr = linear_annealing(None, i, start_step=5, end_step=num_steps-1, start_value=ini_step_lr, end_value=FLAGS.step_lr_decay_factor * ini_step_lr)
        ## Add Noise
        if FLAGS.noise_decay_factor != 1.0:
            noise_coef = linear_annealing(None, i, start_step=0, end_step=num_steps-1, start_value=ini_noise_coef, end_value=FLAGS.step_lr_decay_factor * ini_noise_coef)
        feat_neg = feat_neg + noise_coef * feat_noise

        # Smoothing
        if i % 5 == 0 and i < num_steps - 1: # smooth every 10 and leave the last iterations
            feat_neg = smooth_trajectory(feat_neg, 15, 5.0, 100) # ks, std = 15, 5 # x, kernel_size, std, interp_size

        # Compute energy
        latent_ii, mask = latent
        energy = 0
        curr_latent = latent
        for ii in range(2): # TODO: This was wrong. redo in others.
            if ii == 1: curr_latent = (latent_ii, 1 - mask)
            energy = models[ii].forward(feat_neg, curr_latent) + energy
        # Get grad for current optimization iteration.
        feat_grad, = torch.autograd.grad([energy.sum()], [feat_neg], create_graph=create_graph)

        #### FEATURE TEST #####
        # clamp_val, feat_grad_norm = .8, feat_grad.norm()
        # if feat_grad_norm > clamp_val:
        #     feat_grad = clamp_val * feat_grad / feat_grad_norm
        #### FEATURE TEST #####

        #### FEATURE TEST #####
        # Gradient Dropout - Note: Testing.
        # update_mask = (torch.rand(feat_grad.shape, device=feat_grad.device) > 0.2)
        # feat_grad = feat_grad * update_mask
        #### FEATURE TEST #####

        # KL Term computation ### From Compose visual relations ###
        # Note: TODO: Review this approach
        if i == num_steps - 1 and not sample and FLAGS.kl:
            feat_neg_kl = feat_neg
            rand_idx = torch.randint(2, (1,))
            energy = models[rand_idx].forward(feat_neg_kl, latent)
            feat_grad_kl, = torch.autograd.grad([energy.sum()], [feat_neg_kl], create_graph=create_graph)  # Create_graph true?
            feat_neg_kl = feat_neg_kl - step_lr * feat_grad_kl #[:FLAGS.batch_size]
            feat_neg_kl = torch.clamp(feat_neg_kl, -1, 1)

        ## Momentum update
        if FLAGS.momentum > 0:
            update = FLAGS.step_lr * feat_grad[:, :,  num_fixed_timesteps:] + momentum * old_update
            feat_neg = feat_neg[:, :,  num_fixed_timesteps:] - update # GD computation
            old_update = update
        else:
            feat_neg = feat_neg[:, :,  num_fixed_timesteps:] - FLAGS.step_lr * feat_grad[:, :,  num_fixed_timesteps:] # GD computation


        if num_fixed_timesteps > 0:
            feat_neg = torch.cat([feat_fixed[:, :, :num_fixed_timesteps],
                                  feat_neg], dim=2)

        # latents = latents
        feat_neg = torch.clamp(feat_neg, -1, 1)
        feat_negs.append(feat_neg)
        feat_neg = feat_neg.detach()
        feat_neg.requires_grad_()  # Q: Why detaching and reataching? Is this to avoid backprop through this step?

    return feat_neg, feat_negs, feat_neg_kl, feat_grad

# gen trajectories with differential sampling ###
def gen_trajectories_diff (latent, FLAGS, models, feat_neg, feat, num_steps, sample=False, create_graph=True, idx=None, training_step=0):
    assert FLAGS.num_fixed_timesteps > 0 # In this case, we are encoding velocities and therefore we need at least one GT point.

    feat_noise = torch.randn_like(feat_neg).detach()
    feat_negs_samples = []

    num_fixed_timesteps = FLAGS.num_fixed_timesteps

    ## Step
    step_lr = FLAGS.step_lr
    ini_step_lr = step_lr

    ## Noise
    noise_coef = FLAGS.noise_coef
    ini_noise_coef = noise_coef

    ## Momentum parameters
    momentum = FLAGS.momentum
    old_update = 0.0

    feat_negs = [feat_neg]

    feat_clone = feat.clone() #.requires_grad_(requires_grad=False)

    # TODO: Sample from buffer.
    s = feat.size()

    # Num steps linear annealing
    if not sample:
        if FLAGS.num_steps_end != -1 and training_step > 0:
            num_steps = int(linear_annealing(None, training_step, start_step=100000, end_step=FLAGS.ns_iteration_end, start_value=num_steps, end_value=FLAGS.num_steps_end))

        #### FEATURE TEST ##### varying fixed steps
        # if random.random() < 0.5:
        #     num_fixed_timesteps = random.randint(3, num_fixed_timesteps)
        #### FEATURE TEST #####

    #### FEATURE TEST ##### random fixed steps
    # if FLAGS.num_fixed_timesteps > 0:
    # if random.random() < 0.5:
    #     num_fixed_timesteps = random.randint(3, num_fixed_timesteps)
    # fixed_points_mask = torch.rand_like(feat_neg) < ((num_fixed_timesteps + 6)/FLAGS.num_timesteps)
    # feat_neg[fixed_points_mask] = feat_fixed[fixed_points_mask]

    feat_neg.requires_grad_(requires_grad=True) # noise image [b, n_o, T, f]

    feat_neg_kl = None


    feat_var = torch.cat([feat_clone[:, :,  num_fixed_timesteps-1:num_fixed_timesteps],
                          feat_neg[:, :,  num_fixed_timesteps:]], dim=2)


    #### FEATURE TEST #### TODO: Not fine yet
    delta_neg = torch.randn_like(feat_var[:, :, 1:]) * 0.0001 #(feat_var[:, :, 1:] - feat_var[:, :, :-1])
    delta_neg.requires_grad_(requires_grad=True)

    feat_var_ini = feat_var[:, :, 0:1]
    feat_var_list = [feat_var_ini]
    current = feat_var_ini
    for tt in range(delta_neg.shape[2]):
        current = current + delta_neg[:, :, tt:tt+1]
        feat_var_list.append(current)
    feat_var = torch.cat(feat_var_list, dim=2)

    if FLAGS.num_fixed_timesteps > 1:
        feat_fixed = feat_clone[:, :, :num_fixed_timesteps - 1]
        feat_neg = torch.cat([feat_fixed,
                              feat_var], dim=2)
    else: feat_neg = feat_var

    for i in range(num_steps):
        feat_noise.normal_()

        ## Step - LR
        if FLAGS.step_lr_decay_factor != 1.0:
            step_lr = linear_annealing(None, i, start_step=5, end_step=num_steps-1, start_value=ini_step_lr, end_value=FLAGS.step_lr_decay_factor * ini_step_lr)
        ## Add Noise
        if FLAGS.noise_decay_factor != 1.0:
            noise_coef = linear_annealing(None, i, start_step=0, end_step=num_steps-1, start_value=ini_noise_coef, end_value=FLAGS.step_lr_decay_factor * ini_noise_coef)
        feat_neg = feat_neg + noise_coef * feat_noise

        # Smoothing
        # TODO: put back in place
        if i % 5 == 0 and i < num_steps - 1: # smooth every 10 and leave the last iterations
            feat_neg = smooth_trajectory(feat_neg, 15, 5.0, 100) # ks, std = 15, 5 # x, kernel_size, std, interp_size

        # Compute energy
        latent_ii, mask = latent
        energy = 0
        for ii in range(2):
            if ii == 1: latent = (latent_ii, 1 - mask)
            energy = models[ii].forward(feat_neg, latent) + energy
        # Get grad for current optimization iteration.
        delta_grad, = torch.autograd.grad([energy.sum()], [delta_neg], create_graph=create_graph)
        # feat_grad_2, = torch.autograd.grad([energy.sum()], [feat_neg], create_graph=create_graph) # Note: Only to evaluate expression

        # TODO: Calculate grads without taking into account the fixed points.

        #### FEATURE TEST #####
        # clamp_val, feat_grad_norm = .8, feat_grad.norm()
        # if feat_grad_norm > clamp_val:
        #     feat_grad = clamp_val * feat_grad / feat_grad_norm
        #### FEATURE TEST #####

        #### FEATURE TEST #####
        # Gradient Dropout - Note: Testing.
        # update_mask = (torch.rand(feat_grad.shape, device=feat_grad.device) > 0.2)
        # feat_grad = feat_grad * update_mask
        #### FEATURE TEST #####

        # KL Term computation ### From Compose visual relations ###
        # Note: TODO: Review this approach
        if i == num_steps - 1 and not sample and FLAGS.kl:
            feat_neg_kl = feat_neg
            rand_idx = torch.randint(2, (1,))
            energy = models[rand_idx].forward(feat_neg_kl, latent)

            delta_grad_kl, = torch.autograd.grad([energy.sum()], [delta_neg], create_graph=create_graph)  # Create_graph true?
            delta_neg_kl = delta_neg_kl - step_lr * delta_grad_kl #[:FLAGS.batch_size]


            feat_var_kl = [feat_var_ini]
            current = feat_var_ini
            for tt in range(delta_neg_kl.shape[2]):
                current = current + delta_neg[:, :, tt:tt+1]
                feat_var.append(current)
            feat_var_kl = torch.cat(feat_var_kl, dim=2)
            if FLAGS.num_fixed_timesteps > 1:
                feat_fixed = feat_clone[:, :, :num_fixed_timesteps - 1]
                feat_neg_kl = torch.cat([feat_fixed,
                                      feat_var_kl], dim=2)
            feat_neg_kl = torch.clamp(feat_neg_kl, -1, 1)

        ## Momentum update
        if FLAGS.momentum > 0:
            update = FLAGS.step_lr * delta_grad + momentum * old_update
            delta_neg = delta_neg - update # GD computation
            old_update = update
        else:
            delta_neg = delta_neg - FLAGS.step_lr * delta_grad # GD computation

        feat_var_list = [feat_var_ini]
        current = feat_var_ini
        for tt in range(delta_neg.shape[2]):
            current = current + delta_neg[:, :, tt:tt+1]
            feat_var_list.append(current)
        feat_var = torch.cat(feat_var_list, dim=2)

        if FLAGS.num_fixed_timesteps > 1:
            feat_fixed = feat_clone[:, :, :num_fixed_timesteps - 1]
            feat_neg = torch.cat([feat_fixed,
                                  feat_var], dim=2)
        else: feat_neg = feat_var

        # latents = latents
        feat_neg = torch.clamp(feat_neg, -1, 1)
        feat_negs.append(feat_neg)

        #### FEATURE TEST ##### If commented out, backprop through all. (?)
        feat_neg_out = feat_neg.detach()
        # feat_neg.requires_grad_()  # Q: Why detaching and reataching? Is this to avoid backprop through this step?
    return feat_neg_out, feat_negs, feat_neg_kl, delta_grad

def sync_model(models): # Q: What is this about?
    size = float(dist.get_world_size())

    for model in models:
        for param in model.parameters():
            dist.broadcast(param.data, 0)

def init_model(FLAGS, device, dataset):
    # model = EdgeGraphEBM_LateFusion(FLAGS, dataset).to(device)
    # model = EdgeGraphEBM_CNNOneStep(FLAGS, dataset).to(device)
    # model = EdgeGraphEBM_OneStep(FLAGS, dataset).to(device)
    model = EdgeGraphEBM_CNN_OS_noF(FLAGS, dataset).to(device)
    models = [model for i in range(FLAGS.ensembles)]
    optimizers = [Adam(model.parameters(), lr=FLAGS.lr) for model in models] # Note: From CVR , betas=(0.5, 0.99)
    return models, optimizers

def test_manipulate(dataloaders, models, FLAGS, step=0, save = False, logger = None):
    b_idx_1 = 9
    b_idx_2 = 12

    print(FLAGS)
    if FLAGS.cuda:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    replay_buffer = None

    print('Begin test:')
    (feat_1, edges_1), (rel_rec, rel_send), idx = next(iter(dataloaders[0]))
    (feat_2, edges_2), _, _                     = next(iter(dataloaders[1]))

    feats = [feat_1[b_idx_1:b_idx_1+1].to(dev), feat_2[b_idx_2:b_idx_2+1].to(dev)]
    rel_rec = rel_rec[:1].to(dev)
    rel_send = rel_send[:1].to(dev)
    bs = feat_1.size(0)

    ### SELECTED PAIRS ###
    pairs = get_rel_pairs(rel_send, rel_rec)
    rw_pair = pairs[2]
    affected_nodes = (rel_rec + rel_send)[0, rw_pair].mean(0).clamp_(min=0, max=1).data.cpu().numpy()

    # TODO: Initialization of nodes.
    #  Currently the initialization is taken from one of the datasets
    feat = feats[0]

    latents = []
    for mod_idx, model in enumerate(models):
        if FLAGS.forecast is not -1:
            feat_enc = feats[mod_idx][:, :, :-FLAGS.forecast]
        else: feat_enc = feats[mod_idx]
        if FLAGS.normalize_data_latent:
            feat_enc = normalize_trajectories(feat_enc, augment=False) # We max min normalize the batch
        latents.append(model.embed_latent(feat_enc, rel_rec, rel_send))

    mask = torch.ones(FLAGS.components).to(dev)
    mask[rw_pair] = 0

    # mask = mask * 0
    # mask[rw_pair] = 1

    # mask[3:] = 1

    latent_mix = latents[0] * mask[None, :, None] + latents[1] * (1-mask)[None, :, None]
    latent = (latent_mix, mask)

    # latent = latent * mask[None, :, None]

    ### NOTE: TEST: Random rotation of the input trajectory
    # feat = torch.cat(augment_trajectories((feat[..., :2], feat[..., 2:]), rotation='random'), dim=-1)
    # feat = normalize_trajectories(feat)

    feat_neg = torch.rand_like(feats[0]) * 2 - 1
    # if replay_buffer is not None:
    #     if FLAGS.replay_batch and len(replay_buffer) >= FLAGS.batch_size:
    #         replay_batch, idxs = replay_buffer.sample(feat_neg.size(0))
    #         replay_batch = decompress_x_mod(replay_batch)
    #         feat_neg = torch.Tensor(replay_batch).to(dev)
    #         feat_neg = align_replayed_batch(feat, feat_neg)
    #         if FLAGS.num_fixed_timesteps > 0:
    #             feat_neg = align_replayed_batch(feat, feat_neg)

    if sample_diff:
        feat_neg, feat_negs, feat_neg_kl, feat_grad = gen_trajectories_diff(latent, FLAGS, models, feat_neg, feat, FLAGS.num_steps_test, FLAGS.sample,
                                                                            create_graph=False) # TODO: why create_graph
    else:
        feat_neg, feat_negs, feat_neg_kl, feat_grad = gen_trajectories(latent, FLAGS, models, feat_neg, feat, FLAGS.num_steps_test, FLAGS.sample,
                                                                           create_graph=False) # TODO: why create_graph
    # feat_negs = torch.stack(feat_negs, dim=1) # 5 iterations only
    if save:
        # print('Latent: \n{}'.format(latent[0][b_idx].data.cpu().numpy()))
        limpos = limneg = 1 if FLAGS.plot_attr == 'loc' else 4
        lims = [-limneg, limpos]

        lims = None
        for i_plt in range(len(feat_negs)):
            logger.add_figure('test_manip_gen_rec', get_trajectory_figure(feat_negs[i_plt], lims=lims, b_idx=0, plot_type =FLAGS.plot_attr, highlight_nodes = affected_nodes)[1], i_plt)
        # logger.add_figure('test_manip_gen', get_trajectory_figure(feat_neg, b_idx=b_idx, lims=lims, plot_type =FLAGS.plot_attr)[1], step)
        logger.add_figure('test_manip_gt_1', get_trajectory_figure(feats[0], b_idx=0, lims=lims, plot_type =FLAGS.plot_attr)[1], 0)
        logger.add_figure('test_manip_gt_2', get_trajectory_figure(feats[1], b_idx=0, lims=lims, plot_type =FLAGS.plot_attr)[1], 0)
        print('Plotted.')

        # elif logger is not None:
        #     l2_loss = torch.pow(feat_neg_kl[:, :,  FLAGS.num_fixed_timesteps:] - feat[:, :,  FLAGS.num_fixed_timesteps:], 2).mean()
        #     logger.add_scalar('aaa-L2_loss_test', l2_loss.item(), step)

    print('test done')
    exit()

def main_single(rank, FLAGS):
    rank_idx = FLAGS.node_rank * FLAGS.gpus + rank
    world_size = FLAGS.nodes * FLAGS.gpus

    if not os.path.exists('result/%s_%s' % (FLAGS.exp[0], FLAGS.exp[1])):
        try:
            os.makedirs('result/%s_%s' % (FLAGS.exp[0], FLAGS.exp[1]))
        except:
            pass

    datasets = []
    if 'springs' in FLAGS.dataset:
        dataset = SpringsParticles(FLAGS, 'train')
        test_dataset = SpringsParticles(FLAGS, 'test')
        datasets.append(test_dataset)
    if 'charged' in FLAGS.dataset:
        dataset = ChargedParticles(FLAGS, 'train')
        test_dataset = ChargedParticles(FLAGS, 'test')
        datasets.append(test_dataset)
    if FLAGS.dataset == 'charged_sim':
        dataset = ChargedParticlesSim(FLAGS)
        test_dataset = dataset
        datasets.append(test_dataset)
    FLAGS.timesteps = FLAGS.num_timesteps
    assert len(datasets) == 2

    shuffle=True
    sampler = None
    replay_buffer = None

    # p = random.randint(0, 9)
    if world_size > 1:
        group = dist.init_process_group(backend='nccl', init_method='tcp://localhost:'+str(FLAGS.port), world_size=world_size, rank=rank_idx, group_name="default")
    torch.cuda.set_device(rank)
    device = torch.device('cuda')

    FLAGS_OLD = FLAGS

    for ckpt_idx, (resume_iter, resume_name) in enumerate(zip(FLAGS.resume_iter, FLAGS.resume_name)):
        if resume_iter != 0:
            if resume_name is not '':
                logdir = osp.join(FLAGS.logdir, FLAGS.exp[ckpt_idx], resume_name)
            else: print('Must provide checkpoint names.'); exit()

            model_path = osp.join(logdir, "model_{}.pth".format(resume_iter))
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            if FLAGS.replay_batch:
                try:
                    replay_buffer = torch.load(osp.join(logdir, "rb.pt"))
                except: pass

            if ckpt_idx == 0:
                FLAGS = checkpoint['FLAGS']
                FLAGS.normalize_data_latent = FLAGS_OLD.normalize_data_latent # Maybe we shouldn't override
                FLAGS.factor_encoder = FLAGS_OLD.factor_encoder
                FLAGS.plot_attr = FLAGS_OLD.plot_attr
                FLAGS.resume_iter = FLAGS_OLD.resume_iter
                FLAGS.resume_name = FLAGS_OLD.resume_name
                FLAGS.save_interval = FLAGS_OLD.save_interval
                FLAGS.nodes = FLAGS_OLD.nodes
                FLAGS.gpus = FLAGS_OLD.gpus
                FLAGS.node_rank = FLAGS_OLD.node_rank
                FLAGS.train = FLAGS_OLD.train
                FLAGS.batch_size = FLAGS_OLD.batch_size
                FLAGS.num_visuals = FLAGS_OLD.num_visuals
                FLAGS.num_additional = FLAGS_OLD.num_additional
                FLAGS.decoder = FLAGS_OLD.decoder
                FLAGS.test_manipulate = FLAGS_OLD.test_manipulate
                # FLAGS.sim = FLAGS_OLD.sim
                FLAGS.exp = FLAGS_OLD.exp
                FLAGS.step_lr = FLAGS_OLD.step_lr
                FLAGS.num_steps = FLAGS_OLD.num_steps
                FLAGS.num_steps_test = FLAGS_OLD.num_steps_test
                FLAGS.forecast = FLAGS_OLD.forecast
                FLAGS.autoencode = FLAGS_OLD.autoencode
                FLAGS.entropy_nn = FLAGS_OLD.entropy_nn
                FLAGS.cd_and_ae = FLAGS_OLD.cd_and_ae
                FLAGS.num_fixed_timesteps = FLAGS_OLD.num_fixed_timesteps
                # TODO: Check what attributes we are missing
                models, _ = init_model(FLAGS, device, dataset)

            # Note: We load the first learned model for each of the datasets

            models[ckpt_idx].load_state_dict(checkpoint['model_state_dict_{}'.format(ckpt_idx)], strict=False)
        else: print('Must provide checkpoint resume iteration.'); exit()

    if FLAGS.gpus > 1:
        sync_model(models)

    test_manipulate_dataloader_1 = DataLoader(datasets[0], num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=False, pin_memory=False, drop_last=True)
    test_manipulate_dataloader_2 = DataLoader(datasets[1], num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=False, pin_memory=False, drop_last=True)
    dataloaders = (test_manipulate_dataloader_1, test_manipulate_dataloader_2)

    logdir = osp.join(FLAGS.logdir, FLAGS.exp[0]+'_'+FLAGS.exp[1], FLAGS.resume_name[0])
    logger = SummaryWriter(logdir)

    models = [model.eval() for model in models]
    test_manipulate(dataloaders, models, FLAGS, step=FLAGS.resume_iter, save=True, logger=logger)

def main():
    FLAGS = parser.parse_args()
    FLAGS.components = FLAGS.n_objects ** 2 - FLAGS.n_objects
    FLAGS.ensembles = 2
    FLAGS.tie_weight = True
    FLAGS.sample = True
    FLAGS.exp = FLAGS.dataset

    assert len(FLAGS.exp) == len(FLAGS.resume_name) == len(FLAGS.resume_iter) == 2
    logdir = osp.join(FLAGS.logdir, FLAGS.exp[0] + '_' + FLAGS.exp[1])

    if not osp.exists(logdir):
        os.makedirs(logdir)

    if FLAGS.gpus > 1:
        mp.spawn(main_single, nprocs=FLAGS.gpus, args=(FLAGS,))
    else:
        main_single(FLAGS.gpu_rank, FLAGS)


if __name__ == "__main__":
    main()
