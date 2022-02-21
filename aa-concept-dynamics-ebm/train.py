import torch
import time
from models import TrajEBM, TrajGraphEBM, EdgeGraphEBM #LatentEBM, ToyEBM, BetaVAE_H, LatentEBM128
from scipy.linalg import toeplitz
# from tensorflow.python.platform import flags
import numpy as np
import torch.nn.functional as F
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from dataset import ChargedParticlesSim, ChargedParticles, SpringsParticles
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim import Adam, AdamW
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
    normalize_trajectories, augment_trajectories
from pathlib import Path

# --exp=springs --num_steps=19 --step_lr=30.0 --dataset=springs --cuda --train --batch_size=24 --latent_dim=8 --pos_embed --data_workers=0 --gpus=1 --node_rank=1
# python train.py --exp=charged --num_steps=2 --num_steps_test 50 --step_lr=10.0 --dataset=charged --cuda --train --batch_size=24 --latent_dim=64 --data_workers=4 --gpus=1 --gpu_rank 1 --autoencode

homedir = '/data/Armand/EBM/'
# port = 6021

"""Parse input arguments"""
parser = argparse.ArgumentParser(description='Train EBM model')
parser.add_argument('--train', action='store_true', help='whether or not to train')
parser.add_argument('--test_manipulate', action='store_true', help='whether or not to train')
parser.add_argument('--cuda', default=True, action='store_true', help='whether to use cuda or not')
parser.add_argument('--port', default=6010, type=int, help='Port for distributed')

parser.add_argument('--single', action='store_true', help='test overfitting of the dataset')


parser.add_argument('--dataset', default='charged', type=str, help='Dataset to use (intphys or others or imagenet or cubes)')
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
parser.add_argument('--resume_iter', default=0, type=int, help='iteration to resume training')
parser.add_argument('--resume_name', default='', type=str, help='name of the model to resume')
parser.add_argument('--batch_size', default=64, type=int, help='size of batch of input to use')
parser.add_argument('--num_epoch', default=10000, type=int, help='number of epochs of training to run')
parser.add_argument('--lr', default=3e-4, type=float, help='learning rate for training')
parser.add_argument('--log_interval', default=100, type=int, help='log outputs every so many batches')
parser.add_argument('--save_interval', default=2000, type=int, help='save outputs every so many batches')
parser.add_argument('--autoencode', action='store_true', help='if set to True we use L2 loss instead of Contrastive Divergence')
parser.add_argument('--forecast', action='store_true', help='if set to True we use L2 loss instead of Contrastive Divergence')
parser.add_argument('--independent_energies', action='store_true', help='Use Edge graph ebm')
parser.add_argument('--cd_and_ae', action='store_true', help='if set to True we use L2 loss and Contrastive Divergence')

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
parser.add_argument('--dropout', default=0.2, type=float, help='use spatial latents for object segmentation')
parser.add_argument('--factor_encoder', action='store_true', help='if we use message passing in the encoder')
parser.add_argument('--normalize_data_latent', action='store_true', help='if we normalize data before encoding the latents')

# Model specific - TrajEBM
parser.add_argument('--input_dim', default=4, type=int, help='dimension of an object')
parser.add_argument('--latent_hidden_dim', default=64, type=int, help='hidden dimension of the latent')
parser.add_argument('--latent_dim', default=64, type=int, help='dimension of the latent')
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

def  gen_trajectories(latents, FLAGS, models, models_ema, feat_neg, feat, num_steps, sample=False, create_graph=True, idx=None, training_step=0):
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

    feat_negs = []
    # latents = torch.stack(latents, dim=0)

    feat_neg.requires_grad_(requires_grad=True) # noise image [b, n_o, T, f]
    feat_fixed = feat.clone() #.requires_grad_(requires_grad=False)

    # TODO: Sample from buffer.
    s = feat.size()

    # Num steps linear annealing
    if not sample:
        if FLAGS.num_steps_end != -1 and training_step > 0:
            num_steps = int(linear_annealing(None, training_step, start_step=100000, end_step=FLAGS.ns_iteration_end, start_value=num_steps, end_value=FLAGS.num_steps_end))

        #### FEATURE TEST ##### varying fixed steps
        if random.random() < 0.5:
            num_fixed_timesteps = random.randint(2, num_fixed_timesteps)
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
        if i % 5 == 0 and i < num_steps - 10: # smooth every 10 and leave the last iterations
            feat_neg = smooth_trajectory(feat_neg, 15, 5.0, 100) # ks, std = 15, 5 # x, kernel_size, std, interp_size

        # Compute energy
        energy = 0
        for j in range(len(latents)):
            if FLAGS.sample_ema:
                energy = models_ema[j % FLAGS.components].forward(feat_neg, latents[j]) + energy
            else:
                energy = models[j % FLAGS.components].forward(feat_neg, latents[j]) + energy
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
        # Note: TODO: We are doing one unnecessary step here. we could simply compute feat_neg[-1] as feat_neg_kl.
        if i == num_steps - 1 and not sample:
            feat_neg_kl = feat_neg

            energy = 0
            for j in range(len(latents)):
                energy = models[j % FLAGS.components].forward(feat_neg_kl, latents[j]) + energy
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


        if num_fixed_timesteps > 0: # TODO: Check if this works for num_fixed_timesteps = 0
            feat_neg = torch.cat([feat_fixed[:, :, :num_fixed_timesteps],
                                  feat_neg], dim=2)

        # latents = latents
        feat_neg = torch.clamp(feat_neg, -1, 1)
        feat_negs.append(feat_neg)
        feat_neg = feat_neg.detach()
        feat_neg.requires_grad_()  # Q: Why detaching and reataching? Is this to avoid backprop through this step?

    return feat_neg, feat_negs, feat_neg_kl, feat_grad


def sync_model(models): # Q: What is this about?
    size = float(dist.get_world_size())

    for model in models:
        for param in model.parameters():
            dist.broadcast(param.data, 0)

def init_model(FLAGS, device, dataset):
    if FLAGS.tie_weight:
        if FLAGS.independent_energies:
            model = EdgeGraphEBM(FLAGS, dataset).to(device)
        else:
            # model = TrajEBM(FLAGS, dataset).to(device)
            model = TrajGraphEBM(FLAGS, dataset).to(device)

        models = [model for i in range(FLAGS.ensembles)]
        optimizers = [Adam(model.parameters(), lr=FLAGS.lr)] # Note: From CVR , betas=(0.0, 0.9), eps=1e-8
    else:
        models = [TrajEBM(FLAGS, dataset).to(device) for i in range(FLAGS.ensembles)]
        optimizers = [Adam(model.parameters(), lr=FLAGS.lr) for model in models] # , betas=(0.0, 0.9), eps=1e-8
        # TODO: check if betas are correct.
    return models, optimizers

def test_manipulate(train_dataloader, models, models_ema, FLAGS, step=0, save = False, logger = None):
    if FLAGS.cuda:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    if FLAGS.independent_energies:
        edge_ids = torch.eye(FLAGS.components)[None] \
            .to(dev).requires_grad_(False)

    replay_buffer = None

    print('Begin test:')
    [model.eval() for model in models]
    for (feat, edges), (rel_rec, rel_send), idx in train_dataloader:

        feat = feat.to(dev)
        edges = edges.to(dev)
        rel_rec = rel_rec.to(dev)
        rel_send = rel_send.to(dev)
        # What are these? im = im[:FLAGS.num_visuals], idx = idx[:FLAGS.num_visuals]
        bs = feat.size(0)

        if FLAGS.forecast:
            feat_enc = feat[:, :, :FLAGS.num_fixed_timesteps]
        else: feat_enc = feat
        if FLAGS.normalize_data_latent:
            feat_enc = normalize_trajectories(feat_enc) # We maxmin normalize the batch
        latent = models[0].embed_latent(feat_enc, rel_rec, rel_send)

        latents = torch.chunk(latent, FLAGS.components, dim=1)

        ### NOTE: TEST 1: All but 1 latent
        # latents = latents[:-1]
        # latents = [latents[1], latents[0], *latents[2:]]
        # latents = latents[:-3]

        # if FLAGS.independent_energies and FLAGS.autoencode: print('Autoencode is not compatible with EdgeGraphEBM'); exit()
        if FLAGS.independent_energies:
            latents = [(item.squeeze(1), edge_ids[:, eid, :, None, None]) for eid, item in enumerate(latents)]
        else: latents = [item.squeeze(1) for item in latents]

        feat_neg = torch.rand_like(feat) * 2 - 1

        feat_neg, feat_negs, feat_neg_kl, feat_grad = gen_trajectories(latents, FLAGS, models, models_ema, feat_neg, feat, FLAGS.num_steps_test, sample=True,
                                                                       create_graph=False) # TODO: why create_graph
        # feat_negs = torch.stack(feat_negs, dim=1) # 5 iterations only

        if save:
            b_idx = 4
            # savedir = os.path.join(homedir, "result/%s/") % (FLAGS.exp)
            # Path(savedir).mkdir(parents=True, exist_ok=True)
            # savename = "s%08d"% (step)
            # visualize_trajectories(feat, feat_neg, edges, savedir = os.path.join(savedir,savename))
            logger.add_figure('test_manip_gen', get_trajectory_figure(feat_neg, b_idx=b_idx, plot_type =FLAGS.plot_attr)[1], step)
            logger.add_figure('test_manip_gt', get_trajectory_figure(normalize_trajectories(feat), b_idx=b_idx, plot_type =FLAGS.plot_attr)[1], step)
            print('Plotted.')
        # elif logger is not None:
        #     l2_loss = torch.pow(feat_neg_kl[:, :,  FLAGS.num_fixed_timesteps:] - feat[:, :,  FLAGS.num_fixed_timesteps:], 2).mean()
        #     logger.add_scalar('aaa-L2_loss_test', l2_loss.item(), step)
        break
    print('test done')
    exit()

def test(train_dataloader, models, models_ema, FLAGS, step=0, save = False, logger = None):
    if FLAGS.cuda:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    if FLAGS.independent_energies:
        edge_ids = torch.eye(FLAGS.components)[None]\
            .to(dev).requires_grad_(False)

    replay_buffer = None

    [model.eval() for model in models]
    # [getattr(model, module).train() for model in models for module in ['rnn1', 'rnn2']]
    for (feat, edges), (rel_rec, rel_send), idx in train_dataloader:

        feat = feat.to(dev)
        edges = edges.to(dev)
        rel_rec = rel_rec.to(dev)
        rel_send = rel_send.to(dev)
        bs = feat.size(0)

        if FLAGS.forecast:
            feat_enc = feat[:, :, :FLAGS.num_fixed_timesteps]
        else: feat_enc = feat
        if FLAGS.normalize_data_latent:
            feat_enc = normalize_trajectories(feat_enc)
        latent = models[0].embed_latent(feat_enc, rel_rec, rel_send)

        latents = torch.chunk(latent, FLAGS.components, dim=1)

        # if FLAGS.independent_energies and FLAGS.autoencode: print('Autoencode is not compatible with EdgeGraphEBM'); exit()
        if FLAGS.independent_energies:
            latents = [(item.squeeze(1), edge_ids[:, eid, :, None, None]) for eid, item in enumerate(latents)]
        else: latents = [item.squeeze(1) for item in latents]

        feat_neg = torch.rand_like(feat) * 2 - 1

        feat_neg, feat_negs, feat_neg_kl, feat_grad = gen_trajectories(latents, FLAGS, models, models_ema, feat_neg, feat, FLAGS.num_steps_test, FLAGS.sample,
                                                             create_graph=False) # TODO: why create_graph

        if save:
            # savedir = os.path.join(homedir, "result/%s/") % (FLAGS.exp)
            # Path(savedir).mkdir(parents=True, exist_ok=True)
            # savename = "s%08d"% (step)
            # visualize_trajectories(feat, feat_neg, edges, savedir = os.path.join(savedir,savename))
            logger.add_figure('test_gen', get_trajectory_figure(feat_neg, b_idx=0, plot_type =FLAGS.plot_attr)[1], step)
            logger.add_figure('test_gt', get_trajectory_figure(feat, b_idx=0, plot_type =FLAGS.plot_attr)[1], step)
        elif logger is not None:
            l2_loss = torch.pow(feat_neg[:, :,  FLAGS.num_fixed_timesteps:] - feat[:, :,  FLAGS.num_fixed_timesteps:], 2).mean()
            logger.add_scalar('aaa-L2_loss_test', l2_loss.item(), step)


        # Note: Ask Yilun about all this
        # feat_neg = feat_neg.detach()
        # feat_components = []
        #
        # if FLAGS.components > 1:
        #     for i, latent in enumerate(latents):
        #         feat_init = torch.rand_like(feat)
        #         latents_select = latents[i:i+1]
        #         feat_component, _, _, _ = gen_trajectories(latents_select, FLAGS, models, feat_init, feat, FLAGS.num_steps, sample=FLAGS.sample,
        #                                    create_graph=False)
        #         feat_components.append(feat_component)
        #
        #     feat_init = torch.rand_like(feat)
        #     latents_perm = [torch.cat([latent[i:], latent[:i]], dim=0) for i, latent in enumerate(latents)]
        #     feat_neg_perm, _, feat_grad_perm, _ = gen_trajectories(latents_perm, FLAGS, models, feat_init, feat, FLAGS.num_steps, sample=FLAGS.sample,
        #                                              create_graph=False)
        #     feat_neg_perm = feat_neg_perm.detach()
        #     feat_init = torch.rand_like(feat)
        #     add_latents = list(latents)
        #     for i in range(FLAGS.num_additional):
        #         add_latents.append(torch.roll(latents[i], i + 1, 0))
        #     feat_neg_additional, _, _, _ = gen_trajectories(tuple(add_latents), FLAGS, models, feat_init, feat, FLAGS.num_steps, sample=FLAGS.sample,
        #                                              create_graph=False)
        #
        # feat.requires_grad = True
        # feat_grads = []
        #
        # for i, latent in enumerate(latents):
        #     if FLAGS.decoder:
        #         feat_grad = torch.zeros_like(feat)
        #     else:
        #         energy_pos = models[i].forward(feat, latents[i])
        #         feat_grad = torch.autograd.grad([energy_pos.sum()], [feat])[0]
        #     feat_grads.append(feat_grad)
        #
        # feat_grad = torch.stack(feat_grads, dim=1)
        #
        # s = feat.size()
        # feat_size = s[-1]
        #
        # feat_grad_dense = feat_grad.view(batch_size, FLAGS.components, 1, -1, 1) # [4, 3, 1, 49152, 1]
        # feat_grad_min = feat_grad_dense.min(dim=3, keepdim=True)[0]
        # feat_grad_max = feat_grad_dense.max(dim=3, keepdim=True)[0] # [4, 3, 1, 1, 1]
        #
        # feat_grad = (feat_grad - feat_grad_min) / (feat_grad_max - feat_grad_min + 1e-5) # [4, 3, 3, 128, 128]

        # im_grad[:, :, :, :1, :] = 1
        # im_grad[:, :, :, -1:, :] = 1
        # im_grad[:, :, :, :, :1] = 1
        # im_grad[:, :, :, :, -1:] = 1
        # im_output = im_grad.permute(0, 3, 1, 4, 2).reshape(batch_size * im_size, FLAGS.components * im_size, 3)
        # im_output = im_output.cpu().detach().numpy() * 100
        # im_output = (im_output - im_output.min()) / (im_output.max() - im_output.min())

        # feat = feat.cpu().detach().numpy().transpose((0, 2, 3, 1)).reshape(batch_size*im_size, im_size, 3)

        # im_output = np.concatenate([im_output, im], axis=1)
        # im_output = im_output*255
        # imwrite("result/%s/s%08d_grad.png" % (FLAGS.exp,step), im_output)
        #
        # im_neg = im_neg_tensor = im_neg.detach().cpu()
        # im_components = [im_components[i].detach().cpu() for i in range(len(im_components))]
        # im_neg = torch.cat([im_neg] + im_components)
        # im_neg = np.clip(im_neg, 0.0, 1.0)
        # im_neg = make_grid(im_neg, nrow=int(im_neg.shape[0] / (FLAGS.components + 1))).permute(1, 2, 0)
        # im_neg = im_neg.numpy()*255
        # imwrite("result/%s/s%08d_gen.png" % (FLAGS.exp,step), im_neg)
        #
        # if FLAGS.components > 1:
        #     im_neg_perm = im_neg_perm.detach().cpu()
        #     im_components_perm = []
        #     for i,im_component in enumerate(im_components):
        #         im_components_perm.append(torch.cat([im_component[i:], im_component[:i]]))
        #     im_neg_perm = torch.cat([im_neg_perm] + im_components_perm)
        #     im_neg_perm = np.clip(im_neg_perm, 0.0, 1.0)
        #     im_neg_perm = make_grid(im_neg_perm, nrow=int(im_neg_perm.shape[0] / (FLAGS.components + 1))).permute(1, 2, 0)
        #     im_neg_perm = im_neg_perm.numpy()*255
        #     imwrite("result/%s/s%08d_gen_perm.png" % (FLAGS.exp,step), im_neg_perm)
        #
        #     im_neg_additional = im_neg_additional.detach().cpu()
        #     for i in range(FLAGS.num_additional):
        #         im_components.append(torch.roll(im_components[i], i + 1, 0))
        #     im_neg_additional = torch.cat([im_neg_additional] + im_components)
        #     im_neg_additional = np.clip(im_neg_additional, 0.0, 1.0)
        #     im_neg_additional = make_grid(im_neg_additional,
        #                         nrow=int(im_neg_additional.shape[0] / (FLAGS.components + FLAGS.num_additional + 1))).permute(1, 2, 0)
        #     im_neg_additional = im_neg_additional.numpy()*255
        #     imwrite("result/%s/s%08d_gen_add.png" % (FLAGS.exp,step), im_neg_additional)

        break

    [model.train() for model in models]

def train(train_dataloader, test_dataloader, logger, models, models_ema, optimizers, FLAGS, logdir, rank_idx):

    it = FLAGS.resume_iter
    losses, l2_losses = [], []
    grad_norm_ema = None
    [optimizer.zero_grad() for optimizer in optimizers]

    if FLAGS.replay_batch or FLAGS.entropy_nn:
        replay_buffer = ReplayBuffer(FLAGS.buffer_size, None, FLAGS)# Flags.transform

    # if FLAGS.scheduler:
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers, T_max=1000, eta_min=0, last_epoch=-1)

    dev = torch.device("cuda")

    if FLAGS.independent_energies:
        edge_ids = torch.eye(FLAGS.components)[None] \
            .to(dev).requires_grad_(False)

    start_time = time.perf_counter()
    for epoch in range(FLAGS.num_epoch):
        for (feat, edges), (rel_rec, rel_send), idx in train_dataloader:

            feat = feat.to(dev)
            edges = edges.to(dev)
            rel_rec = rel_rec.to(dev)
            rel_send = rel_send.to(dev)

            if FLAGS.forecast:
                feat_enc = feat[:, :, :FLAGS.num_fixed_timesteps]
            else: feat_enc = feat
            if FLAGS.normalize_data_latent:
                feat_enc = normalize_trajectories(feat_enc, augment=True)
            latent = models[0].embed_latent(feat_enc, rel_rec, rel_send)

            ## Note: TEST FEATURE: Add noise to input after obtaining latents.
            # feat_noise = torch.randn_like(feat).detach()
            # feat_noise.normal_()
            # feat = feat + 0.001 * feat_noise
            ###

            latents = torch.chunk(latent, FLAGS.components, dim=1)

            # if FLAGS.independent_energies and FLAGS.autoencode: print('Autoencode is not compatible with EdgeGraphEBM'); exit()
            if FLAGS.independent_energies:
                latents = [(item.squeeze(1), edge_ids[:, eid, :, None, None]) for eid, item in enumerate(latents)]
            else: latents = [item.squeeze(1) for item in latents]
            latent_norm = latent.norm()

            feat_neg = torch.rand_like(feat) * 2 - 1

            if FLAGS.replay_batch and len(replay_buffer) >= FLAGS.batch_size:
                replay_batch, idxs = replay_buffer.sample(feat_neg.size(0))
                replay_batch = decompress_x_mod(replay_batch)
                replay_mask = (np.random.uniform(0, 1, feat_neg.size(0)) > 0.001)
                feat_neg[replay_mask] = torch.Tensor(replay_batch[replay_mask]).to(dev)

            feat_neg, feat_negs, feat_neg_kl, feat_grad = gen_trajectories(latents, FLAGS, models, models_ema, feat_neg, feat, FLAGS.num_steps, sample=False, training_step=it)

            feat_negs = torch.stack(feat_negs, dim=1) # 5 iterations only

            # if FLAGS.scheduler:
            #     scheduler.step()

            # Note: Supervise with accumulated velocity
            # feat_neg_kl = accumulate_traj(feat_neg_kl)

            ## MSE Loss
                ## Note: Backprop through all sampling
            # feat_loss = torch.pow(feat_negs[:, -1, :,  FLAGS.num_fixed_timesteps:] - feat[:, :,  FLAGS.num_fixed_timesteps:], 2).mean()
                ## Note: Backprop through 1 sampling step
            feat_loss = torch.pow(feat_neg_kl[:, :,  FLAGS.num_fixed_timesteps:] - feat[:, :,  FLAGS.num_fixed_timesteps:], 2).mean()

            ## Compute losses
            loss = 0
            if FLAGS.autoencode or FLAGS.cd_and_ae:
                loss = loss + feat_loss
            if not FLAGS.autoencode or FLAGS.cd_and_ae:
                energy_poss = []
                energy_negs = []
                for i in range(FLAGS.components):
                    energy_poss.append(models[i].forward(feat, latents[i])) # Note: Energy function.
                    energy_negs.append(models[i].forward(feat_neg.detach(), latents[i])) # + 0.001 * feat_noise
                    # Q: Understand this line.

                ## Contrastive Divergence loss.
                energy_pos = torch.stack(energy_poss, dim=1)
                energy_neg = torch.stack(energy_negs, dim=1)
                ml_loss = (energy_pos - energy_neg).mean()

                ## Energy regularization losses
                loss = loss + ml_loss #energy_pos.mean() - energy_neg.mean()
                loss = loss + (torch.pow(energy_pos, 2).mean() + torch.pow(energy_neg, 2).mean())

            ## Add to the replay buffer
            if (FLAGS.replay_batch or FLAGS.entropy_nn) and (feat_neg is not None):
                replay_buffer.add(compress_x_mod(feat_neg.detach().cpu().numpy()))

            ## Smoothness loss
            if FLAGS.sm:
                diff_feat_negs = feat_negs[:, -1, ..., 1:, :] - feat_negs[:, -1, ..., :-1, :]
                # loss_sm = torch.pow(diff_feat_negs, 2).mean()
                # Second order
                loss_sm = torch.pow(diff_feat_negs[..., 1:, :] - diff_feat_negs[..., :-1, :], 2).mean()
                loss = loss + FLAGS.sm_coeff * loss_sm
            else:
                loss_sm = torch.zeros(1).mean()

            ## KL loss
            if FLAGS.kl:
                # raise NotImplementedError
                # ix = random.randint(0, len(models) - 1)
                loss_kl = 0
                for i in range(FLAGS.components):
                    model = models[i]
                    model.requires_grad_(False)
                    loss_kl = loss_kl + model.forward(feat_neg_kl, latents[i]) # Note: Energy function.
                    model.requires_grad_(True)
                loss = loss + FLAGS.kl_coeff * loss_kl.mean()

                if FLAGS.entropy_nn:
                    bs = feat_neg_kl.size(0)

                    feat_flat = torch.clamp(feat_neg_kl.view(bs, -1), -1, 1)

                    if len(replay_buffer) > bs:
                        compare_batch, idxs = replay_buffer.sample(100) #, no_transform=False, downsample=True)
                        compare_batch = decompress_x_mod(compare_batch)
                        compare_batch = torch.Tensor(compare_batch).cuda(dev)
                        compare_flat = compare_batch.view(100, -1)
                        dist_matrix = torch.norm(feat_flat[:, None, :] - compare_flat[None, :, :], p=2, dim=-1)
                        loss_repel = torch.log(dist_matrix.min(dim=1)[0]).mean()
                    else:
                        loss_repel = torch.zeros(1).cuda(dev)
                    loss = loss - 0.3 * loss_repel
                else:
                    loss_repel = torch.zeros(1)
            else:
                loss_kl = torch.zeros(1)

            loss.backward()

            if FLAGS.gpus > 1:
                average_gradients(models)

            # grad_norm = torch.norm(feat_grad)
            # if not grad_norm_ema:
            #     grad_norm_ema = torch.norm(feat_grad)
            # else: grad_norm_ema = ema_grad_norm(grad_norm, grad_norm_ema, mu=0.99)
            # [torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) for model in models] # Note: Takes very long in debug
            # [torch.nn.utils.clip_grad_value_(model.parameters(), 0.05) for model in models] # Note: Takes very long in debug
            # [clip_grad_norm_with_ema(model, grad_norm_ema, std=0.001) for model in models]
            if it % FLAGS.log_interval == 0 and rank_idx == FLAGS.gpu_rank:
                model_grad_norm = get_model_grad_norm(models)
                model_grad_max = get_model_grad_max(models)

            [optimizer.step() for optimizer in optimizers]
            [optimizer.zero_grad() for optimizer in optimizers]

            if FLAGS.sample_ema:
                ema_model(models, models_ema, mu=0.9)

            losses.append(loss.item())
            l2_losses.append(feat_loss.item())
            if it % FLAGS.log_interval == 0 and rank_idx == FLAGS.gpu_rank:
                grad_norm = torch.norm(feat_grad)

                avg_loss = sum(losses) / len(losses)
                avg_feat_loss = sum(l2_losses) / len(l2_losses)
                losses, l2_losses = [], []

                kvs = {}
                kvs['loss'] = avg_loss

                if not FLAGS.autoencode:
                    energy_pos_mean = energy_pos.mean().item()
                    energy_neg_mean = energy_neg.mean().item()
                    energy_pos_std = energy_pos.std().item()
                    energy_neg_std = energy_neg.std().item()

                    kvs['aaa-ml_loss'] = ml_loss.item()
                    kvs['aa-energy_pos_mean'] = energy_pos_mean
                    kvs['aa-energy_neg_mean'] = energy_neg_mean
                    kvs['energy_pos_std'] = energy_pos_std
                    kvs['energy_neg_std'] = energy_neg_std

                kvs['aaa-L2_loss'] = avg_feat_loss
                kvs['latent norm'] = latent_norm.item()

                if FLAGS.kl:
                    kvs['kl_loss'] = loss_kl.mean().item()
                if FLAGS.entropy_nn:
                    kvs['entropy_loss'] = loss_repel.mean().item()
                if FLAGS.sm:
                    kvs['sm_loss'] = loss_sm.item()

                kvs['bb-max_abs_grad'] = torch.abs(feat_grad).max()
                kvs['bb-norm_grad'] = grad_norm
                # kvs['bb-norm_grad_ema'] = grad_norm_ema
                kvs['bb-norm_grad_model'] = model_grad_norm
                kvs['bb-max_abs_grad_model'] = model_grad_max


                string = "It {} ".format(it)

                for k, v in kvs.items():
                    if k in ['aaa-L2_loss', 'aaa-ml_loss', 'kl_loss']:
                        string += "%s: %.6f  " % (k,v)
                    logger.add_scalar(k, v, it)

                logger.add_figure('gen', get_trajectory_figure(feat_neg, b_idx=0, plot_type =FLAGS.plot_attr)[1], it)
                logger.add_figure('gt', get_trajectory_figure(feat, b_idx=0, plot_type =FLAGS.plot_attr)[1], it)

                test(test_dataloader, models, models_ema, FLAGS, step=it, save=False, logger=logger)

                string += 'Time: %.1fs' % (time.perf_counter()-start_time)
                print(string)
                start_time = time.perf_counter()

            if it % FLAGS.save_interval == 0 and rank_idx == FLAGS.gpu_rank:
                model_path = osp.join(logdir, "model_{}.pth".format(it))

                ckpt = {'FLAGS': FLAGS}

                for i in range(len(models)):
                    ckpt['model_state_dict_{}'.format(i)] = models[i].state_dict()

                for i in range(len(optimizers)):
                    ckpt['optimizer_state_dict_{}'.format(i)] = optimizers[i].state_dict()

                torch.save(ckpt, model_path)
                print("Saving model in directory....")
                print('run test')

                test(test_dataloader, models, models_ema, FLAGS, step=it, save=True, logger=logger)
                print('Test at step %d done!' % it)
                print('Experiment: ' + logger.log_dir.split('/')[-1])

            it += 1

def main_single(rank, FLAGS):
    rank_idx = FLAGS.node_rank * FLAGS.gpus + rank
    world_size = FLAGS.nodes * FLAGS.gpus


    if not os.path.exists('result/%s' % FLAGS.exp):
        try:
            os.makedirs('result/%s' % FLAGS.exp)
        except:
            pass


    if FLAGS.dataset == 'springs':
        dataset = SpringsParticles(FLAGS, 'train')
        test_dataset = SpringsParticles(FLAGS, 'test')
    elif FLAGS.dataset == 'charged':
        dataset = ChargedParticles(FLAGS, 'train')
        test_dataset = ChargedParticles(FLAGS, 'test')
    elif FLAGS.dataset == 'charged_sim':
        dataset = ChargedParticlesSim(FLAGS)
        test_dataset = dataset
    else:
        raise NotImplementedError
    FLAGS.timesteps = FLAGS.num_timesteps

    shuffle=True
    sampler = None

    # p = random.randint(0, 9)
    if world_size > 1:
        group = dist.init_process_group(backend='nccl', init_method='tcp://localhost:'+str(FLAGS.port), world_size=world_size, rank=rank_idx, group_name="default") # ,
        # while True:
        #     try:
        #         print('Trying port {}'.format(port))
        #         group = dist.init_process_group(backend='nccl', init_method='tcp://localhost:'+str(port), world_size=world_size, rank=rank_idx, group_name="default") # ,
        #         break
        #     except: port += 1; print('Exception'); pass
    torch.cuda.set_device(rank)
    device = torch.device('cuda')

    if FLAGS.logname == 'debug':
        logdir = osp.join(FLAGS.logdir, FLAGS.exp, FLAGS.logname)
    else:
        logdir = osp.join(FLAGS.logdir, FLAGS.exp,
                            'NO' +str(FLAGS.n_objects)
                          + '_BS' + str(FLAGS.batch_size)
                          + '_S-LR' + str(FLAGS.step_lr)
                          + '_NS' + str(FLAGS.num_steps)
                          + '_LR' + str(FLAGS.lr)
                          + '_LDim' + str(FLAGS.latent_dim)
                          + '_KL' + str(int(FLAGS.kl))
                          + '_SM' + str(int(FLAGS.sm))
                          + '_SN' + str(int(FLAGS.spectral_norm))
                          + '_Mom' + str(FLAGS.momentum)
                          + '_EMA' + str(int(FLAGS.sample_ema))
                          + '_RB' + str(int(FLAGS.replay_batch))
                          + '_AE' + str(int(FLAGS.autoencode))
                          + '_FC' + str(int(FLAGS.forecast))
                          + '_IE' + str(int(FLAGS.independent_energies))
                          + '_CDAE' + str(int(FLAGS.cd_and_ae))
                          + '_FE' + str(int(FLAGS.factor_encoder))
                          + '_NDL' + str(int(FLAGS.normalize_data_latent))
                          + '_SeqL' + str(int(FLAGS.num_timesteps))
                          + '_FSeqL' + str(int(FLAGS.num_fixed_timesteps)))
        if FLAGS.logname != '':
            logdir += '_' + FLAGS.logname

    FLAGS_OLD = FLAGS

    if FLAGS.resume_iter != 0:
        if FLAGS.resume_name is not '':
            logdir = osp.join(FLAGS.logdir, FLAGS.exp, FLAGS.resume_name)

        model_path = osp.join(logdir, "model_{}.pth".format(FLAGS.resume_iter))
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        FLAGS = checkpoint['FLAGS']

        FLAGS.normalize_data_latent = FLAGS_OLD.normalize_data_latent # Maybe we shouldn't override
        FLAGS.factor_encoder = FLAGS_OLD.factor_encoder
        FLAGS.plot_attr = FLAGS_OLD.plot_attr
        FLAGS.resume_iter = FLAGS_OLD.resume_iter
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
        FLAGS.independent_energies = FLAGS_OLD.independent_energies
        FLAGS.cd_and_ae = FLAGS_OLD.cd_and_ae
        FLAGS.num_fixed_timesteps = FLAGS_OLD.num_fixed_timesteps
        # TODO: Check what attributes we are missing
        models, optimizers  = init_model(FLAGS, device, dataset)

        state_dict = models[0].state_dict()

        if FLAGS.sample_ema:
            models_ema, _  = init_model(FLAGS, device, dataset)
            for i, (model, model_ema, optimizer) in enumerate(zip(models, models_ema, optimizers)):
                model.load_state_dict(checkpoint['model_state_dict_{}'.format(i)], strict=False)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict_{}'.format(i)])
                model_ema.load_state_dict(checkpoint['ema_model_state_dict_{}'.format(i)], strict=False)
        else:
            models_ema = None
            for i, (model, optimizer) in enumerate(zip(models, optimizers)):
                model.load_state_dict(checkpoint['model_state_dict_{}'.format(i)], strict=False)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict_{}'.format(i)])
    else:
        models, optimizers = init_model(FLAGS, device, dataset)
        if FLAGS.sample_ema:
            models_ema, _  = init_model(FLAGS, device, dataset)
            ema_model(models, models_ema, mu=0.0)
        else:  models_ema = None

    if FLAGS.gpus > 1:
        sync_model(models)

    train_dataloader = DataLoader(dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=shuffle, pin_memory=False)
    test_manipulate_dataloader = DataLoader(test_dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=False, pin_memory=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=True, pin_memory=False, drop_last=True)

    logger = SummaryWriter(logdir)
    it = FLAGS.resume_iter

    if FLAGS.train:
        models = [model.train() for model in models]
        if FLAGS.sample_ema:
            models_ema = [model_ema.train() for model_ema in models_ema]
    else:
        models = [model.eval() for model in models]
        if FLAGS.sample_ema:
            models_ema = [model_ema.eval() for model_ema in models_ema]

    if FLAGS.train:
        train(train_dataloader, test_dataloader, logger, models, models_ema, optimizers, FLAGS, logdir, rank_idx)

    elif FLAGS.test_manipulate: # TODO: What is this
        test_manipulate(test_manipulate_dataloader, models, models_ema, FLAGS, step=FLAGS.resume_iter, save=True, logger=logger)
    else:
        test(test_dataloader, models, models_ema, FLAGS, step=FLAGS.resume_iter)


def main():
    FLAGS = parser.parse_args()
    FLAGS.components = FLAGS.n_objects ** 2 - FLAGS.n_objects
    FLAGS.ensembles = FLAGS.components
    FLAGS.tie_weight = True
    FLAGS.sample = True

    logdir = osp.join(FLAGS.logdir, FLAGS.exp)

    if not osp.exists(logdir):
        os.makedirs(logdir)

    if FLAGS.gpus > 1:
        mp.spawn(main_single, nprocs=FLAGS.gpus, args=(FLAGS,))
    else:
        main_single(FLAGS.gpu_rank, FLAGS)


if __name__ == "__main__":
    main()
