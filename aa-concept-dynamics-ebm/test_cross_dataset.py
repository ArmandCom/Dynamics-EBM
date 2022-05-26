import torch
import time
from scipy.linalg import toeplitz
import numpy as np
import torch.nn.functional as F
import os
import shutil
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from dataset import TrajnetDataset, ChargedParticlesSim, ChargedParticles, ChargedSpringsParticles, SpringsParticles
from models import UnConditional, NodeGraphEBM_CNNOneStep_2Streams, EdgeGraphEBM_OneStep, EdgeGraphEBM_CNNOneStep, EdgeGraphEBM_CNNOneStep_Light, EdgeGraphEBM_CNN_OS_noF, NodeGraphEBM_CNNOneStep, NodeGraphEBM_CNN# TrajGraphEBM, EdgeGraphEBM, LatentEBM, ToyEBM, BetaVAE_H, LatentEBM128
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os.path as osp
import argparse
from third_party.utils import visualize_trajectories, get_trajectory_figure, \
    linear_annealing, ReplayBuffer, compress_x_mod, decompress_x_mod, accumulate_traj, \
    normalize_trajectories, augment_trajectories, MMD, align_replayed_batch, get_rel_pairs
from third_party.utils import create_masks, save_rel_matrices, smooth_trajectory, get_model_grad_norm, get_model_grad_max
import random

homedir = '/data/Armand/EBM/'
# port = 6021
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

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
parser.add_argument('--pred_only', action='store_true', help='Fix points and predict after the K used for training.')

# data
parser.add_argument('--data_workers', default=4, type=int, help='Number of different data workers to load data in parallel')
parser.add_argument('--ensembles', default=2, type=int, help='use an ensemble of models')
parser.add_argument('--model_name', default='CNNOS_Node', type=str, help='model name')
parser.add_argument('--masking_type', default='by_receiver', type=str, help='type of masking in training')

# EBM specific settings

# Model specific settings
parser.add_argument('--filter_dim', default=64, type=int, help='number of filters to use')
# parser.add_argument('--components', default=2, type=int, help='number of components to explain an image with')
parser.add_argument('--component_weight', action='store_true', help='optimize for weights of the components also')
parser.add_argument('--tie_weight', action='store_true', help='tie the weights between seperate models')
parser.add_argument('--optimize_mask', action='store_true', help='also optimize a segmentation mask over image')
parser.add_argument('--pos_embed', action='store_true', help='add a positional embedding to model')
parser.add_argument('--spatial_feat', action='store_true', help='use spatial latents for object segmentation')
parser.add_argument('--dropout', default=0.0, type=float, help='use spatial latents for object segmentation')
parser.add_argument('--factor_encoder', action='store_true', help='if we use message passing in the encoder')
parser.add_argument('--normalize_data_latent', action='store_true', help='if we normalize data before encoding the latents')
parser.add_argument('--obj_id_embedding', action='store_true', help='add object identifier')

# Model specific - TrajEBM
parser.add_argument('--input_dim', default=4, type=int, help='dimension of an object')
parser.add_argument('--latent_hidden_dim', default=64, type=int, help='hidden dimension of the latent')
parser.add_argument('--latent_dim', default=64, type=int, help='dimension of the latent')
parser.add_argument('--obj_id_dim', default=6, type=int, help='size of the object id embedding')
parser.add_argument('--num_fixed_timesteps', default=5, type=int, help='constraints')
parser.add_argument('--num_timesteps', default=49, type=int, help='constraints')
parser.add_argument('--latent_ln', action='store_true', help='layernorm in the latent')

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
parser.add_argument('--ns_iteration_end', default=300000, type=int, help='training iteration where the number of sampling steps reach their max.')
parser.add_argument('--num_steps_end', default=-1, type=int, help='number of sampling steps')

parser.add_argument('--additional_model', action='store_true', help='Extra unconditional model')

parser.add_argument('--sample', action='store_true', help='generate negative samples through Langevin')
parser.add_argument('--decoder', action='store_true', help='decoder for model')

# Distributed training hyperparameters
parser.add_argument('--nodes', default=1, type=int, help='number of nodes for training')
parser.add_argument('--gpus', default=1, type=int, help='number of gpus per nodes')
parser.add_argument('--node_rank', default=0, type=int, help='rank of node')
parser.add_argument('--gpu_rank', default=0, type=int, help='number of gpus per nodes')

# python train.py --num_steps=3 --num_steps_test 5 --step_lr=10.0 --dataset=trajnet_orca_five_synth --cuda --train --batch_size=24 --latent_dim=8 --data_workers=4 --gpus=1 --gpu_rank 1 --normalize_data_latent --num_fixed_timesteps 1 --autoencode --logname cnnosNodek3 --n_objects 5 --factor --num_timesteps 21 --spectral_norm --input_dim 2

def init_model(FLAGS, device, dataset):

    if FLAGS.model_name == 'CNNOS_Edge':
        modelname = EdgeGraphEBM_CNNOneStep
    elif FLAGS.model_name == 'CNNOS_Edge_Light':
        modelname = EdgeGraphEBM_CNNOneStep_Light
    elif FLAGS.model_name == 'CNNOS_Node_Light':
        modelname = NodeGraphEBM_CNNOneStep
    elif FLAGS.model_name == 'CNNOS_Node':
        modelname = NodeGraphEBM_CNN
    elif FLAGS.model_name == 'CNNOS_Edge_noFactor':
        modelname = EdgeGraphEBM_CNN_OS_noF
    elif FLAGS.model_name == 'OS_Edge':
        modelname = EdgeGraphEBM_OneStep
    elif FLAGS.model_name == 'CNNOS_Node_2S':
        modelname = NodeGraphEBM_CNNOneStep_2Streams

    else: raise NotImplementedError

    # Option 1: All same model
    model = modelname(FLAGS, dataset).to(device)
    models = [model for i in range(FLAGS.ensembles)]

    # Option 2: All different models
    # models = [modelname(FLAGS, dataset).to(device) for i in range(FLAGS.ensembles)]

    # Option 3: Same model per complementary masks, different across latents
    # models = []
    # [models.extend([modelname(FLAGS, dataset).to(device)]*2)for i in range(FLAGS.ensembles//2)]

    if FLAGS.additional_model:
        models.append(UnConditional(FLAGS, dataset).to(device))

    optimizers = [Adam(model.parameters(), lr=FLAGS.lr) for model in models] # Note: From CVR , betas=(0.5, 0.99)
    return models, optimizers

def forward_pass_models(models, feat_in, latent, FLAGS):
    ## Match between loaded and executed.
    # M L  Mod   Ckpt ---> Seems OK
    # 0 0   0    0
    # 0 1   2    2
    # 1 0   1    0
    # 1 1   3    2
    # Compute energy
    latent_ii, mask = latent
    energy = 0
    for ii in range(2):
        for iii in range(FLAGS.ensembles//2):
            if ii == 1:     curr_latent = (latent_ii[..., iii, :], 1 - mask)
            else:           curr_latent = (latent_ii[..., iii, :],     mask)
            energy = models[ii + 2*iii].forward(feat_in, curr_latent) + energy
    if FLAGS.additional_model:
        print('additional')
        energy = models[-1].forward(feat_in) + energy
    return energy

def  gen_trajectories(latent, FLAGS, models, models_ema, feat_neg, feat, num_steps, sample=False, create_graph=True, idx=None, training_step=0, energy_mask=None, fixed_mask=None):
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


    # impute = False
    # if not impute:
    feat_negs = [feat_neg]

    feat_neg = feat_neg[:, :, num_fixed_timesteps:]

    feat_noise = torch.randn_like(feat_neg).detach()

    feat_neg.requires_grad_(requires_grad=True) # noise image [b, n_o, T, f]
    feat_fixed = feat.clone() #.requires_grad_(requires_grad=False)

    # TODO: Sample from buffer.
    s = feat.size()

    # Num steps linear annealing
    if not sample:
        if FLAGS.num_steps_end != -1 and training_step > 0:
            num_steps = int(linear_annealing(None, training_step, start_step=20000, end_step=FLAGS.ns_iteration_end, start_value=num_steps, end_value=FLAGS.num_steps_end))

    feat_neg_kl = None

    for i in range(num_steps):
        feat_noise.normal_()

        if FLAGS.num_fixed_timesteps > 0:
            feat_in = torch.cat([feat[:, :, :num_fixed_timesteps], feat_neg], dim=2)

        ## Step - LR
        if FLAGS.step_lr_decay_factor != 1.0:
            step_lr = linear_annealing(None, i, start_step=5, end_step=num_steps-1, start_value=ini_step_lr, end_value=FLAGS.step_lr_decay_factor * ini_step_lr)

        ## Add Noise
        if FLAGS.noise_decay_factor != 1.0:
            noise_coef = linear_annealing(None, i, start_step=0, end_step=num_steps-1, start_value=ini_noise_coef, end_value=FLAGS.step_lr_decay_factor * ini_noise_coef)
        if noise_coef > 0:
            feat_neg = feat_neg + noise_coef * feat_noise

        # Smoothing
        # if i % 5 == 0 and i < num_steps - 1: # smooth every 10 and leave the last iterations
        #     feat_neg = smooth_trajectory(feat_neg, 15, 5.0, 100) # ks, std = 15, 5 # x, kernel_size, std, interp_size

        # Compute energy
        energy = forward_pass_models(models, feat_in, latent, FLAGS)

        if energy_mask is not None:
            energy = energy * energy_mask[None]

        # Get grad for current optimization iteration.
        feat_grad, = torch.autograd.grad([energy.sum()], [feat_neg], create_graph=create_graph)

        # feat_grad = torch.clamp(feat_grad, min=-0.5, max=0.5) # TODO: Remove if useless

        feat_neg = feat_neg - FLAGS.step_lr * feat_grad # GD computation

        feat_neg = torch.clamp(feat_neg, -1, 1) # TODO: put back on
        if FLAGS.num_fixed_timesteps > 0:
            feat_out = torch.cat([feat[:, :, :num_fixed_timesteps], feat_neg], dim=2)
        else: feat_out = feat_neg
        feat_negs.append(feat_out)
        feat_neg = feat_neg.detach()
        feat_neg.requires_grad_()  # Q: Why detaching and reataching? Is this to avoid backprop through this step?


    return feat_out, feat_negs, energy, feat_grad

# def  gen_trajectories(latent, FLAGS, models, models_ema, feat_neg, feat, num_steps, sample=False, create_graph=True, idx=None, training_step=0, energy_mask=None, fixed_mask=None):
#
#     feat_noise = torch.randn_like(feat_neg).detach()
#     feat_negs_samples = []
#
#     num_fixed_timesteps = FLAGS.num_fixed_timesteps
#
#     ## Step
#     step_lr = FLAGS.step_lr
#     ini_step_lr = step_lr
#
#     ## Noise
#     noise_coef = FLAGS.noise_coef
#     ini_noise_coef = noise_coef
#
#     ## Momentum parameters
#     momentum = FLAGS.momentum
#     old_update = 0.0
#
#     feat_negs = [feat_neg]
#
#     feat_neg.requires_grad_(requires_grad=True) # noise image [b, n_o, T, f]
#     feat_fixed = feat.clone() #.requires_grad_(requires_grad=False)
#
#     # TODO: Sample from buffer.
#     s = feat.size()
#
#     # Num steps linear annealing
#     if not sample:
#         if FLAGS.num_steps_end != -1 and training_step > 0:
#             num_steps = int(linear_annealing(None, training_step, start_step=20000, end_step=FLAGS.ns_iteration_end, start_value=num_steps, end_value=FLAGS.num_steps_end))
#
#     if fixed_mask is None:
#         fixed_mask = torch.zeros_like(feat_neg)
#         if FLAGS.num_fixed_timesteps > 0:
#             fixed_mask[:, :, :num_fixed_timesteps] = 1
#             # #        feat_neg = torch.cat([feat_fixed[:, :, :num_fixed_timesteps],
#             # #                              feat_neg[:, :,  num_fixed_timesteps:]], dim=2)
#             # indices = torch.randint(feat_neg.shape[2]-num_fixed_timesteps+1,
#             #                       (feat_neg.shape[0],))
#             # for fixed_id in range(num_fixed_timesteps):
#             #     indices += fixed_id
#             #     fixed_mask[torch.arange(0,feat_neg.shape[0]), :, indices, :] = 1
#             #     feat_fixed[torch.arange(0,feat_neg.shape[0]), :, indices, :2] += 0.01
#             #     feat_neg = feat_fixed * (fixed_mask) + feat_neg * (1-fixed_mask)
#
#     feat_neg_kl = None
#
#     for i in range(num_steps):
#         feat_noise.normal_()
#
#         ## Step - LR
#         if FLAGS.step_lr_decay_factor != 1.0:
#             step_lr = linear_annealing(None, i, start_step=5, end_step=num_steps-1, start_value=ini_step_lr, end_value=FLAGS.step_lr_decay_factor * ini_step_lr)
#         ## Add Noise
#         if FLAGS.noise_decay_factor != 1.0:
#             noise_coef = linear_annealing(None, i, start_step=0, end_step=num_steps-1, start_value=ini_noise_coef, end_value=FLAGS.step_lr_decay_factor * ini_noise_coef)
#         feat_neg = feat_neg + noise_coef * feat_noise
#
#         # Smoothing
#         # if i % 5 == 0 and i < num_steps - 1: # smooth every 10 and leave the last iterations
#         #     feat_neg = smooth_trajectory(feat_neg, 15, 5.0, 100) # ks, std = 15, 5 # x, kernel_size, std, interp_size
#
#         ## Match between loaded and executed.
#         # M L  Mod   Ckpt ---> Seems OK
#         # 0 0   0    0
#         # 0 1   2    2
#         # 1 0   1    0
#         # 1 1   3    2
#         # Compute energy
#         latent_ii, mask = latent
#         energy = 0
#         curr_latent = latent
#         # TODO: make more efficient --> if mask[mask == 0].sum() == 0 or mask[mask == 1].sum() == 0: # Do only 1 loop
#         for ii in range(2):
#             for iii in range(FLAGS.ensembles//2):
#                 if ii == 1:     curr_latent = (latent_ii[..., iii, :], 1 - mask)
#                 else:           curr_latent = (latent_ii[..., iii, :],     mask)
#                 # print(ii, iii, ii+2*iii)
#                 energy = models[ii + 2*iii].forward(feat_neg, curr_latent) + energy
#                 if energy_mask is not None:
#                     energy = energy * energy_mask[None]
#         # Get grad for current optimization iteration.
#         feat_grad, = torch.autograd.grad([energy.sum()], [feat_neg], create_graph=create_graph)
#         # feat_grad = torch.clamp(feat_grad, min=-0.5, max=0.5) # TODO: Remove if useless
#
#         ## Momentum update
#         if FLAGS.momentum > 0:
#             update = FLAGS.step_lr * feat_grad * (1-fixed_mask) + momentum * old_update
#             feat_neg = feat_neg * (1-fixed_mask) - update # GD computation
#             old_update = update
#         else:
#             feat_neg = feat_neg * (1-fixed_mask) - FLAGS.step_lr * feat_grad * (1-fixed_mask) # GD computation
#
#         if num_fixed_timesteps > 0:
#             feat_neg = feat_fixed * (fixed_mask) + feat_neg * (1-fixed_mask)
#
#         feat_neg = torch.clamp(feat_neg, -1, 1)
#         feat_negs.append(feat_neg)
#         feat_neg = feat_neg.detach()
#         feat_neg.requires_grad_()  # Q: Why detaching and reataching? Is this to avoid backprop through this step?
#
#     return feat_neg, feat_negs, energy, feat_grad

def sync_model(models): # Q: What is this about?
    size = float(dist.get_world_size())

    for model in models:
        for param in model.parameters():
            dist.broadcast(param.data, 0)

def test_manipulate(dataloaders, models, FLAGS, step=0, save = False, logger = None):
    print(FLAGS)
    switch_ini = False

    # print(FLAGS)
    if FLAGS.cuda:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    replay_buffer = None

    print('Begin test:')
    (feat_1, edges_1), (rel_rec, rel_send), idx = next(iter(dataloaders[0]))
    (feat_2, edges_2), _, _                     = next(iter(dataloaders[1]))


    while True:
        batchs = input('Batch id S, id C: ').split(',')
        b_idx_1 = int(batchs[0]) % FLAGS.batch_size
        b_idx_2 = int(batchs[1]) % FLAGS.batch_size
        print('Batch ids: S-{}, C-{}'.format(b_idx_1, b_idx_2))

        node_colors =  ['b', 'r', 'c', 'y', 'k', 'm', 'g']
        pairs = get_rel_pairs(rel_send, rel_rec)
        print('Pairs:')
        for pair_id in range(100):
            try:
                affected_nodes = (rel_rec + rel_send)[0, pairs[pair_id]].mean(0).clamp_(min=0, max=1).data.cpu().numpy()
                print('{}:{}'.format(pair_id,[node_colors[i] for i in range(len(affected_nodes)) if affected_nodes[i] == 1]), end=', ')
            except: break
        print('')
        inp = input('Base dataset: (S: 0, C: 1), Switch ID, mode (Node: n, Edge Pair: ep): ').split(',')
        ini_id, switch_id, mode = int(inp[0]), int(inp[1]), inp[2]
        feats = [feat_1[b_idx_1:b_idx_1+1].to(dev), feat_2[b_idx_2:b_idx_2+1].to(dev)]
        rel_rec = rel_rec[:1].to(dev)
        rel_send = rel_send[:1].to(dev)
        bs = feat_1.size(0)
        [save_rel_matrices(model, rel_rec, rel_send) for model in models]

        rw_pair = None
        affected_nodes = None
        lims = None
        energy_mask = None

        # node colors =  ['b', 'r', 'c', 'y', 'k', 'm', 'g']

        ### Mask energy by node
        # node_mask_id = 3
        # energy_mask = torch.ones(FLAGS.n_objects).to(dev) * 0.7
        # energy_mask[node_mask_id] = 1

        ### SELECT BY PAIRS ###
        if switch_id is not -1 and (mode == 'ep' or mode == 'n'):
            if mode == 'ep':
                pair_id = switch_id
                pairs = get_rel_pairs(rel_send, rel_rec)
                rw_pair = pairs[pair_id]
                # print(pairs, '\n', rel_send,'\n', rel_rec)
                affected_nodes = (rel_rec + rel_send)[0, rw_pair].mean(0).clamp_(min=0, max=1).data.cpu().numpy()

            ### SELECT BY NODES ###
            if mode == 'n':
                node_id = switch_id
                rw_pair = range((FLAGS.n_objects - 1)*node_id, (FLAGS.n_objects - 1)*(node_id + 1))
                affected_nodes = torch.zeros(FLAGS.n_objects).to(dev)
                affected_nodes[node_id] = 1

            node_colors =  ['b', 'r', 'c', 'y', 'k', 'm', 'g']
            print('Affected nodes: {}'.format([node_colors[i] for i in range(len(affected_nodes)) if affected_nodes[i] == 1]))

        ### Mask definition
        mask = torch.ones(FLAGS.components).to(dev)
        if rw_pair is not None:
            mask[rw_pair] = 0

        if ini_id == 1:
            mask = mask * 0
            if rw_pair is not None:
                mask[rw_pair] = 1


        latents = []
        nonan_masks = []
        for idx in range(2):
            nonan_masks.append(feats[idx] == feats[idx])
            feats[idx][torch.logical_not(nonan_masks[idx])] = 10
            if FLAGS.forecast is not -1:
                feat_enc = feats[idx][:, :, :-FLAGS.forecast]
            else: feat_enc = feats[idx]
            feat_enc = normalize_trajectories(feat_enc[:, :, FLAGS.num_fixed_timesteps:], augment=False, normalize=FLAGS.normalize_data_latent) # We max min normalize the batch
            latents.append(models[idx].embed_latent(feat_enc, rel_rec, rel_send))

        latent_mix = latents[0] * mask[None, :, None, None] + latents[1] * (1-mask)[None, :, None, None]
        latent = (latent_mix, mask)
        if isinstance(latents[0], tuple):
            raise NotImplementedError

        # TODO: Initialization of nodes.
        #  Currently the initialization is taken from one of the datasets
        if switch_ini:
            ini_id = 1 - ini_id
        feat = feats[ini_id]
        nonan_mask = nonan_masks[ini_id]

        ### NOTE: TEST: Random rotation of the input trajectory
        # feat = normalize_trajectories(feat, augment=True, normalize=False) # We max min normalize the batch

        if FLAGS.pred_only:
            feat = feat[: ,:, -FLAGS.forecast:]
            nonan_mask = feat == feat

        feat_neg = torch.rand_like(feats[0]) * 2 - 1
        feat_neg, feat_negs, energy, feat_grad = gen_trajectories(latent, FLAGS, models, None, feat_neg, feat, FLAGS.num_steps_test, FLAGS.sample,
                                                                    energy_mask=energy_mask, create_graph=False, fixed_mask=None)
        # feat_negs = torch.stack(feat_negs, dim=1) # 5 iterations only

        # TODO: Save files if good, to be able to visualize them later.
        # print('Latent: \n{}'.format(latent[0][b_idx].data.cpu().numpy()))
        limpos = limneg = 1
        if torch.logical_not(nonan_mask).sum() > 0 and lims is None:
            lims = [-limneg, limpos]

        for i_plt in range(len(feat_negs)):
            logger.add_figure('test_manip_gen_rec', get_trajectory_figure(feat_negs[i_plt][:, :, :-FLAGS.forecast], lims=lims, b_idx=0, plot_type =FLAGS.plot_attr, highlight_nodes = affected_nodes)[1], i_plt)
        # logger.add_figure('test_manip_gen', get_trajectory_figure(feat_neg, b_idx=b_idx, lims=lims, plot_type =FLAGS.plot_attr)[1], step)
        logger.add_figure('test_manip_gt_1', get_trajectory_figure(feats[0][:, :, :-FLAGS.forecast], b_idx=0, lims=lims, plot_type =FLAGS.plot_attr, highlight_nodes = affected_nodes)[1], 0)
        logger.add_figure('test_manip_gt_2', get_trajectory_figure(feats[1][:, :, :-FLAGS.forecast], b_idx=0, lims=lims, plot_type =FLAGS.plot_attr, highlight_nodes = affected_nodes)[1], 0)
        logger.add_figure('test_manip_gt_2', get_trajectory_figure(feats[1][:, :, :-FLAGS.forecast], b_idx=0, lims=lims, plot_type =FLAGS.plot_attr, highlight_nodes = affected_nodes)[1], 0)
        print('Plotted.')
        print('test done')

        end = input('Exit: e; Continue: c, Save with name: s+(anything else)')
        if end[0] == 's':
            savedir = os.path.join('/',*(logger.log_dir.split('/')[:-3]),'results/test_recombine/')
            savedir += end[1:] + '_SC_recomb_'
            np.save(savedir + 'affected_nodes.npy', affected_nodes)
            np.save(savedir + 'all_samples.npy', torch.stack(feat_negs).detach().cpu().numpy())
            np.save(savedir + 'gt_springs.npy', feats[0].detach().cpu().numpy())
            np.save(savedir + 'gt_charged.npy', feats[1].detach().cpu().numpy())
            print('All saved in dir {}'.format(savedir))
            exit()
        elif end == 'e':
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
    branch_folder = 'experiments'
    for ckpt_idx, (resume_iter, resume_name) in enumerate(zip(FLAGS.resume_iter, FLAGS.resume_name)):
        if resume_iter != 0:
            if resume_name is not '':
                logdir = osp.join(FLAGS.logdir, branch_folder, FLAGS_OLD.exp[ckpt_idx], resume_name)
            else: print('Must provide checkpoint names.'); exit()
            model_path = osp.join(logdir, "model_{}.pth".format(resume_iter))
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

            if ckpt_idx == 0:
                FLAGS = checkpoint['FLAGS']

                FLAGS.n_objects = FLAGS_OLD.n_objects
                FLAGS.components = FLAGS_OLD.components
                # FLAGS.normalize_data_latent = FLAGS_OLD.normalize_data_latent # Maybe we shouldn't override
                # FLAGS.factor_encoder = FLAGS_OLD.factor_encoder
                FLAGS.plot_attr = FLAGS_OLD.plot_attr
                FLAGS.resume_iter = FLAGS_OLD.resume_iter
                # FLAGS.resume_name = FLAGS_OLD.resume_name
                FLAGS.save_interval = FLAGS_OLD.save_interval
                FLAGS.nodes = FLAGS_OLD.nodes
                FLAGS.gpus = FLAGS_OLD.gpus
                FLAGS.node_rank = FLAGS_OLD.node_rank
                FLAGS.train = FLAGS_OLD.train
                FLAGS.batch_size = FLAGS_OLD.batch_size
                # FLAGS.num_visuals = FLAGS_OLD.num_visuals
                # FLAGS.num_additional = FLAGS_OLD.num_additional
                FLAGS.decoder = FLAGS_OLD.decoder
                FLAGS.test_manipulate = FLAGS_OLD.test_manipulate
                # FLAGS.ensembles = FLAGS_OLD.ensembles
                # FLAGS.sim = FLAGS_OLD.sim
                FLAGS.lr = FLAGS_OLD.lr
                # FLAGS.exp = FLAGS_OLD.exp
                FLAGS.step_lr = FLAGS_OLD.step_lr
                FLAGS.num_steps = FLAGS_OLD.num_steps
                FLAGS.num_steps_test = FLAGS_OLD.num_steps_test
                FLAGS.forecast = FLAGS_OLD.forecast
                FLAGS.ns_iteration_end = FLAGS_OLD.ns_iteration_end
                FLAGS.num_steps_end = FLAGS_OLD.num_steps_end

                # FLAGS.model_name = FLAGS_OLD.model_name
                # FLAGS.additional_model = FLAGS_OLD.additional_model
                # FLAGS.new_dataset = FLAGS_OLD.new_dataset
                # FLAGS.compute_energy = FLAGS_OLD.compute_energy
                # FLAGS.pred_only = FLAGS_OLD.pred_only
                FLAGS.latent_ln = FLAGS_OLD.latent_ln
                FLAGS.masking_type = FLAGS_OLD.masking_type
                FLAGS.no_mask = FLAGS_OLD.no_mask
                # FLAGS.autoencode = FLAGS_OLD.autoencode
                # FLAGS.entropy_nn = FLAGS_OLD.entropy_nn
                FLAGS.cd_and_ae = FLAGS_OLD.cd_and_ae
                FLAGS.num_fixed_timesteps = FLAGS_OLD.num_fixed_timesteps
                # TODO: Check what attributes we are missing
                models, _ = init_model(FLAGS, device, dataset)

            # Note: We load the first learned model for each of the datasets
            for ii in range(FLAGS.ensembles//2):
                # M i Ckpt
                # 0 0 0
                # 2 1 2
                # 1 0 0
                # 3 1 2
                models[ckpt_idx + ii*2].load_state_dict(checkpoint['model_state_dict_{}'.format(ii*2)], strict=False)
        else: print('Must provide checkpoint resume iteration.'); exit()

    if FLAGS.gpus > 1:
        sync_model(models)

    test_manipulate_dataloader_1 = DataLoader(datasets[0], num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=False, pin_memory=False, drop_last=True)
    test_manipulate_dataloader_2 = DataLoader(datasets[1], num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=False, pin_memory=False, drop_last=True)
    dataloaders = (test_manipulate_dataloader_1, test_manipulate_dataloader_2)


    logdir = osp.join(FLAGS.logdir, 'experiments_recombine',  FLAGS_OLD.exp[0]+'_'+FLAGS_OLD.exp[1], FLAGS_OLD.resume_name[0])
    logger = SummaryWriter(logdir)

    models = [model.eval() for model in models]
    test_manipulate(dataloaders, models, FLAGS, step=FLAGS.resume_iter, save=True, logger=logger)

def main():
    FLAGS = parser.parse_args()
    FLAGS.components = FLAGS.n_objects ** 2 - FLAGS.n_objects
    # FLAGS.ensembles = 1
    # FLAGS.tie_weight = True
    FLAGS.sample = True
    FLAGS.no_mask = False
    FLAGS.exp = FLAGS.dataset

    assert len(FLAGS.exp) == len(FLAGS.resume_name) == len(FLAGS.resume_iter) == 2
    logdir = osp.join(FLAGS.logdir, 'experiments_recombine', FLAGS.exp[0] + '_' + FLAGS.exp[1])

    if not osp.exists(logdir):
        os.makedirs(logdir)

    if FLAGS.gpus > 1:
        mp.spawn(main_single, nprocs=FLAGS.gpus, args=(FLAGS,))
    else:
        main_single(FLAGS.gpu_rank, FLAGS)


if __name__ == "__main__":
    main()
