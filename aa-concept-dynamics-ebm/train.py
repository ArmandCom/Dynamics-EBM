import torch
import time
from scipy.linalg import toeplitz
# from tensorflow.python.platform import flags
import numpy as np
import torch.nn.functional as F
import os
import shutil
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from dataset import TrajnetDataset, ChargedParticlesSim, ChargedParticles, ChargedSpringsParticles, SpringsParticles
from models import NodeGraphEBM_CNNOneStep_2Streams, EdgeGraphEBM_OneStep, EdgeGraphEBM_CNNOneStep, EdgeGraphEBM_CNNOneStep_Light, EdgeGraphEBM_CNN_OS_noF, NodeGraphEBM_CNNOneStep, NodeGraphEBM_CNN# TrajGraphEBM, EdgeGraphEBM, LatentEBM, ToyEBM, BetaVAE_H, LatentEBM128
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os.path as osp
import argparse
from third_party.utils import visualize_trajectories, get_trajectory_figure, \
    linear_annealing, ReplayBuffer, compress_x_mod, decompress_x_mod, accumulate_traj, \
    normalize_trajectories, augment_trajectories, MMD, align_replayed_batch, get_rel_pairs
from third_party.utils import create_masks, save_rel_matrices, smooth_trajectory, get_model_grad_norm, get_model_grad_max

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
parser.add_argument('--exp', default='', type=str, help='name of experiments')

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
# parser.add_argument('--forecast', action='store_true', help='if set to True we use L2 loss instead of Contrastive Divergence')
parser.add_argument('--forecast', default=-1, type=int, help='forecast N steps in the future (the encoder only sees the previous). -1 invalidates forecasting')
parser.add_argument('--cd_and_ae', action='store_true', help='if set to True we use L2 loss and Contrastive Divergence')
parser.add_argument('--cd_mode', default='', type=str, help='chooses between options for energy-based objectives. zeros: zeroes out some of the latents. mix: mixes latents of other samples in the batch.')

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

    optimizers = [Adam(model.parameters(), lr=FLAGS.lr) for model in models] # Note: From CVR , betas=(0.5, 0.99)
    return models, optimizers

def  gen_trajectories(latent, FLAGS, models, models_ema, feat_neg, feat, num_steps, sample=False, create_graph=True, idx=None, training_step=0, fixed_mask=None):
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
            num_steps = int(linear_annealing(None, training_step, start_step=20000, end_step=FLAGS.ns_iteration_end, start_value=num_steps, end_value=FLAGS.num_steps_end))

    if fixed_mask is None:
        fixed_mask = torch.zeros_like(feat_neg)
        if FLAGS.num_fixed_timesteps > 0:
            fixed_mask[:, :, :num_fixed_timesteps] = 1
            # #        feat_neg = torch.cat([feat_fixed[:, :, :num_fixed_timesteps],
            # #                              feat_neg[:, :,  num_fixed_timesteps:]], dim=2)
            # indices = torch.randint(feat_neg.shape[2]-num_fixed_timesteps+1,
            #                       (feat_neg.shape[0],))
            # for fixed_id in range(num_fixed_timesteps):
            #     indices += fixed_id
            #     fixed_mask[torch.arange(0,feat_neg.shape[0]), :, indices, :] = 1
            #     feat_fixed[torch.arange(0,feat_neg.shape[0]), :, indices, :2] += 0.01
            #     feat_neg = feat_fixed * (fixed_mask) + feat_neg * (1-fixed_mask)

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
        # if i % 5 == 0 and i < num_steps - 1: # smooth every 10 and leave the last iterations
        #     feat_neg = smooth_trajectory(feat_neg, 15, 5.0, 100) # ks, std = 15, 5 # x, kernel_size, std, interp_size

        # Compute energy
        latent_ii, mask = latent
        energy = 0
        curr_latent = latent
        # TODO: make more efficient --> if mask[mask == 0].sum() == 0 or mask[mask == 1].sum() == 0: # Do only 1 loop
        for ii in range(2):
            for iii in range(FLAGS.ensembles//2):
                if ii == 1:     curr_latent = (latent_ii[..., iii, :], 1 - mask)
                else:           curr_latent = (latent_ii[..., iii, :],     mask)
                energy = models[ii + 2*iii].forward(feat_neg, curr_latent) + energy
        # Get grad for current optimization iteration.
        feat_grad, = torch.autograd.grad([energy.sum()], [feat_neg], create_graph=create_graph)
        # feat_grad = torch.clamp(feat_grad, min=-0.5, max=0.5) # TODO: Remove if useless

        ## Momentum update
        if FLAGS.momentum > 0:
            update = FLAGS.step_lr * feat_grad * (1-fixed_mask) + momentum * old_update
            feat_neg = feat_neg * (1-fixed_mask) - update # GD computation
            old_update = update
        else:
            feat_neg = feat_neg * (1-fixed_mask) - FLAGS.step_lr * feat_grad * (1-fixed_mask) # GD computation

        if num_fixed_timesteps > 0:
            feat_neg = feat_fixed * (fixed_mask) + feat_neg * (1-fixed_mask)

        feat_neg = torch.clamp(feat_neg, -1, 1)
        feat_negs.append(feat_neg)
        feat_neg = feat_neg.detach()
        feat_neg.requires_grad_()  # Q: Why detaching and reataching? Is this to avoid backprop through this step?

    return feat_neg, feat_negs, energy, feat_grad

# def  gen_trajectories(latent, FLAGS, models, models_ema, feat_neg, feat, num_steps, sample=False, create_graph=True, idx=None, training_step=0, fixed_mask=None):
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
#         # Compute energy
#         latent_ii, mask = latent
#         NR = latent_ii.shape[1]
#         energy = 0
#         # TODO: make more efficient --> if mask[mask == 0].sum() == 0 or mask[mask == 1].sum() == 0: # Do only 1 loop
#         for ii in range(FLAGS.ensembles):
#             curr_latent = (latent_ii[:, ii:ii+1, 0, :].repeat(1, NR, 1), mask)
#             energy = models[ii].forward(feat_neg, curr_latent) + energy
#
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

def test_manipulate(train_dataloader, models, models_ema, FLAGS, step=0, save = False, logger = None):
    if FLAGS.cuda:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    replay_buffer = None

    print('Begin test:')
    [model.eval() for model in models]
    for (feat, edges), (rel_rec, rel_send), idx in train_dataloader:

        feat = feat.to(dev)
        nonan_mask = feat == feat
        feat[torch.logical_not(nonan_mask)] = 10
        edges = edges.to(dev)
        rel_rec = rel_rec.to(dev)
        rel_send = rel_send.to(dev)
        # What are these? im = im[:FLAGS.num_visuals], idx = idx[:FLAGS.num_visuals]
        bs = feat.size(0)
        # [save_rel_matrices(model, rel_rec, rel_send) for model in models]

        b_idx = 0
        b_idx_ref = 3

        rw_pair = None
        affected_nodes = None
        fixed_mask = None

        ### SELECT BY PAIRS ###
        # pair_id = 2
        # pairs = get_rel_pairs(rel_send, rel_rec)
        # rw_pair = pairs[pair_id]
        # affected_nodes = (rel_rec + rel_send)[0, rw_pair].mean(0).clamp_(min=0, max=1).data.cpu().numpy()

        ### SELECT BY NODES ###
        # node_id = 1
        # rw_pair = range((FLAGS.n_objects - 1)*node_id, (FLAGS.n_objects - 1)*(node_id + 1))
        # affected_nodes = torch.zeros(FLAGS.n_objects).to(dev)
        # affected_nodes[node_id] = 1

        ### Switch examples ###
        rw_pair = torch.arange(FLAGS.components).to(dev)

        ### Mask definition
        mask = torch.ones(FLAGS.components).to(dev)
        if rw_pair is not None:
            mask[rw_pair] = 0
        # mask = mask * 0
        # mask[rw_pair] = 1

        if FLAGS.forecast is not -1:
            feat_enc = feat[:, :, :-FLAGS.forecast]
        else: feat_enc = feat

        feat_enc = normalize_trajectories(feat_enc[:, :, FLAGS.num_fixed_timesteps:], augment=False, normalize=FLAGS.normalize_data_latent) # We max min normalize the batch
        latent = models[0].embed_latent(feat_enc, rel_rec, rel_send, edges=edges)
        if isinstance(latent, tuple):
            latent, weights = latent
        ### NOTE: TEST: Random rotation of the input trajectory
        # feat = normalize_trajectories(feat, augment=False, normalize=True) # We max min normalize the batch

        # fixed_idx = 2
        # feat[:, fixed_idx:fixed_idx+1, :, :] = feat[:, fixed_idx:fixed_idx+1, :1, :]
        # fixed_mask = torch.zeros_like(feat)
        # if FLAGS.num_fixed_timesteps > 0:
        #     fixed_mask[:, :, :FLAGS.num_fixed_timesteps] = 1
        #     fixed_mask[:, fixed_idx:fixed_idx+1, :, :] = 1


        # Latent codes manipulation.
        # latent = latent * mask[None, :, None]
        if rw_pair is not None:
            latent[:, rw_pair] = latent[b_idx_ref:b_idx_ref+1, rw_pair]
        # latent[:] = latent[b_idx_ref:b_idx_ref+1]
        #
        factors = [[1, 1]]
        # factors = [[1,1], [0.5, 1], [1, 0.5], [0.5, 0.5], [0, 1], [1, 0]]
        ori_latent = latent
        lims = None
        for factor_id, factor in enumerate(factors):
            # if len(factor) == 2:
            latent = torch.cat([ori_latent[..., :1, :] * factor[0],
                                ori_latent[..., 1:, :] * factor[1]], dim=-2)
            # elif len(factor) ==  1: latent = ori_latent[..., :1, :] * factor[0]
            # else: raise NotImplementedError

            latent = (latent, mask)

            feat_neg = torch.rand_like(feat) * 2 - 1


            feat_neg, feat_negs, energy_neg, feat_grad = gen_trajectories(latent, FLAGS, models, models_ema, feat_neg, feat, FLAGS.num_steps_test, FLAGS.sample,
                                                                                   create_graph=False, fixed_mask=fixed_mask) # TODO: why create_graph

            if save:
                # print('Latents: \n{}'.format(latent[0][b_idx].data.cpu().numpy()))
                # savedir = os.path.join(homedir, "result/%s/") % (FLAGS.exp)
                # Path(savedir).mkdir(parents=True, exist_ok=True)
                # savename = "s%08d"% (step)
                # visualize_trajectories(feat, feat_neg, edges, savedir = os.path.join(savedir,savename))
                limpos = limneg = 1
                if torch.logical_not(nonan_mask).sum() > 0 and lims is None:
                    lims = [-limneg, limpos]
                elif lims is None and len(factors)>1: lims = [feat_negs[-1][b_idx][..., :2].min().detach().cpu().numpy(),
                                           feat_negs[-1][b_idx][..., :2].max().detach().cpu().numpy()]
                for i_plt in range(len(feat_negs)):
                    logger.add_figure('test_manip_gen_rec', get_trajectory_figure(feat_negs[i_plt], lims=lims, b_idx=b_idx, plot_type =FLAGS.plot_attr, highlight_nodes = affected_nodes)[1], step + i_plt + 100*factor_id)
                # logger.add_figure('test_manip_gen', get_trajectory_figure(feat_neg, b_idx=b_idx, lims=lims, plot_type =FLAGS.plot_attr)[1], step)
                logger.add_figure('test_manip_gen', get_trajectory_figure(feat_negs[-1], lims=lims, b_idx=b_idx, plot_type =FLAGS.plot_attr, highlight_nodes = affected_nodes)[1], step + 100*factor_id)
                logger.add_figure('test_manip_gt', get_trajectory_figure(feat, b_idx=b_idx, lims=lims, plot_type =FLAGS.plot_attr)[1], step + 100*factor_id)
                logger.add_figure('test_manip_gt_ref', get_trajectory_figure(feat, b_idx=b_idx_ref, lims=lims, plot_type =FLAGS.plot_attr)[1], step + 100*factor_id)
                print('Plotted.')
            # elif logger is not None:
            #     l2_loss = torch.pow(feat_neg_kl[:, :,  FLAGS.num_fixed_timesteps:] - feat[:, :,  FLAGS.num_fixed_timesteps:], 2).mean()
            #     logger.add_scalar('aaa-L2_loss_test', l2_loss.item(), step)
        break
    print('test done')
    exit()

def test(train_dataloader, models, models_ema, FLAGS, mask,  step=0, save = False, logger=None, replay_buffer=None):
    if FLAGS.cuda:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    [model.eval() for model in models]
    for (feat, edges), (rel_rec, rel_send), idx in train_dataloader:

        feat = feat.to(dev)
        nonan_mask = feat == feat
        feat[torch.logical_not(nonan_mask)] = 10
        # edges = edges.to(dev)
        # rel_rec = rel_rec.to(dev)
        # rel_send = rel_send.to(dev)

        if FLAGS.forecast is not -1:
            feat_enc = feat[:, :, :-FLAGS.forecast]
        else: feat_enc = feat
        feat_enc = normalize_trajectories(feat_enc, augment=False, normalize=FLAGS.normalize_data_latent)
        # [:, :, FLAGS.num_fixed_timesteps:]
        latent = models[0].embed_latent(feat_enc, rel_rec, rel_send, edges=edges)
        if isinstance(latent, tuple):
            latent, weights = latent

        feat_neg = torch.rand_like(feat) * 2 - 1

        mask=mask[torch.randperm(FLAGS.batch_size)]
        latent = (latent, mask)

        feat_neg, feat_negs, energy_neg, feat_grad = gen_trajectories(latent, FLAGS, models, models_ema, feat_neg, feat, FLAGS.num_steps_test, FLAGS.sample,
                                                                                create_graph=False) # TODO: why create_graph

        if save:
            limpos = limneg = 1
            if torch.logical_not(nonan_mask).sum() > 0:
                lims = [-limneg, limpos]
            else: lims = None
            logger.add_figure('test_gt', get_trajectory_figure(feat, lims=lims, b_idx=0, plot_type =FLAGS.plot_attr)[1], step)
            for i_plt in range(len(feat_negs)):
                logger.add_figure('test_gen', get_trajectory_figure(feat_negs[i_plt], lims=lims, b_idx=0, plot_type =FLAGS.plot_attr)[1], step + i_plt)
            # input('a')
            if replay_buffer is not None:
                replay_buffer_path = osp.join(logger.log_dir, "rb.pt")
                torch.save(replay_buffer, replay_buffer_path)
            # print('Masks: {}'.format( latent[1][0].detach().cpu().numpy()))
            if latent[0] is not None:
                print('Latents: \n{}'.format( latent[0][0,...,0].detach().cpu().numpy()))

        elif logger is not None:
            l2_loss = torch.pow(feat_neg[:, :,  FLAGS.num_fixed_timesteps:][nonan_mask[:, :,  FLAGS.num_fixed_timesteps:]]
                                - feat[:, :,  FLAGS.num_fixed_timesteps:][nonan_mask[:, :,  FLAGS.num_fixed_timesteps:]], 2).mean()

            logger.add_scalar('aaa-L2_loss_test', l2_loss.item(), step)
        break

    [model.train() for model in models]

def train(train_dataloader, test_dataloader, logger, models, models_ema, optimizers, FLAGS, mask, logdir, rank_idx, replay_buffer=None):
    step_lr = FLAGS.step_lr
    num_steps = FLAGS.num_steps
    vel_exists = FLAGS.input_dim == 4
    it = FLAGS.resume_iter
    losses, l2_losses = [], []
    schedulers = [StepLR(optimizer, step_size=50000, gamma=0.5) for optimizer in optimizers]
    # schedulers = [CosineAnnealingLR(optimizer, T_max=350000, eta_min=2e-5, last_epoch=-1) for optimizer in optimizers]
    [optimizer.zero_grad() for optimizer in optimizers]

    replay_buffer_test = None

    dev = torch.device("cuda")

    start_time = time.perf_counter()
    for epoch in range(FLAGS.num_epoch):
        for (feat, edges), (rel_rec, rel_send), idx in train_dataloader:

            loss = 0.0
            feat = feat.to(dev)
            nonan_mask = feat == feat
            feat[torch.logical_not(nonan_mask)] = 10
            edges = edges.to(dev)
            rel_rec = rel_rec.to(dev)
            rel_send = rel_send.to(dev)
            if it == FLAGS.resume_iter: [save_rel_matrices(model, rel_rec, rel_send) for model in models]

            #### Note: Opt. 2 ####
            feat_copy = feat.clone()
            max_loc = feat[..., :2].abs().max()

            if vel_exists:
                max_vel = feat[..., 2:].abs().max()
            else: max_vel = 0
            # feat_noise = torch.randn_like(feat).detach()
            # feat_noise.normal_()
            # feat_noise[..., :2] *= max_loc
            # feat_noise[..., 2:] *= max_vel
            # feat = feat + 0.002 * feat_noise
            #### Note: TEST FEATURE ####

            if FLAGS.forecast is not -1:
                feat_enc = feat[:, :, :-FLAGS.forecast]
            else: feat_enc = feat
            feat_enc = normalize_trajectories(feat_enc, augment=True, normalize=FLAGS.normalize_data_latent) # TODO: CHECK ROTATION
            # [:, :, FLAGS.num_fixed_timesteps+torch.randint(5, (1,)):]
            rand_idx = torch.randint(len(models), (1,))
            latent = models[rand_idx].embed_latent(feat_enc, rel_rec, rel_send, edges=edges)

            if isinstance(latent, tuple):
                latent, weights = latent
                l2_w_loss = torch.pow(weights, 2).mean()
                # prod_w_loss = torch.prod(weights, 1).mean()
                # loss = loss + 0.0005 * prod_w_loss - 0.0005 * l2_w_loss
            else: l2_w_loss = 0


            feat_noise = torch.randn_like(feat).detach()
            feat_noise.normal_()
            feat_noise[..., :2] *= max_loc
            feat_noise[..., 2:] *= max_vel
            feat = feat_copy + 0.002 * feat_noise

            latent_norm = latent.norm()

            feat_neg = torch.rand_like(feat) * 2 - 1

            feat_neg_gen = feat_neg

            mask = mask[torch.randperm(FLAGS.batch_size)]
            latent = (latent, mask)

            feat_neg, feat_negs, energy_neg, feat_grad = gen_trajectories(latent, FLAGS, models, models_ema, feat_neg, feat, FLAGS.num_steps, sample=False, training_step=it)
            feat_negs = torch.stack(feat_negs, dim=1)

            ## MSE Loss
                ## Note: Backprop through all sampling
            feat_loss = torch.pow(feat_negs[:, -1, :,  FLAGS.num_fixed_timesteps:][nonan_mask[:, :,  FLAGS.num_fixed_timesteps:]]
                                  - feat[:, :,  FLAGS.num_fixed_timesteps:][nonan_mask[:, :,  FLAGS.num_fixed_timesteps:]], 2).mean()

            pow_energy_rec = torch.pow(energy_neg, 2).mean()
            loss = loss + feat_loss #+ feat_loss_l1

            ### TEST FEATURE ###
            if FLAGS.cd_and_ae and it > 30000:
                latent, mask = latent

                latent_cyc = torch.cat([latent[1:], latent[:1]], dim=0)
                latent = latent * mask[:, :, None, None] + latent_cyc * (1-mask)[:, :, None, None]
                latent = (latent, mask)

                nfts_old = FLAGS.num_fixed_timesteps; FLAGS.num_fixed_timesteps = 0
                feat_neg_gen, _, energy_neg_gen, feat_grad_gen = gen_trajectories(latent, FLAGS, models, models_ema, feat_neg_gen, feat, FLAGS.num_steps, sample=False, training_step=it)
                FLAGS.num_fixed_timesteps = nfts_old # Set fixed tsteps to old value

                energy_neg = energy_neg_gen # Note: Only for logging

                ## Energy loss.
                energy_loss = energy_neg_gen.mean() + energy_neg.mean() # Normally we would maximize it but here we are supervising with L2
                pow_energy_gen = torch.pow(energy_neg_gen, 2).mean()

                ## Energy regularization losses
                loss = loss + 0.000005 * energy_loss # HARDCODED WEIGHTS!
                loss = loss + 0.0001 * (pow_energy_gen + pow_energy_rec)
            else: loss = loss #+ pow_energy_rec


            loss_sm = torch.zeros(1).mean()
            loss_kl = torch.zeros(1)
            loss.backward()

            # if it > 30000:
            #     [torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05) for model in models] # TODO: removed. Must add back?
            if it % FLAGS.log_interval == 0 :# and rank_idx == FLAGS.gpu_rank
                model_grad_norm = get_model_grad_norm(models)
                model_grad_max = get_model_grad_max(models)

            [optimizer.step() for optimizer in optimizers]
            [optimizer.zero_grad() for optimizer in optimizers]
            [scheduler.step() for scheduler in schedulers]

            losses.append(loss.item())
            l2_losses.append(feat_loss.item())
            if it % FLAGS.log_interval == 0 : #and rank_idx == FLAGS.gpu_rank

                grad_norm = torch.norm(feat_grad)

                avg_loss = sum(losses) / len(losses)
                avg_feat_loss = sum(l2_losses) / len(l2_losses)
                losses, l2_losses = [], []

                kvs = {}
                kvs['loss'] = avg_loss
                # TODO: print learning rate.
                # # if not FLAGS.autoencode:
                #     energy_pos_mean = energy_pos.mean().item()
                #     energy_pos_std = energy_pos.std().item()
                #
                #     kvs['aaa-ml_loss'] = ml_loss.item()
                #     kvs['aa-energy_pos_mean'] = energy_pos_mean
                #     kvs['energy_pos_std'] = energy_pos_std

                energy_neg_mean = energy_neg.mean().item()
                energy_neg_std = energy_neg.std().item()

                kvs['aa-energy_neg_mean'] = energy_neg_mean
                kvs['energy_neg_std'] = energy_neg_std

                kvs['LR'] = schedulers[0].get_last_lr()[0]
                kvs['aaa-L2_loss'] = avg_feat_loss
                kvs['latent norm'] = latent_norm.item()

                if FLAGS.kl:
                    kvs['kl_loss'] = loss_kl.mean().item()

                kvs['weight_l2_loss'] = l2_w_loss

                if FLAGS.sm:
                    kvs['sm_loss'] = loss_sm.item()

                kvs['bb-max_abs_grad'] = torch.abs(feat_grad).max()
                kvs['bb-norm_grad'] = grad_norm
                kvs['bb-norm_grad_model'] = model_grad_norm
                kvs['bb-max_abs_grad_model'] = model_grad_max


                string = "It {} ".format(it)

                for k, v in kvs.items():
                    if k in ['aaa-L2_loss', 'aaa-ml_loss', 'kl_loss']:
                        string += "%s: %.6f  " % (k,v)
                    logger.add_scalar(k, v, it)

                limpos = limneg = 1
                if torch.logical_not(nonan_mask).sum() > 0:
                    lims = [-limneg, limpos]
                else: lims = None
                if it % 500 == 0:
                    for i_plt in range(feat_negs.shape[1]):
                        logger.add_figure('gen', get_trajectory_figure(feat_negs[:, i_plt], lims=lims, b_idx=0, plot_type =FLAGS.plot_attr)[1], it + i_plt)
                else: logger.add_figure('gen', get_trajectory_figure(feat_neg, lims=lims, b_idx=0, plot_type =FLAGS.plot_attr)[1], it)

                logger.add_figure('gt', get_trajectory_figure(feat, lims=lims, b_idx=0, plot_type =FLAGS.plot_attr)[1], it)
                test(test_dataloader, models, models_ema, FLAGS, mask, step=it, save=False, logger=logger, replay_buffer=replay_buffer_test)

                string += 'Time: %.1fs' % (time.perf_counter()-start_time)
                print(string)
                start_time = time.perf_counter()

            if it == FLAGS.resume_iter:
                shutil.copy(sys.argv[0], logdir + '/train_EBM_saved.py')
                # print('sysargv', sys.argv)
                shutil.copy('models.py', logdir + '/models_EBM_saved.py')

            if it % FLAGS.save_interval == 0: #  and rank_idx == FLAGS.gpu_rank
                mask = create_masks(FLAGS, dev) # Change them once overy log_interval

                model_path = osp.join(logdir, "model_{}.pth".format(it))

                ckpt = {'FLAGS': FLAGS}

                for i in range(len(models)):
                    ckpt['model_state_dict_{}'.format(i)] = models[i].state_dict()

                for i in range(len(optimizers)):
                    ckpt['optimizer_state_dict_{}'.format(i)] = optimizers[i].state_dict()

                torch.save(ckpt, model_path)
                print("Saving model in directory....")
                print('run test')

                if replay_buffer is not None:
                    replay_buffer_path = osp.join(logdir, "rb.pt")
                    torch.save(replay_buffer, replay_buffer_path)
                    # a = torch.load(replay_buffer_path)

                test(test_dataloader, models, models_ema, FLAGS, mask, step=it, save=True, logger=logger, replay_buffer=replay_buffer_test)
                print('Test at step %d done!' % it)
                exp_name = logger.log_dir.split('/')
                print('Experiment: ' + exp_name[-2] + '/' + exp_name[-1])

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
    elif FLAGS.dataset == 'charged-springs':
        dataset = ChargedSpringsParticles(FLAGS, 'train')
        test_dataset = ChargedSpringsParticles(FLAGS, 'test')
    elif FLAGS.dataset.split('_')[0] == 'trajnet':
        dataset = TrajnetDataset(FLAGS, 'train')
        test_dataset = TrajnetDataset(FLAGS, 'test')
    else:
        raise NotImplementedError
    FLAGS.timesteps = FLAGS.num_timesteps

    shuffle=True
    sampler = None
    replay_buffer = None

    # p = random.randint(0, 9)
    if world_size > 1:
        group = dist.init_process_group(backend='nccl', init_method='tcp://localhost:'+str(FLAGS.port), world_size=world_size, rank=rank_idx, group_name="default")
    torch.cuda.set_device(rank)
    device = torch.device('cuda')

    branch_folder = 'experiments'
    if FLAGS.logname == 'debug':
        logdir = osp.join(FLAGS.logdir, FLAGS.exp, FLAGS.logname)
    else:
        logdir = osp.join(FLAGS.logdir, branch_folder, FLAGS.exp,
                            'NO' +str(FLAGS.n_objects)
                          + '_BS' + str(FLAGS.batch_size)
                          + '_S-LR' + str(FLAGS.step_lr)
                          + '_NS' + str(FLAGS.num_steps)
                          + '_LR' + str(FLAGS.lr)
                          + '_LDim' + str(FLAGS.latent_dim)
                          + '_SN' + str(int(FLAGS.spectral_norm))
                          + '_AE' + str(int(FLAGS.autoencode))
                          + '_CDAE' + str(int(FLAGS.cd_and_ae))
                          + '_Mod' + str(FLAGS.model_name)
                          + '_NMod' + str(FLAGS.ensembles//2)
                          + '_Mask-' + str(FLAGS.masking_type)
                          + '_FC' + str(FLAGS.forecast)
                          + '_FE' + str(int(FLAGS.factor_encoder))
                          + '_NDL' + str(int(FLAGS.normalize_data_latent))
                          + '_SeqL' + str(int(FLAGS.num_timesteps))
                          + '_FSeqL' + str(int(FLAGS.num_fixed_timesteps)))
        if FLAGS.logname != '':
            logdir += '_' + FLAGS.logname
    FLAGS_OLD = FLAGS

    if FLAGS.resume_iter != 0:
        if FLAGS.resume_name is not '':
            logdir = osp.join(FLAGS.logdir, branch_folder, FLAGS.exp, FLAGS.resume_name)
        # logdir = osp.join(FLAGS.logdir,'charged/joint-split-onestep', FLAGS.resume_name)
        model_path = osp.join(logdir, "model_{}.pth".format(FLAGS.resume_iter))
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        FLAGS = checkpoint['FLAGS']
        # logdir += '_FC10' # TODO: remove

        # FLAGS.normalize_data_latent = FLAGS_OLD.normalize_data_latent # Maybe we shouldn't override
        # FLAGS.factor_encoder = FLAGS_OLD.factor_encoder
        # FLAGS.ensembles = FLAGS_OLD.ensembles
        FLAGS.n_objects = FLAGS_OLD.n_objects
        FLAGS.components = FLAGS_OLD.components

        FLAGS.plot_attr = FLAGS_OLD.plot_attr
        FLAGS.resume_iter = FLAGS_OLD.resume_iter
        FLAGS.save_interval = FLAGS_OLD.save_interval
        FLAGS.nodes = FLAGS_OLD.nodes
        FLAGS.gpus = FLAGS_OLD.gpus
        FLAGS.node_rank = FLAGS_OLD.node_rank
        FLAGS.train = FLAGS_OLD.train
        FLAGS.batch_size = FLAGS_OLD.batch_size
        FLAGS.num_visuals = FLAGS_OLD.num_visuals
        FLAGS.decoder = FLAGS_OLD.decoder
        FLAGS.test_manipulate = FLAGS_OLD.test_manipulate
        FLAGS.lr = FLAGS_OLD.lr
        # FLAGS.sim = FLAGS_OLD.sim
        FLAGS.exp = FLAGS_OLD.exp
        FLAGS.step_lr = FLAGS_OLD.step_lr
        FLAGS.num_steps = FLAGS_OLD.num_steps
        FLAGS.num_steps_test = FLAGS_OLD.num_steps_test
        FLAGS.forecast = FLAGS_OLD.forecast
        FLAGS.ns_iteration_end = FLAGS_OLD.ns_iteration_end
        FLAGS.num_steps_end = FLAGS_OLD.num_steps_end

        FLAGS.model_name = FLAGS_OLD.model_name
        FLAGS.masking_type = FLAGS_OLD.masking_type

        # FLAGS.autoencode = FLAGS_OLD.autoencode
        # FLAGS.entropy_nn = FLAGS_OLD.entropy_nn
        FLAGS.cd_and_ae = FLAGS_OLD.cd_and_ae
        FLAGS.num_fixed_timesteps = FLAGS_OLD.num_fixed_timesteps
        # TODO: Check what attributes we are missing

        print(FLAGS)

        models, optimizers  = init_model(FLAGS, device, dataset)

        state_dict = models[0].state_dict()

        models_ema = None
        for i, (model, optimizer) in enumerate(zip(models, optimizers)):
            model.load_state_dict(checkpoint['model_state_dict_{}'.format(i)], strict=False)
            # optimizer.load_state_dict(checkdpoint['optimizer_state_dict_{}'.format(i)])
    else:
        models, optimizers = init_model(FLAGS, device, dataset)
        models_ema = None

    if FLAGS.gpus > 1:
        sync_model(models)

    train_dataloader = DataLoader(dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=shuffle, pin_memory=False)
    test_manipulate_dataloader = DataLoader(test_dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=False, pin_memory=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=True, pin_memory=False, drop_last=True)

    logger = SummaryWriter(logdir)
    it = FLAGS.resume_iter

    mask = create_masks(FLAGS, torch.device("cuda"))

    if FLAGS.train:
        models = [model.train() for model in models]
    else:
        models = [model.eval() for model in models]

    if FLAGS.train:
        train(train_dataloader, test_dataloader, logger, models, models_ema, optimizers, FLAGS, mask, logdir, rank_idx, replay_buffer)

    elif FLAGS.test_manipulate:
        test_manipulate(test_manipulate_dataloader, models, models_ema, FLAGS, step=FLAGS.resume_iter, save=True, logger=logger)
    else:
        test(test_dataloader, models, models_ema, FLAGS, mask, step=FLAGS.resume_iter)


def main():
    FLAGS = parser.parse_args()
    FLAGS.components = FLAGS.n_objects ** 2 - FLAGS.n_objects
    FLAGS.tie_weight = True
    FLAGS.sample = True
    FLAGS.exp = FLAGS.exp + FLAGS.dataset
    logdir = osp.join(FLAGS.logdir, 'experiments', FLAGS.exp)

    if not osp.exists(logdir):
        os.makedirs(logdir)

    if FLAGS.gpus > 1:
        mp.spawn(main_single, nprocs=FLAGS.gpus, args=(FLAGS,))
    else:
        main_single(FLAGS.gpu_rank, FLAGS)


if __name__ == "__main__":
    main()
