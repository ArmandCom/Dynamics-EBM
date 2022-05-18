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
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os.path as osp
import argparse
from third_party.utils import visualize_trajectories, get_trajectory_figure, \
    linear_annealing, ReplayBuffer, compress_x_mod, decompress_x_mod, accumulate_traj, \
    normalize_trajectories, augment_trajectories, MMD, align_replayed_batch, get_rel_pairs
from third_party.utils import create_masks, save_rel_matrices, smooth_trajectory, get_model_grad_norm, get_model_grad_max
import random
from torch import nn

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

parser.add_argument('--dataset', default='charged', type=str, help='Dataset to use (intphys or others or imagenet or cubes)')
parser.add_argument('--new_dataset', default='', type=str, help='New Dataset to use (not the one used for training)')
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
parser.add_argument('--pred_only', action='store_true', help='Fix points and predict after the K used for training.')
parser.add_argument('--no_mask', action='store_true', help='No partition of the edges, all processed in one graph.')

# data
parser.add_argument('--data_workers', default=4, type=int, help='Number of different data workers to load data in parallel')
parser.add_argument('--ensembles', default=2, type=int, help='use an ensemble of models')
parser.add_argument('--model_name', default='CNNOS_Node', type=str, help='model name')
parser.add_argument('--masking_type', default='by_receiver', type=str, help='type of masking in training')

# Test
parser.add_argument('--compute_energy', action='store_true', help='Get energies')

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
    models = [model for i in range(1)]

    n_latents = FLAGS.ensembles if FLAGS.no_mask else FLAGS.ensembles//2
    models.append(nn.Sequential(
                                # nn.Linear(FLAGS.latent_dim, FLAGS.latent_dim),
                                # nn.ReLU(),
                                nn.Linear(FLAGS.latent_dim * n_latents, 1),
                                ).to(device))

    optimizers = [Adam(model.parameters(), lr=FLAGS.lr) for model in models] # Note: From CVR , betas=(0.5, 0.99)
    return models, optimizers

def sync_model(models): # Q: What is this about?
    size = float(dist.get_world_size())

    for model in models:
        for param in model.parameters():
            dist.broadcast(param.data, 0)

def edge_accuracy(preds, target):
    _, preds = preds.max(-1)
    correct = preds.float().data.eq(
        target.float().data.view_as(preds)).cpu().sum()
    return np.float(correct) / (target.size(0) * target.size(1))


def test(train_dataloader, models, models_ema, FLAGS, mask,  step=0, save = False, logger=None, replay_buffer=None):
    if FLAGS.cuda:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    test_accuracy = []

    [model.eval() for model in models]
    for (feat, edges), (rel_rec, rel_send), idx in train_dataloader:
        bs = feat.shape[0]
        feat = feat.to(dev)
        nonan_mask = feat == feat
        feat[torch.logical_not(nonan_mask)] = 10
        edges = edges.to(dev)

        if step == FLAGS.resume_iter:
            [save_rel_matrices(model, rel_rec.to(dev), rel_send.to(dev)) for model in models[:1]]
            step += 1
        # Note: If charged-springs edges is of shape [bs, 3, NO, NO]
        #   3 stands for (edges_spring, edges_charged, edge_selection_mask)
        # rel_rec = rel_rec.to(dev)
        # rel_send = rel_send.to(dev)

        if FLAGS.forecast is not -1:
            feat_enc = feat[:, :, :-FLAGS.forecast]
        else: feat_enc = feat
        feat_enc = normalize_trajectories(feat_enc[:, :, FLAGS.num_fixed_timesteps:], augment=False, normalize=FLAGS.normalize_data_latent)
        # [:, :, FLAGS.num_fixed_timesteps:]
        latent = models[0].embed_latent(feat_enc, rel_rec, rel_send, edges=edges)
        if isinstance(latent, tuple):
            latent, weights = latent

        pred_logits = models[1](latent.flatten(-2,-1))
        pred_edge = F.sigmoid(pred_logits)
        pred_edge_oh = pred_edge.clone()
        pred_edge_oh[pred_edge_oh < 0.5] = 0
        pred_edge_oh[pred_edge_oh >= 0.5] = 1
        pred_edge_oh = torch.cat([pred_edge_oh, 1-pred_edge_oh], dim=-1)
        target = edges
        acc = edge_accuracy(pred_edge_oh, target)
        test_accuracy.append(acc)

        # if save:
        #     if latent[0] is not None:
        #         print('Latents: \n{}'.format( latent[0][0,...,0].detach().cpu().numpy()))

        # elif logger is not None:
        #     l2_loss = torch.pow(feat_neg[:, :,  FLAGS.num_fixed_timesteps:][nonan_mask[:, :,  FLAGS.num_fixed_timesteps:]]
        #                         - feat[:, :,  FLAGS.num_fixed_timesteps:][nonan_mask[:, :,  FLAGS.num_fixed_timesteps:]], 2).mean()
        #
        #     logger.add_scalar('aaa-L2_loss_test', l2_loss.item(), step)

    print('Accuracy: {}'.format( sum(test_accuracy) / len(test_accuracy)))
    [model.train() for model in models]

def train(train_dataloader, test_dataloader, logger, models, models_ema, optimizers, FLAGS, mask, logdir, rank_idx, replay_buffer=None):
    step_lr = FLAGS.step_lr
    num_steps = FLAGS.num_steps
    vel_exists = FLAGS.input_dim == 4
    it = FLAGS.resume_iter
    losses = []
    schedulers = [StepLR(optimizer, step_size=50000, gamma=0.5) for optimizer in optimizers]
    # schedulers = [CosineAnnealingLR(optimizer, T_max=350000, eta_min=2e-5, last_epoch=-1) for optimizer in optimizers]
    [optimizer.zero_grad() for optimizer in optimizers]
    dev = torch.device("cuda")
    bceloss = nn.BCELoss()
    start_time = time.perf_counter()
    for epoch in range(FLAGS.num_epoch):
        print('Epoch {}\n'.format(epoch))
        test_accuracy = []
        for (feat, edges), (rel_rec, rel_send), idx in train_dataloader:
            bs = feat.shape[0]
            loss = 0.0
            feat = feat.to(dev)
            nonan_mask = feat == feat
            feat[torch.logical_not(nonan_mask)] = 10
            edges = edges.to(dev)

            if it == FLAGS.resume_iter: [save_rel_matrices(model, rel_rec.to(dev), rel_send.to(dev)) for model in models[:1]]

            if FLAGS.forecast is not -1:
                feat_enc = feat[:, :, :-FLAGS.forecast]
            else: feat_enc = feat
            feat_enc = normalize_trajectories(feat_enc[:, :, FLAGS.num_fixed_timesteps:], augment=True, normalize=FLAGS.normalize_data_latent) # TODO: CHECK ROTATION
            # [:, :, FLAGS.num_fixed_timesteps+torch.randint(5, (1,)):]
            latent = models[0].embed_latent(feat_enc, rel_rec, rel_send, edges=edges)

            if isinstance(latent, tuple):
                latent, weights = latent

            # print(latent.shape)
            pred_logits = models[1](latent.flatten(-2,-1))
            pred_edge = F.sigmoid(pred_logits)
            pred_edge_oh = pred_edge.clone()
            pred_edge_oh[pred_edge_oh < 0.5] = 0
            pred_edge_oh[pred_edge_oh >= 0.5] = 1
            pred_edge_oh = torch.cat([pred_edge_oh, 1-pred_edge_oh], dim=-1)
            target = edges
            acc = edge_accuracy(pred_edge_oh, target)
            test_accuracy.append(acc)
            loss = bceloss(pred_edge[..., 0], edges.float())
            loss.backward()
            # if it > 30000:
            #     [torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1) for model in models] # TODO: removed. Must add back?

            [optimizer.step() for optimizer in optimizers]
            [optimizer.zero_grad() for optimizer in optimizers]
            [scheduler.step() for scheduler in schedulers]

            losses.append(loss.item())
            if it % FLAGS.log_interval == 0 : #and rank_idx == FLAGS.gpu_rank

                avg_loss = sum(losses) / len(losses)
                avg_acc = sum(test_accuracy) / len(test_accuracy)

                losses, test_accuracy = [], []

                kvs = {}
                kvs['loss'] = avg_loss
                kvs['acc'] = avg_acc
                kvs['LR'] = schedulers[0].get_last_lr()[0]

                string = "It {} ".format(it)

                for k, v in kvs.items():
                    string += "%s: %.6f  " % (k,v)
                    logger.add_scalar(k, v, it)

                test(test_dataloader, models, models_ema, FLAGS, mask, step=it, save=False, logger=logger, replay_buffer=None)

                string += 'Time: %.1fs' % (time.perf_counter()-start_time)
                print(string)
                start_time = time.perf_counter()

            if it == FLAGS.resume_iter:
                shutil.copy(sys.argv[0], logdir + '/train_EBM_saved.py')
                # print('sysargv', sys.argv)
                shutil.copy('models.py', logdir + '/models_EBM_saved.py')
                shutil.copy('encoder_models.py', logdir + '/enc_models_EBM_saved.py')
                shutil.copy('dataset.py', logdir + '/dataset_saved.py')

            if it % FLAGS.save_interval == 0: #  and rank_idx == FLAGS.gpu_rank

                model_path = osp.join(logdir, "model_{}.pth".format(it))

                ckpt = {'FLAGS': FLAGS}

                ckpt['model_state_dict_{}'.format(1)] = models[1].state_dict()
                ckpt['optimizer_state_dict_{}'.format(1)] = optimizers[1].state_dict()

                torch.save(ckpt, model_path)
                print("Saving model in directory....")
                print('run test')
                print('sysargv: ', sys.argv)

                test(test_dataloader, models, models_ema, FLAGS, mask, step=it, save=True, logger=logger, replay_buffer=None)
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

    if FLAGS.new_dataset != '':
        FLAGS.dataset = FLAGS.new_dataset
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

    branch_folder = 'experiments_edge_classification'
    if FLAGS.logname == 'debug':
        logdir = osp.join(FLAGS.logdir, FLAGS.exp, FLAGS.logname)
    else:
        logdir = osp.join(FLAGS.logdir, branch_folder, FLAGS.exp,
                            'NO' +str(FLAGS.n_objects)
                          + '_BS' + str(FLAGS.batch_size)
                          + '_S-LR' + str(FLAGS.step_lr)
                          + '_NS' + str(FLAGS.num_steps)
                          + '_NSEnd' + str(FLAGS.num_steps_end)
                          + 'at{}'.format(str(int(FLAGS.ns_iteration_end/1000)) + 'k' if FLAGS.num_steps_end > 0 else '')
                          + '_LR' + str(FLAGS.lr)
                          + '_LDim' + str(FLAGS.latent_dim)
                          + '{}'.format('LN' if FLAGS.latent_ln else '')
                          + '_SN' + str(int(FLAGS.spectral_norm))
                          + '_AE' + str(int(FLAGS.autoencode))
                          + '_CDAE' + str(int(FLAGS.cd_and_ae))
                          + '_Mod' + str(FLAGS.model_name)
                          + '_NMod{}'.format(str(FLAGS.ensembles) if FLAGS.no_mask else str(FLAGS.ensembles//2))
                          + '{}'.format('+1' if FLAGS.additional_model else '')
                          + '_Mask-' + str(FLAGS.masking_type)
                          + '_NoM' + str(int(FLAGS.no_mask))
                          + '_FE' + str(int(FLAGS.factor_encoder))
                          + '_NDL' + str(int(FLAGS.normalize_data_latent))
                          + '_SeqL' + str(int(FLAGS.num_timesteps))
                          + '_FSeqL' + str(int(FLAGS.num_fixed_timesteps))
                          + '_FC' + str(FLAGS.forecast)
                          + '{}'.format('Only' if FLAGS.pred_only else '')
                          )
        if FLAGS.logname != '':
            logdir += '_' + FLAGS.logname
    FLAGS_OLD = FLAGS

    if FLAGS.resume_iter != 0:
        logdir_resume = logdir
        if FLAGS.resume_name is not '':
            logdir_resume = osp.join(FLAGS.logdir, 'experiments', FLAGS.exp, FLAGS.resume_name)
        # logdir = osp.join(FLAGS.logdir,'charged/joint-split-onestep', FLAGS.resume_name)
        model_path = osp.join(logdir_resume, "model_{}.pth".format(FLAGS.resume_iter))
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        FLAGS = checkpoint['FLAGS']
        # logdir += '_5at150' # TODO: remove

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


        # FLAGS.model_name = FLAGS_OLD.model_name
        # FLAGS.no_mask = FLAGS_OLD.no_mask
        # FLAGS.masking_type = FLAGS_OLD.masking_type
        # FLAGS.additional_model = FLAGS_OLD.additional_model
        # FLAGS.new_dataset = FLAGS_OLD.new_dataset
        # FLAGS.compute_energy = FLAGS_OLD.compute_energy
        # FLAGS.pred_only = FLAGS_OLD.pred_only
        # FLAGS.latent_ln = FLAGS_OLD.latent_ln
        FLAGS.cd_and_ae = FLAGS_OLD.cd_and_ae

        FLAGS.train = FLAGS_OLD.train

        # FLAGS.autoencode = FLAGS_OLD.autoencode
        # FLAGS.entropy_nn = FLAGS_OLD.entropy_nn
        FLAGS.cd_and_ae = FLAGS_OLD.cd_and_ae
        FLAGS.num_fixed_timesteps = FLAGS_OLD.num_fixed_timesteps
        # TODO: Check what attributes we are missing

        print(FLAGS)

        models, optimizers  = init_model(FLAGS, device, dataset)

        state_dict = models[0].state_dict()

        models_ema = None
        print(FLAGS)
        for i, (model, optimizer) in enumerate(zip(models[:1], optimizers[:1])):
            model.load_state_dict(checkpoint['model_state_dict_{}'.format(i)], strict=False)
            # optimizer.load_state_dict(checkdpoint['optimizer_state_dict_{}'.format(i)])
            for param in model.parameters():
                param.requires_grad = False
            print('Model {}/{} loaded and frozen.'.format(i+1,len(models)))
    else:
        models, optimizers = init_model(FLAGS, device, dataset)
        models_ema = None

    if FLAGS.gpus > 1:
        sync_model(models)

    if FLAGS.train:
        train_dataloader = DataLoader(dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=True, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=True, pin_memory=False, drop_last=True)

    logger = SummaryWriter(logdir)
    it = FLAGS.resume_iter

    # mask = create_masks(FLAGS, torch.device("cuda"))
    mask = None
    if FLAGS.train:
        models = [model.train() for model in models]
    else:
        models = [model.eval() for model in models]

    if FLAGS.train:
        train(train_dataloader, test_dataloader, logger, models, models_ema, optimizers, FLAGS, mask, logdir, rank_idx, replay_buffer)

    else:
        test(test_dataloader, models, models_ema, FLAGS, mask, step=FLAGS.resume_iter, save=False, logger=logger)


def main():

    FLAGS = parser.parse_args()
    if FLAGS.no_mask: FLAGS.masking_type = 'ones'

    FLAGS.components = FLAGS.n_objects ** 2 - FLAGS.n_objects
    # FLAGS.tie_weight = True
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
