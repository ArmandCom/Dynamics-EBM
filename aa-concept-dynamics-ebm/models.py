from torch.nn import ModuleList
import math
import torch.nn.functional as F
import torch.nn as nn
import torch
from easydict import EasyDict
from downsample import Downsample
import numpy as np
import warnings
from torch.nn.utils import spectral_norm as sn
from encoder_models import MLPLatentEncoder, CNNLatentEncoder, CNNEmbeddingLatentEncoder, my_softmax
from itertools import groupby

warnings.filterwarnings("ignore")

def swish(x):
    return x * torch.sigmoid(x)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class EdgeGraphEBM_LateFusion(nn.Module):
    def __init__(self, args, dataset):
        super(EdgeGraphEBM_LateFusion, self).__init__()
        do_prob = args.dropout
        self.dropout_prob = do_prob

        filter_dim = args.filter_dim
        self.filter_dim = filter_dim
        latent_dim = args.latent_dim
        spectral_norm = args.spectral_norm

        self.factor = True

        state_dim = args.input_dim

        self.obj_id_embedding = args.obj_id_embedding
        if args.obj_id_embedding:
            state_dim += args.obj_id_dim
            self.obj_embedding = NodeID(args)

        self.cnn = CNNBlock(state_dim * 2, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)

        # self.mlp2 = MLPBlock(filter_dim, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)
        # self.mlp3 = MLPBlock(filter_dim * 3, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)

        if spectral_norm:
            self.energy_map = sn(nn.Linear(filter_dim, 1))
            self.mlp2 = sn(nn.Linear(filter_dim, filter_dim))
            self.mlp3 = sn(nn.Linear(filter_dim * 3, filter_dim))
        else:
            self.energy_map = nn.Linear(filter_dim, 1)
            self.mlp2 = nn.Linear(filter_dim, filter_dim)
            self.mlp3 = nn.Linear(filter_dim * 3, filter_dim)

        self.layer_cnn_encode = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, rescale=False, spectral_norm=spectral_norm)
        self.layer1_cnn = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, downsample=False, rescale=False, spectral_norm=spectral_norm)

        self.layer1 = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        self.layer2 = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        self.layer3 = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)

        if self.factor:
            print("Using factor graph CNN encoder.")
        else:
            print("Using CNN encoder.")

        self.init_weights()

        # New
        if args.forecast:
            timesteps_enc = args.num_fixed_timesteps
        else: timesteps_enc = args.timesteps
        # self.latent_encoder = MLPLatentEncoder(timesteps_enc * args.input_dim, args.latent_hidden_dim, args.latent_dim,
        #                                        do_prob=args.dropout, factor=True)
        self.latent_encoder = CNNLatentEncoder(state_dim, args.latent_hidden_dim, args.latent_dim,
                                               do_prob=args.dropout, factor=True)

        self.rel_rec = None
        self.rel_send = None

    def embed_latent(self, traj, rel_rec, rel_send):
        if self.rel_rec is None and self.rel_send is None:
            self.rel_rec, self.rel_send = rel_rec[0:1], rel_send[0:1]

        if self.obj_id_embedding:
            traj = self.obj_embedding(traj)

        return self.latent_encoder(traj, rel_rec, rel_send)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge_temporal(self, inputs, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.

        x = inputs.reshape(inputs.size(0), inputs.size(1), -1)

        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.reshape(inputs.size(0) * receivers.size(1),
                                      inputs.size(2), inputs.size(3))
        receivers = receivers.transpose(2, 1)

        senders = torch.matmul(rel_send, x)
        senders = senders.reshape(inputs.size(0) * senders.size(1),
                                  inputs.size(2),
                                  inputs.size(3))
        senders = senders.transpose(2, 1)

        # receivers and senders have shape:
        # [num_sims * num_edges, num_dims, num_timesteps]
        edges = torch.cat([senders, receivers], dim=1)
        return edges

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.transpose(2, 1), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, latent):

        rel_rec = self.rel_rec #.repeat_interleave(inputs.size(0), dim=0)
        rel_send = self.rel_send #.repeat_interleave(inputs.size(0), dim=0)

        if self.obj_id_embedding:
            inputs = self.obj_embedding(inputs)

        if isinstance(latent, tuple):
            latent, mask = latent
        else: mask = None

        BS, NO, _, ND = inputs.shape
        NR = NO * (NO-1)

        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send) #[N, 4, T] --> [R, 8, T] # Marshalling
        x = swish(self.cnn(edges))  # [R, 8, T] --> [R, F, T'] # CNN layers

        # Unconditional pass for the rest of edges
        x = self.layer_cnn_encode(x, latent=None) # [R, F, T'] --> [R, F, T'']
        x = self.layer1_cnn(x, latent=None) # [R, F, T'] --> [R, F, T'']

        # Join all edges with the edge of interest
        x = x.mean(-1) # [R, F, T'''] --> [R, F] # Temporal Avg pool
        x = x.reshape(BS, NR, x.size(-1))

        x_skip = x

        x = self.edge2node(x, rel_rec, rel_send) # [R, F] --> [N, F] # marshalling
        x = swish(self.mlp2(x)) # [N, F] --> [N, F]

        x = self.node2edge(x, rel_rec, rel_send) # [N, F] --> [R, 2F] # marshalling
        x = torch.cat((x, x_skip), dim=2)  # [R, 2F] --> [R, 3F] # Skip connection
        x = swish(self.mlp3(x)) # [R, 3F] --> [R, F]

        x = self.layer1(x, latent) # [R, F] --> [R, F] # Conditioning layer
        x = self.layer2(x, latent) # [R, F] --> [R, F] # Conditioning layer
        x = self.layer3(x, latent) # [R, F] --> [R, F] # Conditioning layer

        energy = self.energy_map(x).squeeze(-1) # [R, F] --> [R, 1] # Project features to scalar

        if mask is not None:
            if len(mask.shape) < 2:
                mask = mask[None, :]
            energy = energy * mask

        return energy

class EdgeGraphEBM_CNNOneStep(nn.Module):
    def __init__(self, args, dataset):
        super(EdgeGraphEBM_CNNOneStep, self).__init__()
        do_prob = args.dropout
        self.dropout_prob = do_prob

        filter_dim = args.filter_dim
        self.filter_dim = filter_dim
        latent_dim = args.latent_dim
        spectral_norm = args.spectral_norm

        self.factor = True
        self.num_time_instances = 3
        self.stride = 1
        state_dim = args.input_dim

        self.obj_id_embedding = args.obj_id_embedding
        if args.obj_id_embedding:
            state_dim += args.obj_id_dim
            self.obj_embedding = NodeID(args)

        self.cnn = CNNBlock(state_dim * 2, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)

        self.mlp1_trans = MLPBlock(filter_dim * self.num_time_instances, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)
        # self.mlp2 = MLPBlock(filter_dim, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)
        # self.mlp3 = MLPBlock(filter_dim * 3, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)

        if spectral_norm:
            self.energy_map_cnn = sn(nn.Linear(filter_dim, 1))
            self.energy_map_trans = sn(nn.Linear(filter_dim, 1))
            self.mlp2 = sn(nn.Linear(filter_dim, filter_dim))
            self.mlp3 = sn(nn.Linear(filter_dim * 3, filter_dim))
            self.mlp2_trans = sn(nn.Linear(filter_dim, filter_dim))
            self.mlp3_trans = sn(nn.Linear(filter_dim * 3, filter_dim))
        else:
            self.energy_map_cnn = nn.Linear(filter_dim, 1)
            self.energy_map_trans = nn.Linear(filter_dim, 1)
            self.mlp2 = nn.Linear(filter_dim, filter_dim)
            self.mlp3 = nn.Linear(filter_dim * 3, filter_dim)
            self.mlp2_trans = nn.Linear(filter_dim, filter_dim)
            self.mlp3_trans = nn.Linear(filter_dim * 3, filter_dim)

        self.layer_cnn_encode = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, rescale=False, spectral_norm=spectral_norm)
        self.layer1_cnn = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, downsample=False, rescale=False, spectral_norm=spectral_norm)
        self.layer1 = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        self.layer2 = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        # self.layer3 = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)

        self.layer1_trans = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        self.layer2_trans = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        self.layer3_trans = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        self.layer4_trans = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)

        if self.factor:
            print("Using factor graph CNN encoder.")
        else:
            print("Using CNN encoder.")

        self.init_weights()

        # New
        if args.forecast:
            timesteps_enc = args.num_fixed_timesteps
        else: timesteps_enc = args.timesteps
        # self.latent_encoder = MLPLatentEncoder(timesteps_enc * args.input_dim, args.latent_hidden_dim, args.latent_dim,
        #                                        do_prob=args.dropout, factor=True)
        # self.latent_encoder = CNNLatentEncoder(state_dim, args.latent_hidden_dim, args.latent_dim,
        #                                        do_prob=args.dropout, factor=True)
        self.latent_encoder = CNNEmbeddingLatentEncoder(state_dim, args.latent_hidden_dim, args.latent_dim,
                                                        do_prob=args.dropout, factor=True)

        self.rel_rec = None
        self.rel_send = None
        self.ones_mask = None

    def embed_latent(self, traj, rel_rec, rel_send):
        if self.rel_rec is None and self.rel_send is None:
            self.rel_rec, self.rel_send = rel_rec[0:1], rel_send[0:1]
        if self.obj_id_embedding:
            traj = self.obj_embedding(traj)

        return self.latent_encoder(traj, rel_rec, rel_send)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge_temporal(self, inputs, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.

        x = inputs.reshape(inputs.size(0), inputs.size(1), -1)

        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.reshape(inputs.size(0) * receivers.size(1),
                                      inputs.size(2), inputs.size(3))
        receivers = receivers.transpose(2, 1)

        senders = torch.matmul(rel_send, x)
        senders = senders.reshape(inputs.size(0) * senders.size(1),
                                  inputs.size(2),
                                  inputs.size(3))
        senders = senders.transpose(2, 1)

        # receivers and senders have shape:
        # [num_sims * num_edges, num_dims, num_timesteps]
        edges = torch.cat([senders, receivers], dim=1)
        return edges

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.transpose(2, 1), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, latent):

        rel_rec = self.rel_rec
        rel_send = self.rel_send
        BS, NO, T, ND = inputs.shape
        NR = NO * (NO - 1)

        if self.obj_id_embedding:
            inputs = self.obj_embedding(inputs)
            ND = inputs.shape[-1]

        if isinstance(latent, tuple):
            latent, mask = latent
            if len(mask.shape) < 2:
                mask = mask[None]
        else:
            if self.ones_mask is None:
                self.ones_mask = torch.ones((1, 1)).to(inputs.device)
            mask = self.ones_mask

        if latent is not None:
            latent = latent * mask[..., None]

        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send) #[N, 4, T] --> [R, 8, T] # Marshalling
        edges_cnn = swish(self.cnn(edges))  # [R, 8, T] --> [R, F, T'] # CNN layers

        ### Note: CNN features ###
        # Unconditional pass for the rest of edges
        x = self.layer_cnn_encode(edges_cnn, latent=latent) # [R, F, T'] --> [R, F, T'']
        x = self.layer1_cnn(x, latent=latent) # [R, F, T'] --> [R, F, T'']

        # Join all edges with the edge of interest
        x = x.mean(-1) # [R, F, T'''] --> [R, F] # Temporal Avg pool
        x = x.reshape(BS, NR, x.size(-1))

        # Mask before message passing
        # x = x * mask[:, :, None]

        # Skip connection
        x_skip = x

        x = self.edge2node(x, rel_rec, rel_send) # [R, F] --> [N, F] # marshalling
        x = swish(self.mlp2(x)) # [N, F] --> [N, F]

        x = self.node2edge(x, rel_rec, rel_send) # [N, F] --> [R, 2F] # marshalling
        x = torch.cat((x, x_skip), dim=2)  # [R, 2F] --> [R, 3F] # Skip connection
        x = swish(self.mlp3(x)) # [R, 3F] --> [R, F]

        x = self.layer1(x, latent) # [R, F] --> [R, F] # Conditioning layer
        out_cnn = self.layer2(x, latent) # [R, F] --> [R, F] # Conditioning layer
        # out_cnn = self.layer3(x, latent) # [R, F] --> [R, F] # Conditioning layer

        energy_cnn = self.energy_map_cnn(out_cnn).squeeze(-1) # [R, F] --> [R, 1] # Project features to scalar

        ### Note: Pairwise / Transitions ###
        edges_unfold = edges_cnn.unfold(-1, self.num_time_instances, self.stride)
        _, _, NC, NTI = edges_unfold.shape
        edges_unfold = edges_unfold.reshape(BS, NR, -1, NC, NTI).permute(0, 3, 1, 2, 4)

        x = swish(self.mlp1_trans(edges_unfold.reshape(BS, NC*NR, -1))).reshape(BS, NC, NR, -1)  # [R, 8 * NTI] --> [R, F] # CNN layers

        # # Unconditional pass for the rest of edges
        x = self.layer1_trans(x, latent)
        x = self.layer2_trans(x, latent)
        # x = swish(self.mlp2_trans(x))
        # x = x * mask[:, None, :, None]

        # Join all edges with the edge of interest
        x = x.reshape(BS*NC, NR, x.size(-1))

        x_skip = x

        x = self.edge2node(x, rel_rec, rel_send) # [R, F] --> [N, F] # marshalling
        x = swish(self.mlp2_trans(x)) # [N, F] --> [N, F] # TODO share mlps?

        x = self.node2edge(x, rel_rec, rel_send) # [N, F] --> [R, 2F] # marshalling
        x = torch.cat((x, x_skip), dim=2)  # [R, 2F] --> [R, 3F] # Skip connection
        x = swish(self.mlp3_trans(x)) # [R, 3F] --> [R, F]

        x = self.layer1(x.reshape(BS, NC, NR, -1), latent=latent) # [R, F, T'] --> [R, F, T'']
        x = self.layer2(x, latent=latent) # [R, F, T'] --> [R, F, T'']
        # x = self.layer3(x, latent=latent) # [R, F] --> [R, F] # Conditioning layer

        energy_trans = self.energy_map_trans(x).squeeze(-1).mean(1) # [R, F] --> [R, 1] # Project features to scalar

        energy = torch.stack([energy_cnn, energy_trans]) * mask[None]

        return energy

class EdgeGraphEBM_OneStep(nn.Module):
    def __init__(self, args, dataset):
        super(EdgeGraphEBM_OneStep, self).__init__()
        do_prob = args.dropout
        self.dropout_prob = do_prob

        filter_dim = args.filter_dim
        self.filter_dim = filter_dim
        latent_dim = args.latent_dim
        spectral_norm = args.spectral_norm

        self.factor = True
        self.global_feats = False
        self.num_time_instances = 3
        self.stride = 1
        state_dim = args.input_dim

        self.obj_id_embedding = args.obj_id_embedding
        if args.obj_id_embedding:
            state_dim += args.obj_id_dim
            self.obj_embedding = NodeID(args)

        if self.global_feats:
            # self.cnn = CNNBlock(state_dim * 2, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)
            self.layer4 = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm, downsample=False, kernel_size=3)

        self.mlp1 = MLPBlock(filter_dim * self.num_time_instances * 2, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)
        # self.mlp3 = MLPBlock(filter_dim * 3, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)

        if spectral_norm:
            self.energy_map = sn(nn.Linear(filter_dim, 1))
            self.cnn_bf = sn(nn.Conv1d(state_dim * 2, filter_dim, kernel_size=3, padding=1))
            self.cnn_pre_bf = sn(nn.Conv1d(state_dim * 2, filter_dim, kernel_size=3, padding=1))
            self.mlp1_2 = sn(nn.Linear(filter_dim, filter_dim))
            self.mlp2 = sn(nn.Linear(filter_dim, filter_dim))
            self.mlp3 = sn(nn.Linear(filter_dim * 3, filter_dim))
            if self.global_feats:
                self.cnn = sn(nn.Conv1d(state_dim * 2, filter_dim, kernel_size=3, padding=1))
                self.energy_map_glob = sn(nn.Linear(filter_dim, 1))
        else:
            self.energy_map = nn.Linear(filter_dim, 1)
            self.cnn_bf = nn.Conv1d(state_dim * 2, filter_dim, kernel_size=3, padding=1)
            self.cnn_pre_bf = nn.Conv1d(state_dim * 2, filter_dim, kernel_size=3, padding=1)
            self.mlp1_2 = nn.Linear(filter_dim, filter_dim)
            self.mlp2 = nn.Linear(filter_dim, filter_dim)
            self.mlp3 = nn.Linear(filter_dim * 3, filter_dim)
            if self.global_feats:
                self.cnn = nn.Conv1d(state_dim * 2, filter_dim, kernel_size=3, padding=1)
                self.energy_map_global = nn.Linear(filter_dim, 1)

        self.layer1_cnn_bf = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, downsample=False, rescale=False, spectral_norm=spectral_norm)
        self.layer2_cnn_bf = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, downsample=False, rescale=False, spectral_norm=spectral_norm)

        self.layer1 = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        self.layer2 = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        self.layer3 = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)

        # self.layer1_bf = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        # self.layer2_bf = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)

        if self.factor:
            print("Using factor graph CNN encoder.")
        else:
            print("Using CNN encoder.")

        self.init_weights()

        # New
        if args.forecast:
            timesteps_enc = args.num_fixed_timesteps
        else: timesteps_enc = args.timesteps
        # self.latent_encoder = MLPLatentEncoder(timesteps_enc * args.input_dim, args.latent_hidden_dim, args.latent_dim,
        #                                        do_prob=args.dropout, factor=True)
        self.latent_encoder = CNNLatentEncoder(state_dim, args.latent_hidden_dim, args.latent_dim,
                                               do_prob=args.dropout, factor=True)

        self.rel_rec = None
        self.rel_send = None

    def embed_latent(self, traj, rel_rec, rel_send):
        if self.rel_rec is None and self.rel_send is None:
            self.rel_rec, self.rel_send = rel_rec[0:1], rel_send[0:1]

        if self.obj_id_embedding:
            traj = self.obj_embedding(traj)

        return self.latent_encoder(traj, rel_rec, rel_send)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge_temporal(self, inputs, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.

        x = inputs.reshape(inputs.size(0), inputs.size(1), -1)

        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.reshape(inputs.size(0) * receivers.size(1),
                                      inputs.size(2), inputs.size(3))
        receivers = receivers.transpose(2, 1)

        senders = torch.matmul(rel_send, x)
        senders = senders.reshape(inputs.size(0) * senders.size(1),
                                  inputs.size(2),
                                  inputs.size(3))
        senders = senders.transpose(2, 1)

        # receivers and senders have shape:
        # [num_sims * num_edges, num_dims, num_timesteps]
        edges = torch.cat([senders, receivers], dim=1)
        return edges

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.transpose(2, 1), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, latent):

        rel_rec = self.rel_rec #.repeat_interleave(inputs.size(0), dim=0)
        rel_send = self.rel_send #.repeat_interleave(inputs.size(0), dim=0)
        BS, NO, T, ND = inputs.shape
        NR = NO * (NO - 1)

        if self.obj_id_embedding:
            inputs = self.obj_embedding(inputs)
            ND = inputs.shape[-1]

        if isinstance(latent, tuple):
            latent, mask = latent
            if len(mask.shape) < 2:
                mask = mask[None]
        else: mask = torch.ones((1, 1)).to(inputs.device)

        # NTI: number of time instances

        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send) #[N, 4, T] --> [R, 8, T] # Marshalling

        # edges_pre = swish(self.cnn_pre_bf(edges))  # [R, 8, T] --> [R, F, T'] # CNN layers # Note: removed

        # Convolutional Conditioning
        edges = swish(self.cnn_bf(edges))  # [R, 8, T] --> [R, F, T'] # CNN layers
        edges = self.layer1_cnn_bf(edges, latent)
        edges = self.layer2_cnn_bf(edges, latent)

        # edges_cat = torch.cat([edges[..., :-1], edges_pre[..., 1:]], dim=-2) # Note: removed
        edges_cat = edges

        # TODO: Implement multi-resolution.
        edges_unfold = edges_cat.unfold(-1, self.num_time_instances, self.stride)
        _, _, NC, NTI = edges_unfold.shape
        edges_unfold = edges_unfold.reshape(BS, NR, -1, NC, NTI).permute(0, 3, 1, 2, 4)


        x = swish(self.mlp1(edges_unfold.reshape(BS, NC*NR, -1))).reshape(BS, NC, NR, -1)  # [R, 8 * NTI] --> [R, F] # CNN layers

        # TODO: message passing to each individual step before concat to the previous stage. (In couples)
        # Unconditional pass for the rest of edges
        # x = self.layer1_bf(x, latent)
        # x = self.layer2_bf(x, latent)
        # x = swish(self.mlp1_2(x))
        x = x * mask[:, None, :, None]

        # Join all edges with the edge of interest
        x = x.reshape(BS*NC, NR, x.size(-1))

        x_skip = x

        x = self.edge2node(x, rel_rec, rel_send) # [R, F] --> [N, F] # marshalling
        x = swish(self.mlp2(x)) # [N, F] --> [N, F]

        x = self.node2edge(x, rel_rec, rel_send) # [N, F] --> [R, 2F] # marshalling
        x = torch.cat((x, x_skip), dim=2)  # [R, 2F] --> [R, 3F] # Skip connection
        x = swish(self.mlp3(x)) # [R, 3F] --> [R, F]

        x = self.layer1(x.reshape(BS, NC, NR, -1), latent=latent) # [R, F, T'] --> [R, F, T'']
        x = self.layer2(x, latent=latent) # [R, F, T'] --> [R, F, T'']
        # x = self.layer3(x, latent=latent) # [R, F] --> [R, F] # Conditioning layer

        energy = self.energy_map(x).squeeze(-1).mean(1) # [R, F] --> [R, 1] # Project features to scalar

        # if self.global_feats:
        #     x = swish(self.cnn(edges))  # [R, 8, T] --> [R, F, T'] # CNN layers
        #     x = self.layer4(x, latent=latent) # [R, F] --> [R, F] # Conditioning layer
        #     x = x.mean(-1).reshape(BS, NR, -1)
        #     energy = energy + self.energy_map_global(x).squeeze(-1) # [R, F] --> [R, 1] # Project features to scalar

        energy = energy * mask

        return energy

class EdgeGraphEBM_CNN_OS_noF(nn.Module):
    def __init__(self, args, dataset):
        super(EdgeGraphEBM_CNN_OS_noF, self).__init__()
        do_prob = args.dropout
        self.dropout_prob = do_prob

        filter_dim = args.filter_dim
        self.filter_dim = filter_dim
        latent_dim = args.latent_dim
        spectral_norm = args.spectral_norm

        self.factor = True
        self.num_time_instances = 3
        self.stride = 1
        state_dim = args.input_dim

        self.obj_id_embedding = args.obj_id_embedding
        if args.obj_id_embedding:
            state_dim += args.obj_id_dim
            self.obj_embedding = NodeID(args)

        self.cnn = CNNBlock(state_dim * 2, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)

        self.mlp1_trans = MLPBlock(filter_dim * self.num_time_instances, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)
        # self.mlp2 = MLPBlock(filter_dim, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)
        # self.mlp3 = MLPBlock(filter_dim * 3, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)

        if spectral_norm:
            self.energy_map_cnn = sn(nn.Linear(filter_dim, 1))
            self.energy_map_trans = sn(nn.Linear(filter_dim, 1))
            self.mlp2 = sn(nn.Linear(filter_dim, filter_dim))
            # self.mlp3 = sn(nn.Linear(filter_dim * 3, filter_dim))
            # self.mlp2_trans = sn(nn.Linear(filter_dim, filter_dim))
            # self.mlp3_trans = sn(nn.Linear(filter_dim * 3, filter_dim))
        else:
            self.energy_map_cnn = nn.Linear(filter_dim, 1)
            self.energy_map_trans = nn.Linear(filter_dim, 1)
            self.mlp2 = nn.Linear(filter_dim, filter_dim)
            # self.mlp3 = nn.Linear(filter_dim * 3, filter_dim)
            # self.mlp2_trans = nn.Linear(filter_dim, filter_dim)
            # self.mlp3_trans = nn.Linear(filter_dim * 3, filter_dim)

        self.layer_cnn_encode = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, rescale=False, spectral_norm=spectral_norm)
        self.layer1_cnn = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, downsample=False, rescale=False, spectral_norm=spectral_norm)
        self.layer1 = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        # self.layer2 = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        # self.layer3 = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)

        self.layer1_trans = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        self.layer2_trans = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        self.layer3_trans = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        # self.layer4_trans = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)

        if self.factor:
            print("Using factor graph CNN encoder.")
        else:
            print("Using CNN encoder.")

        self.init_weights()

        # New
        if args.forecast:
            timesteps_enc = args.num_fixed_timesteps
        else: timesteps_enc = args.timesteps
        # self.latent_encoder = MLPLatentEncoder(timesteps_enc * args.input_dim, args.latent_hidden_dim, args.latent_dim,
        #                                        do_prob=args.dropout, factor=True)
        # self.latent_encoder = CNNLatentEncoder(state_dim, args.latent_hidden_dim, args.latent_dim,
        #                                        do_prob=args.dropout, factor=True)
        self.latent_encoder = CNNEmbeddingLatentEncoder(state_dim, args.latent_hidden_dim, args.latent_dim,
                                               do_prob=args.dropout, factor=True)
        self.rel_rec = None
        self.rel_send = None
        self.ones_mask = None

    def embed_latent(self, traj, rel_rec, rel_send):
        if self.rel_rec is None and self.rel_send is None:
            self.rel_rec, self.rel_send = rel_rec[0:1], rel_send[0:1]

        if self.obj_id_embedding:
            traj = self.obj_embedding(traj)

        return self.latent_encoder(traj, rel_rec, rel_send)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge_temporal(self, inputs, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.

        x = inputs.reshape(inputs.size(0), inputs.size(1), -1)

        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.reshape(inputs.size(0) * receivers.size(1),
                                      inputs.size(2), inputs.size(3))
        receivers = receivers.transpose(2, 1)

        senders = torch.matmul(rel_send, x)
        senders = senders.reshape(inputs.size(0) * senders.size(1),
                                  inputs.size(2),
                                  inputs.size(3))
        senders = senders.transpose(2, 1)

        # receivers and senders have shape:
        # [num_sims * num_edges, num_dims, num_timesteps]
        edges = torch.cat([senders, receivers], dim=1)
        return edges

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.transpose(2, 1), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, latent):

        rel_rec = self.rel_rec
        rel_send = self.rel_send
        BS, NO, T, ND = inputs.shape
        NR = NO * (NO - 1)

        if self.obj_id_embedding:
            inputs = self.obj_embedding(inputs)
            ND = inputs.shape[-1]

        if isinstance(latent, tuple):
            latent, mask = latent
            if len(mask.shape) < 2:
                mask = mask[None]
        else:
            if self.ones_mask is None:
                self.ones_mask = torch.ones((1, 1)).to(inputs.device)
            mask = self.ones_mask

        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send) #[N, 4, T] --> [R, 8, T] # Marshalling
        edges_cnn = swish(self.cnn(edges))  # [R, 8, T] --> [R, F, T'] # CNN layers

        ### Note: CNN features ###
        # Unconditional pass for the rest of edges
        x = self.layer_cnn_encode(edges_cnn, latent=latent) # [R, F, T'] --> [R, F, T'']
        x = self.layer1_cnn(x, latent=latent) # [R, F, T'] --> [R, F, T'']

        # Join all edges with the edge of interest
        x = x.mean(-1) # [R, F, T'''] --> [R, F] # Temporal Avg pool

        x = self.mlp2(x)
        x = x.reshape(BS, NR, x.size(-1))
        x = self.layer1(x, latent) # [R, F] --> [R, F] # Conditioning layer
        out_cnn = x * mask[:, :, None]
        energy_cnn = self.energy_map_cnn(out_cnn).squeeze(-1) # [R, F] --> [R, 1] # Project features to scalar

        ### Note: Pairwise / Transitions ###
        edges_unfold = edges_cnn.unfold(-1, self.num_time_instances, self.stride)
        _, _, NC, NTI = edges_unfold.shape
        edges_unfold = edges_unfold.reshape(BS, NR, -1, NC, NTI).permute(0, 3, 1, 2, 4)

        x = swish(self.mlp1_trans(edges_unfold.reshape(BS, NC*NR, -1))).reshape(BS, NC, NR, -1)  # [R, 8 * NTI] --> [R, F] # CNN layers

        # # Unconditional pass for the rest of edges
        x = self.layer1_trans(x, latent)
        x = self.layer2_trans(x, latent)
        x = self.layer3_trans(x, latent)
        x = x * mask[:, None, :, None]
        energy_trans = self.energy_map_trans(x).squeeze(-1).mean(1) # [R, F] --> [R, 1] # Project features to scalar

        energy = torch.stack([energy_cnn, energy_trans])

        return energy

# class RNNdec(nn.Module):
#     """Recurrent decoder module."""
#
#     def __init__(self, n_in_node, edge_types, n_hid,
#                  do_prob=0., skip_first=False):
#         super(RNNdec, self).__init__()
#         self.msg_fc1 = nn.ModuleList(
#             [nn.Linear(2 * n_hid, n_hid) for _ in range(edge_types)])
#         self.msg_fc2 = nn.ModuleList(
#             [nn.Linear(n_hid, n_hid) for _ in range(edge_types)])
#         self.msg_out_shape = n_hid
#         self.skip_first_edge_type = skip_first
#
#         self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
#         self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
#         self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)
#
#         self.input_r = nn.Linear(n_in_node, n_hid, bias=True)
#         self.input_i = nn.Linear(n_in_node, n_hid, bias=True)
#         self.input_n = nn.Linear(n_in_node, n_hid, bias=True)
#
#         self.out_fc1 = nn.Linear(n_hid, n_hid)
#         self.out_fc2 = nn.Linear(n_hid, n_hid)
#         self.out_fc3 = nn.Linear(n_hid, n_in_node)
#
#         print('Using learned recurrent interaction net decoder.')
#
#         self.dropout_prob = do_prob
#
#     def single_step_forward(self, inputs, rel_rec, rel_send,
#                             rel_type, hidden):
#
#         # node2edge
#         receivers = torch.matmul(rel_rec, hidden)
#         senders = torch.matmul(rel_send, hidden)
#         pre_msg = torch.cat([senders, receivers], dim=-1)
#
#         all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
#                                         self.msg_out_shape))
#         if inputs.is_cuda:
#             all_msgs = all_msgs.cuda()
#
#         if self.skip_first_edge_type:
#             start_idx = 1
#             norm = float(len(self.msg_fc2)) - 1.
#         else:
#             start_idx = 0
#             norm = float(len(self.msg_fc2))
#
#         # Run separate MLP for every edge type
#         # NOTE: To exlude one edge type, simply offset range by 1
#         for i in range(start_idx, len(self.msg_fc2)):
#             msg = F.tanh(self.msg_fc1[i](pre_msg))
#             msg = F.dropout(msg, p=self.dropout_prob)
#             msg = F.tanh(self.msg_fc2[i](msg))
#             msg = msg * rel_type[:, :, i:i + 1]
#             all_msgs += msg / norm
#
#         agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2,
#                                                                         -1)
#         agg_msgs = agg_msgs.contiguous() / inputs.size(2)  # Average
#
#         # GRU-style gated aggregation
#         r = F.sigmoid(self.input_r(inputs) + self.hidden_r(agg_msgs))
#         i = F.sigmoid(self.input_i(inputs) + self.hidden_i(agg_msgs))
#         n = F.tanh(self.input_n(inputs) + r * self.hidden_h(agg_msgs))
#         hidden = (1 - i) * n + i * hidden
#
#         # Output MLP
#         pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout_prob)
#         pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
#         pred = self.out_fc3(pred)
#
#         # Predict position/velocity difference
#         pred = inputs + pred
#
#         return pred, hidden
#
#     def forward(self, data, rel_type, rel_rec, rel_send, pred_steps=1,
#                 burn_in=False, burn_in_steps=1, dynamic_graph=False,
#                 encoder=None, temp=None):
#
#         inputs = data.transpose(1, 2).contiguous()
#
#         time_steps = inputs.size(1)
#
#         # inputs has shape
#         # [batch_size, num_timesteps, num_atoms, num_dims]
#
#         # rel_type has shape:
#         # [batch_size, num_atoms*(num_atoms-1), num_edge_types]
#
#         hidden = Variable(
#             torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape))
#         if inputs.is_cuda:
#             hidden = hidden.cuda()
#
#         pred_all = []
#
#         for step in range(0, inputs.size(1) - 1):
#
#             if burn_in:
#                 if step <= burn_in_steps:
#                     ins = inputs[:, step, :, :]
#                 else:
#                     ins = pred_all[step - 1]
#             else:
#                 assert (pred_steps <= time_steps)
#                 # Use ground truth trajectory input vs. last prediction
#                 if not step % pred_steps:
#                     ins = inputs[:, step, :, :]
#                 else:
#                     ins = pred_all[step - 1]
#
#             if dynamic_graph and step >= burn_in_steps:
#                 # NOTE: Assumes burn_in_steps = args.timesteps
#                 logits = encoder(
#                     data[:, :, step - burn_in_steps:step, :].contiguous(),
#                     rel_rec, rel_send)
#                 rel_type = gumbel_softmax(logits, tau=temp, hard=True)
#
#             pred, hidden = self.single_step_forward(ins, rel_rec, rel_send,
#                                                     rel_type, hidden)
#             pred_all.append(pred)
#
#         preds = torch.stack(pred_all, dim=1)
#
#         return preds.transpose(1, 2).contiguous()
#
# class dec(nn.Module):
#     """MLP decoder module."""
#
#     def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid,
#                  do_prob=0., skip_first=False):
#         super(dec, self).__init__()
#         self.msg_fc1 = nn.ModuleList(
#             [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
#         self.msg_fc2 = nn.ModuleList(
#             [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
#         self.msg_out_shape = msg_out
#         self.skip_first_edge_type = skip_first
#
#         self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
#         self.out_fc2 = nn.Linear(n_hid, n_hid)
#         self.out_fc3 = nn.Linear(n_hid, n_in_node)
#
#         print('Using learned interaction net decoder.')
#
#         self.dropout_prob = do_prob
#
#     def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
#                             single_timestep_rel_type):
#
#         # single_timestep_inputs has shape
#         # [batch_size, num_timesteps, num_atoms, num_dims]
#
#         # single_timestep_rel_type has shape:
#         # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]
#
#         # Node2edge
#         receivers = torch.matmul(rel_rec, single_timestep_inputs)
#         senders = torch.matmul(rel_send, single_timestep_inputs)
#         pre_msg = torch.cat([senders, receivers], dim=-1)
#
#         all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
#                                         pre_msg.size(2), self.msg_out_shape))
#         if single_timestep_inputs.is_cuda:
#             all_msgs = all_msgs.cuda()
#
#         if self.skip_first_edge_type:
#             start_idx = 1
#         else:
#             start_idx = 0
#
#         # Run separate MLP for every edge type
#         # NOTE: To exlude one edge type, simply offset range by 1
#         for i in range(start_idx, len(self.msg_fc2)):
#             msg = F.relu(self.msg_fc1[i](pre_msg))
#             msg = F.dropout(msg, p=self.dropout_prob)
#             msg = F.relu(self.msg_fc2[i](msg))
#             msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
#             all_msgs += msg
#
#         # Aggregate all msgs to receiver
#         agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
#         agg_msgs = agg_msgs.contiguous()
#
#         # Skip connection
#         aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)
#
#         # Output MLP
#         pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
#         pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
#         pred = self.out_fc3(pred)
#
#         # Predict position/velocity difference
#         return single_timestep_inputs + pred
#
#     def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
#         # NOTE: Assumes that we have the same graph across all samples.
#
#         inputs = inputs.transpose(1, 2).contiguous()
#
#         sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1),
#                  rel_type.size(2)]
#         rel_type = rel_type.unsqueeze(1).expand(sizes)
#
#         time_steps = inputs.size(1)
#         assert (pred_steps <= time_steps)
#         preds = []
#
#         # Only take n-th timesteps as starting points (n: pred_steps)
#         last_pred = inputs[:, 0::pred_steps, :, :]
#         curr_rel_type = rel_type[:, 0::pred_steps, :, :]
#         # NOTE: Assumes rel_type is constant (i.e. same across all time steps).
#
#         # Run n prediction steps
#         for step in range(0, pred_steps):
#             last_pred = self.single_step_forward(last_pred, rel_rec, rel_send,
#                                                  curr_rel_type)
#             preds.append(last_pred)
#
#         sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
#                  preds[0].size(2), preds[0].size(3)]
#
#         output = Variable(torch.zeros(sizes))
#         if inputs.is_cuda:
#             output = output.cuda()
#
#         # Re-assemble correct timeline
#         for i in range(len(preds)):
#             output[:, i::pred_steps, :, :] = preds[i]
#
#         pred_all = output[:, :(inputs.size(1) - 1), :, :]
#
#         return pred_all.transpose(1, 2).contiguous()

class NodeID(nn.Module):
    def __init__(self, args):
        super(NodeID, self).__init__()
        self.obj_id_dim = args.obj_id_dim
        # dim_id = int(np.ceil(np.log2(args.n_objects)))
        self.index_embedding = nn.Embedding(args.n_objects, args.obj_id_dim)
        # obj_ids = [torch.tensor(np.array(list(np.binary_repr(i, width=dim_id)), dtype=int)) for i in range(args.n_objects)]
        # obj_ids = torch.stack(obj_ids)
        obj_ids = torch.arange(args.n_objects)
        self.register_buffer('obj_ids', obj_ids, persistent=False)

    def forward(self, states):
        embeddings = self.index_embedding(self.obj_ids)[None, :, None].expand(*states.shape[:-1], self.obj_id_dim)
        return torch.cat([states, embeddings], dim=-1)

# Support Modules
class CondResBlock1d(nn.Module):
    def __init__(self, downsample=True, rescale=False, filters=64, latent_dim=64, im_size=64, kernel_size=5, latent_grid=False, spectral_norm=False):
        super(CondResBlock1d, self).__init__()

        self.filters = filters
        self.latent_dim = latent_dim
        self.im_size = im_size
        self.downsample = downsample
        self.latent_grid = latent_grid

        # TODO: Why not used?
        if filters <= 128: # Q: Why this condition?
            self.bn1 = nn.InstanceNorm1d(filters, affine=False)
        else:
            self.bn1 = nn.GroupNorm(32, filters, affine=False)

        if spectral_norm:
            self.conv1 = sn(nn.Conv1d(filters, filters, kernel_size=kernel_size, stride=1, padding=kernel_size//2))
            self.conv2 = sn(nn.Conv1d(filters, filters, kernel_size=kernel_size, stride=1, padding=kernel_size//2))
        else:
            self.conv1 = nn.Conv1d(filters, filters, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            self.conv2 = nn.Conv1d(filters, filters, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

        if filters <= 128:
            self.bn2 = nn.InstanceNorm1d(filters, affine=False)
        else:
            self.bn2 = nn.GroupNorm(32, filters, affine=False)

        # TODO: conv1?
        torch.nn.init.normal_(self.conv2.weight, mean=0.0, std=1e-5)

        # Upscale to an mask of image
        self.latent_fc1 = nn.Linear(latent_dim, 2*filters)
        self.latent_fc2 = nn.Linear(latent_dim, 2*filters)

        # Upscale to mask of image
        if downsample:
            if rescale:
                self.conv_downsample = nn.Conv1d(filters, 2 * filters, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            else:
                self.conv_downsample = nn.Conv1d(filters, filters, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

            self.avg_pool = nn.AvgPool1d(kernel_size, stride=2, padding=kernel_size//2)

    def forward(self, x, latent):

        # TODO: How to properly condition the different objects. Equally? or should we concatenate them.
        x_orig = x

        if latent is not None:
            latent_1 = self.latent_fc1(latent)
            latent_2 = self.latent_fc2(latent)

            gain = latent_1[:, :, :self.filters]
            bias = latent_1[:, :, self.filters:]

            gain2 = latent_2[:, :, :self.filters]
            bias2 = latent_2[:, :, self.filters:]


            x = self.conv1(x)
            x = x.reshape(latent.size(0), -1, *x.shape[-2:])
            x = gain[..., None] * x + bias[..., None]
            x = swish(x)
            x = x.flatten(0,1)
            x = self.conv2(x)
            x = x.reshape(latent.size(0), -1, *x.shape[-2:])
            x = gain2[..., None] * x + bias2[..., None]
            x = swish(x)
            x = x.flatten(0,1)

        else:
            x = self.conv1(x)
            x = swish(x)
            x = self.conv2(x)
            x = swish(x)

        #     if latent.size(0) < x.size(0):
        #
        #         x = self.conv1(x)
        #         x = x.reshape(latent.size(0), -1, *x.shape[-2:])
        #         x = gain[:, None] * x + bias[:, None]
        #         x = swish(x)
        #
        #         x = x.flatten(0,1)
        #         x = self.conv2(x)
        #         x = x.reshape(latent.size(0), -1, *x.shape[-2:])
        #         x = gain2[:, None] * x + bias2[:, None]
        #         x = swish(x)
        #         x = x.flatten(0,1)
        #     else:
        #         x = self.conv1(x)
        #         x = gain * x + bias
        #         x = swish(x)
        #
        #
        #         x = self.conv2(x)
        #         x = gain2 * x + bias2
        #         x = swish(x)

        x_out = x_orig + x

        if self.downsample:
            x_out = swish(self.conv_downsample(x_out))
            x_out = self.avg_pool(x_out)
        return x_out

class CondMLPResBlock1d(nn.Module):
    def __init__(self, filters=64, latent_dim=64, im_size=64, latent_grid=False, spectral_norm=False):
        super(CondMLPResBlock1d, self).__init__()

        self.filters = filters
        self.latent_dim = latent_dim
        self.im_size = im_size
        self.latent_grid = latent_grid

        if spectral_norm:
            self.fc1 = sn(nn.Linear(filters, filters))
            self.fc2 = sn(nn.Linear(filters, filters))
        else:
            self.fc1 = nn.Linear(filters, filters)
            self.fc2 = nn.Linear(filters, filters)

        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=1e-5)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=1e-5)

        # Upscale to an mask of image
        # TODO: Spectral norm?
        self.latent_fc1 = nn.Linear(latent_dim, 2*filters)
        self.latent_fc2 = nn.Linear(latent_dim, 2*filters)


    def forward(self, x, latent):

        # TODO: How to properly condition the different objects. Equally? or should we concatenate them.
        x_orig = x

        if latent is not None:
            latent_1 = self.latent_fc1(latent)
            latent_2 = self.latent_fc2(latent)

            gain = latent_1[:, :, :self.filters]
            bias = latent_1[:, :, self.filters:]

            gain2 = latent_2[:, :, :self.filters]
            bias2 = latent_2[:, :, self.filters:]

            if len(x.shape) > 3:
                x = self.fc1(x)
                x = gain[:, None] * x + bias[:, None]
                x = swish(x)


                x = self.fc2(x)
                x = gain2[:, None] * x + bias2[:, None]
                x = swish(x)
            else:
                x = self.fc1(x)
                x = gain * x + bias
                x = swish(x)


                x = self.fc2(x)
                x = gain2 * x + bias2
                x = swish(x)
        else:
            x = self.fc1(x)
            x = swish(x)

            x = self.fc2(x)
            x = swish(x)

        x_out = x_orig + x

        return x_out

class CNNBlock(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., spectral_norm = False):
        super(CNNBlock, self).__init__()
        # self.pool = nn.MaxPool1d(kernel_size=2, stride=None, padding=0,
        #                          dilation=1, return_indices=False,
        #                          ceil_mode=False)

        if spectral_norm:
            self.conv1 = sn(nn.Conv1d(n_in, n_hid, kernel_size=5, stride=1, padding=0))
            self.conv2 = sn(nn.Conv1d(n_hid, n_hid, kernel_size=5, stride=1, padding=0))
            self.conv_predict = sn(nn.Conv1d(n_hid, n_out, kernel_size=1))
            self.conv_attention = sn(nn.Conv1d(n_hid, 1, kernel_size=1))
        else:
            self.conv1 = nn.Conv1d(n_in, n_hid, kernel_size=5, stride=1, padding=0)
            self.conv2 = nn.Conv1d(n_hid, n_hid, kernel_size=5, stride=1, padding=0)
            self.conv_predict = nn.Conv1d(n_hid, n_out, kernel_size=1)
            self.conv_attention = nn.Conv1d(n_hid, 1, kernel_size=1)

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        # Input shape: [num_sims * num_edges, num_dims, num_timesteps]

        x = swish(self.conv1(inputs))
        # x = self.bn1(x)
        # x = F.dropout(x, self.dropout_prob, training=self.training)
        # x = self.pool(x)

        # x = swish(self.conv2(x))

        # x = self.bn2(x)
        pred = self.conv_predict(x)

        # attention = my_softmax(self.conv_attention(x), axis=2)
        # edge_prob = (pred * attention).mean(dim=2)

        return pred

class MLPBlock(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0., spectral_norm = False):
        super(MLPBlock, self).__init__()
        if spectral_norm:
            self.fc1 = sn(nn.Linear(n_in, n_hid))
            self.fc2 = sn(nn.Linear(n_hid, n_out))
        else:
            self.fc1 = nn.Linear(n_in, n_hid)
            self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = swish(self.fc1(inputs)) # TODO: Switch to swish? (or only in ebm)
        # x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.fc2(x)
        return x # Note: removed normalization self.batch_norm(x)


# class TrajGraphEBM(nn.Module):
#     def __init__(self, args, dataset):
#         super(TrajGraphEBM, self).__init__()
#         do_prob = args.dropout
#         self.dropout_prob = do_prob
#
#         filter_dim = args.filter_dim
#         self.filter_dim = filter_dim
#         # latent_dim_expand = args.latent_dim * args.components
#         latent_dim = args.latent_dim
#         spectral_norm = args.spectral_norm
#         # self.components = args.components
#
#
#         self.factor = True
#
#         state_dim = args.input_dim
#         self.obj_id_embedding = args.obj_id_embedding
#         if args.obj_id_embedding:
#             state_dim += args.obj_id_dim
#             self.obj_embedding = NodeID(args)
#
#         self.cnn = CNNBlock(state_dim * 2, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)
#
#         # self.mlp2 = MLPBlock(filter_dim, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)
#         # self.mlp3 = MLPBlock(filter_dim * 3, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)
#
#         if spectral_norm:
#             self.energy_map = sn(nn.Linear(filter_dim, 1))
#             self.mlp2 = sn(nn.Linear(filter_dim, filter_dim))
#             self.mlp3 = sn(nn.Linear(filter_dim * 3, filter_dim))
#         else:
#             self.energy_map = nn.Linear(filter_dim, 1)
#             self.mlp2 = nn.Linear(filter_dim, filter_dim)
#             self.mlp3 = nn.Linear(filter_dim * 3, filter_dim)
#
#         self.layer_encode = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, rescale=False, spectral_norm=spectral_norm)
#         self.layer1 = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, downsample=False, rescale=False, spectral_norm=spectral_norm)
#         self.layer2 = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
#         self.layer3 = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
#
#         if self.factor:
#             print("Using factor graph CNN encoder.")
#         else:
#             print("Using CNN encoder.")
#
#         self.init_weights()
#
#         # New
#         if args.forecast:
#             timesteps_enc = args.num_fixed_timesteps
#         else: timesteps_enc = args.timesteps
#         # self.latent_encoder = MLPLatentEncoder(timesteps_enc * args.input_dim, args.latent_hidden_dim, args.latent_dim,
#         #                                        do_prob=args.dropout, factor=True)
#         self.latent_encoder = CNNLatentEncoder(state_dim, args.latent_hidden_dim, args.latent_dim,
#                                                do_prob=args.dropout, factor=True)
#
#         self.rel_rec = None
#         self.rel_send = None
#
#     def embed_latent(self, traj, rel_rec, rel_send):
#         if self.rel_rec is None and self.rel_send is None:
#             self.rel_rec, self.rel_send = rel_rec[0:1], rel_send[0:1]
#
#         if self.obj_id_embedding:
#             traj = self.obj_embedding(traj)
#
#         return self.latent_encoder(traj, rel_rec, rel_send)
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal(m.weight.data)
#                 m.bias.data.fill_(0.1)
#
#     def node2edge_temporal(self, inputs, rel_rec, rel_send):
#         # NOTE: Assumes that we have the same graph across all samples.
#
#         x = inputs.reshape(inputs.size(0), inputs.size(1), -1)
#
#         receivers = torch.matmul(rel_rec, x)
#         receivers = receivers.reshape(inputs.size(0) * receivers.size(1),
#                                    inputs.size(2), inputs.size(3))
#         receivers = receivers.transpose(2, 1)
#
#         senders = torch.matmul(rel_send, x)
#         senders = senders.reshape(inputs.size(0) * senders.size(1),
#                                inputs.size(2),
#                                inputs.size(3))
#         senders = senders.transpose(2, 1)
#
#         # receivers and senders have shape:
#         # [num_sims * num_edges, num_dims, num_timesteps]
#         edges = torch.cat([senders, receivers], dim=1)
#         return edges
#
#     def edge2node(self, x, rel_rec, rel_send):
#         # NOTE: Assumes that we have the same graph across all samples.
#         incoming = torch.matmul(rel_rec.transpose(2, 1), x)
#         return incoming / incoming.size(1)
#
#     def node2edge(self, x, rel_rec, rel_send):
#         # NOTE: Assumes that we have the same graph across all samples.
#         receivers = torch.matmul(rel_rec, x)
#         senders = torch.matmul(rel_send, x)
#         edges = torch.cat([senders, receivers], dim=2)
#         return edges
#
#     def forward(self, inputs, latent):
#
#         rel_rec = self.rel_rec #.repeat_interleave(inputs.size(0), dim=0)
#         rel_send = self.rel_send #.repeat_interleave(inputs.size(0), dim=0)
#         if self.obj_id_embedding:
#             inputs = self.obj_embedding(inputs)
#
#         # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
#         edges = self.node2edge_temporal(inputs, rel_rec, rel_send) #[N, 4, T] --> [R, 8, T] # Marshalling
#         x = swish(self.cnn(edges))  # [R, 8, T] --> [R, F, T'] # CNN layers
#         x = self.layer_encode(x, latent) # [R, F, T'] --> [R, F, T''] # Conditioning layer
#         x = self.layer1(x, latent) # [R, F, T''] --> [R, F, T'''] # Conditioning layer
#
#         x = x.mean(-1) # [R, F, T'''] --> [R, F] # Temporal Avg pool
#         x = x.reshape(latent.size(0), -1, x.size(-1))
#         x_skip = x
#
#         if self.factor:
#             x = self.edge2node(x, rel_rec, rel_send) # [R, F] --> [N, F] # marshalling
#             x = swish(self.mlp2(x)) # [N, F] --> [N, F]
#             x = self.layer2(x, latent) # [N, F] --> [N, F] # Conditioning layer
#
#             x = self.node2edge(x, rel_rec, rel_send) # [N, F] --> [R, 2F] # marshalling
#             x = torch.cat((x, x_skip), dim=2)  # [R, 2F] --> [R, 3F] # Skip connection
#             x = swish(self.mlp3(x)) # [R, 3F] --> [R, F]
#             x = self.layer3(x, latent) # [R, F] --> [R, F] # Conditioning layer
#
#         x = x.mean(1) # Avg across nodes
#         x = x.view(x.size(0), -1)
#         energy = self.energy_map(x) # [F] --> [1] # Project features to scalar
#
#         return energy
#
# class EdgeGraphEBM(nn.Module):
#     def __init__(self, args, dataset):
#         super(EdgeGraphEBM, self).__init__()
#         do_prob = args.dropout
#         self.dropout_prob = do_prob
#
#         filter_dim = args.filter_dim
#         self.filter_dim = filter_dim
#         # latent_dim_expand = args.latent_dim * args.components
#         latent_dim = args.latent_dim
#         spectral_norm = args.spectral_norm
#         # self.components = args.components
#
#
#         self.factor = True
#
#         state_dim = args.input_dim
#
#         self.obj_id_embedding = args.obj_id_embedding
#         if args.obj_id_embedding:
#             state_dim += args.obj_id_dim
#             self.obj_embedding = NodeID(args)
#
#         self.cnn = CNNBlock(state_dim * 2, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)
#
#         # self.mlp2 = MLPBlock(filter_dim, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)
#         # self.mlp3 = MLPBlock(filter_dim * 3, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)
#
#         if spectral_norm:
#             self.energy_map = sn(nn.Linear(filter_dim, 1))
#             self.mlp2 = sn(nn.Linear(filter_dim, filter_dim))
#             self.mlp3 = sn(nn.Linear(filter_dim * 3, filter_dim))
#         else:
#             self.energy_map = nn.Linear(filter_dim, 1)
#             self.mlp2 = nn.Linear(filter_dim, filter_dim)
#             self.mlp3 = nn.Linear(filter_dim * 3, filter_dim)
#
#         self.layer_encode = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, rescale=False, spectral_norm=spectral_norm)
#         self.layer1 = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, downsample=False, rescale=False, spectral_norm=spectral_norm)
#         self.layer2 = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
#         self.layer3 = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
#
#         if self.factor:
#             print("Using factor graph CNN encoder.")
#         else:
#             print("Using CNN encoder.")
#
#         self.init_weights()
#
#         # New
#         if args.forecast:
#             timesteps_enc = args.num_fixed_timesteps
#         else: timesteps_enc = args.timesteps
#         # self.latent_encoder = MLPLatentEncoder(timesteps_enc * args.input_dim, args.latent_hidden_dim, args.latent_dim,
#         #                                        do_prob=args.dropout, factor=True)
#         self.latent_encoder = CNNLatentEncoder(state_dim, args.latent_hidden_dim, args.latent_dim,
#                                                do_prob=args.dropout, factor=True)
#
#         self.rel_rec = None
#         self.rel_send = None
#         self.edge_ids = torch.zeros(2, 6)
#         self.edge_ids[:, 1] = 1
#
#     def embed_latent(self, traj, rel_rec, rel_send):
#         if self.rel_rec is None and self.rel_send is None:
#             self.rel_rec, self.rel_send = rel_rec[0:1], rel_send[0:1]
#
#         if self.obj_id_embedding:
#             traj = self.obj_embedding(traj)
#
#         return self.latent_encoder(traj, rel_rec, rel_send)
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal(m.weight.data)
#                 m.bias.data.fill_(0.1)
#
#     def node2edge_temporal(self, inputs, rel_rec, rel_send):
#         # NOTE: Assumes that we have the same graph across all samples.
#
#         x = inputs.reshape(inputs.size(0), inputs.size(1), -1)
#
#         receivers = torch.matmul(rel_rec, x)
#         receivers = receivers.reshape(inputs.size(0) * receivers.size(1),
#                                       inputs.size(2), inputs.size(3))
#         receivers = receivers.transpose(2, 1)
#
#         senders = torch.matmul(rel_send, x)
#         senders = senders.reshape(inputs.size(0) * senders.size(1),
#                                   inputs.size(2),
#                                   inputs.size(3))
#         senders = senders.transpose(2, 1)
#
#         # receivers and senders have shape:
#         # [num_sims * num_edges, num_dims, num_timesteps]
#         edges = torch.cat([senders, receivers], dim=1)
#         return edges
#
#     def edge2node(self, x, rel_rec, rel_send):
#         # NOTE: Assumes that we have the same graph across all samples.
#         incoming = torch.matmul(rel_rec.transpose(2, 1), x)
#         return incoming / incoming.size(1)
#
#     def node2edge(self, x, rel_rec, rel_send):
#         # NOTE: Assumes that we have the same graph across all samples.
#         receivers = torch.matmul(rel_rec, x)
#         senders = torch.matmul(rel_send, x)
#         edges = torch.cat([senders, receivers], dim=2)
#         return edges
#
#     def forward(self, inputs, latent):
#
#         rel_rec = self.rel_rec #.repeat_interleave(inputs.size(0), dim=0)
#         rel_send = self.rel_send #.repeat_interleave(inputs.size(0), dim=0)
#         latent, edge_ids = latent
#
#         if self.obj_id_embedding:
#             inputs = self.obj_embedding(inputs)
#
#         BS, NO, _, ND = inputs.shape
#         NR = NO * (NO-1)
#
#         # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
#         edges = self.node2edge_temporal(inputs, rel_rec, rel_send) #[N, 4, T] --> [R, 8, T] # Marshalling
#         x = swish(self.cnn(edges))  # [R, 8, T] --> [R, F, T'] # CNN layers
#
#         # Conditional pass for edge of interest
#         x_sel = (x.reshape(BS, -1, *x.shape[1:]) * edge_ids).sum(1)
#         x_sel = self.layer_encode(x_sel, latent) # [F, T'] --> [F, T''] # Conditioning layer
#         x_sel = self.layer1(x_sel, latent) # [F, T''] --> [F, T'''] # Conditioning layer
#
#         # Unconditional pass for the rest of edges
#         x = self.layer_encode(x, latent=None) # [R, F, T'] --> [R, F, T''] # Conditioning layer
#         x = self.layer1(x, latent=None) # [R, F, T''] --> [R, F, T'''] # Conditioning layer
#
#         # Join all edges with the edge of interest
#         x = (x.reshape(BS, -1, *x.shape[1:]) * (1 - edge_ids)) + (x_sel[:, None] * edge_ids)
#         x = x.mean(-1).flatten(0,1) # [R, F, T'''] --> [R, F] # Temporal Avg pool
#         x = x.reshape(BS, NR, x.size(-1))
#
#         x_skip = x_sel.mean(-1)[:, None] # [F, T'''] --> [1, F] # Temporal Avg pool
#
#         if self.factor:
#             x = self.edge2node(x, rel_rec, rel_send) # [R, F] --> [N, F] # marshalling
#             x = swish(self.mlp2(x)) # [N, F] --> [N, F]
#
#             x = self.node2edge(x, rel_rec, rel_send) # [N, F] --> [R, 2F] # marshalling
#             x_sel = (x * edge_ids[..., 0]).sum(1, keepdims=True)
#             x = torch.cat((x_sel, x_skip), dim=2)  # [R, 2F] --> [R, 3F] # Skip connection
#             x = swish(self.mlp3(x)) # [R, 3F] --> [R, F]
#             x = self.layer2(x, latent) # [R, F] --> [R, F] # Conditioning layer
#             x = self.layer3(x, latent) # [R, F] --> [R, F] # Conditioning layer
#
#         x = x.mean(1) # Avg across nodes
#         x = x.view(x.size(0), -1)
#         energy = self.energy_map(x) # [F] --> [1] # Project features to scalar
#
#         return energy

# if __name__ == "__main__":
#     args = EasyDict()
#     args.filter_dim = 64
#     args.latent_dim = 64
#     args.im_size = 256
#
#     model = TrajEBM(args).cuda()
#     x = torch.zeros(1, 3, 256, 256).cuda()
#     latent = torch.zeros(1, 64).cuda()
#     model(x, latent)
