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
from encoder_models import MLPLatentEncoder, CNNLatentEncoder, my_softmax

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

class TrajGraphEBM(nn.Module):
    def __init__(self, args, dataset):
        super(TrajGraphEBM, self).__init__()
        do_prob = args.dropout
        self.dropout_prob = do_prob

        filter_dim = args.filter_dim
        self.filter_dim = filter_dim
        # latent_dim_expand = args.latent_dim * args.components
        latent_dim = args.latent_dim
        spectral_norm = args.spectral_norm
        # self.components = args.components


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

        self.layer_encode = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, rescale=False, spectral_norm=spectral_norm)
        self.layer1 = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, downsample=False, rescale=False, spectral_norm=spectral_norm)
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

        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send) #[N, 4, T] --> [R, 8, T] # Marshalling
        x = swish(self.cnn(edges))  # [R, 8, T] --> [R, F, T'] # CNN layers
        x = self.layer_encode(x, latent) # [R, F, T'] --> [R, F, T''] # Conditioning layer
        x = self.layer1(x, latent) # [R, F, T''] --> [R, F, T'''] # Conditioning layer

        x = x.mean(-1) # [R, F, T'''] --> [R, F] # Temporal Avg pool
        x = x.reshape(latent.size(0), -1, x.size(-1))
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send) # [R, F] --> [N, F] # marshalling
            x = swish(self.mlp2(x)) # [N, F] --> [N, F]
            x = self.layer2(x, latent) # [N, F] --> [N, F] # Conditioning layer

            x = self.node2edge(x, rel_rec, rel_send) # [N, F] --> [R, 2F] # marshalling
            x = torch.cat((x, x_skip), dim=2)  # [R, 2F] --> [R, 3F] # Skip connection
            x = swish(self.mlp3(x)) # [R, 3F] --> [R, F]
            x = self.layer3(x, latent) # [R, F] --> [R, F] # Conditioning layer

        x = x.mean(1) # Avg across nodes
        x = x.view(x.size(0), -1)
        energy = self.energy_map(x) # [F] --> [1] # Project features to scalar

        return energy

class EdgeGraphEBM(nn.Module):
    def __init__(self, args, dataset):
        super(EdgeGraphEBM, self).__init__()
        do_prob = args.dropout
        self.dropout_prob = do_prob

        filter_dim = args.filter_dim
        self.filter_dim = filter_dim
        # latent_dim_expand = args.latent_dim * args.components
        latent_dim = args.latent_dim
        spectral_norm = args.spectral_norm
        # self.components = args.components


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

        self.layer_encode = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, rescale=False, spectral_norm=spectral_norm)
        self.layer1 = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, downsample=False, rescale=False, spectral_norm=spectral_norm)
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
        self.edge_ids = torch.zeros(2, 6)
        self.edge_ids[:, 1] = 1

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
        latent, edge_ids = latent

        if self.obj_id_embedding:
            inputs = self.obj_embedding(inputs)

        BS, NO, _, ND = inputs.shape
        NR = NO * (NO-1)

        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send) #[N, 4, T] --> [R, 8, T] # Marshalling
        x = swish(self.cnn(edges))  # [R, 8, T] --> [R, F, T'] # CNN layers

        # Conditional pass for edge of interest
        x_sel = (x.reshape(BS, -1, *x.shape[1:]) * edge_ids).sum(1)
        x_sel = self.layer_encode(x_sel, latent) # [F, T'] --> [F, T''] # Conditioning layer
        x_sel = self.layer1(x_sel, latent) # [F, T''] --> [F, T'''] # Conditioning layer

        # Unconditional pass for the rest of edges
        x = self.layer_encode(x, latent=None) # [R, F, T'] --> [R, F, T''] # Conditioning layer
        x = self.layer1(x, latent=None) # [R, F, T''] --> [R, F, T'''] # Conditioning layer

        # Join all edges with the edge of interest
        x = (x.reshape(BS, -1, *x.shape[1:]) * (1 - edge_ids)) + (x_sel[:, None] * edge_ids)
        x = x.mean(-1).flatten(0,1) # [R, F, T'''] --> [R, F] # Temporal Avg pool
        x = x.reshape(BS, NR, x.size(-1))

        x_skip = x_sel.mean(-1)[:, None] # [F, T'''] --> [1, F] # Temporal Avg pool

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send) # [R, F] --> [N, F] # marshalling
            x = swish(self.mlp2(x)) # [N, F] --> [N, F]

            x = self.node2edge(x, rel_rec, rel_send) # [N, F] --> [R, 2F] # marshalling
            x_sel = (x * edge_ids[..., 0]).sum(1, keepdims=True)
            x = torch.cat((x_sel, x_skip), dim=2)  # [R, 2F] --> [R, 3F] # Skip connection
            x = swish(self.mlp3(x)) # [R, 3F] --> [R, F]
            x = self.layer2(x, latent) # [R, F] --> [R, F] # Conditioning layer
            x = self.layer3(x, latent) # [R, F] --> [R, F] # Conditioning layer

        x = x.mean(1) # Avg across nodes
        x = x.view(x.size(0), -1)
        energy = self.energy_map(x) # [F] --> [1] # Project features to scalar

        return energy

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
    def __init__(self, downsample=True, rescale=True, filters=64, latent_dim=64, im_size=64, latent_grid=False, spectral_norm=False):
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
            self.conv1 = sn(nn.Conv1d(filters, filters, kernel_size=5, stride=1, padding=2))
            self.conv2 = sn(nn.Conv1d(filters, filters, kernel_size=5, stride=1, padding=2))
        else:
            self.conv1 = nn.Conv1d(filters, filters, kernel_size=5, stride=1, padding=2)
            self.conv2 = nn.Conv1d(filters, filters, kernel_size=5, stride=1, padding=2)

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
                self.conv_downsample = nn.Conv1d(filters, 2 * filters, kernel_size=5, stride=1, padding=2)
            else:
                self.conv_downsample = nn.Conv1d(filters, filters, kernel_size=5, stride=1, padding=2)

            self.avg_pool = nn.AvgPool1d(5, stride=2, padding=1)

    def forward(self, x, latent):

        # TODO: How to properly condition the different objects. Equally? or should we concatenate them.
        x_orig = x

        if latent is not None:
            latent_1 = self.latent_fc1(latent)
            latent_2 = self.latent_fc2(latent)

            gain = latent_1[:, :self.filters, None]
            bias = latent_1[:, self.filters:, None]

            gain2 = latent_2[:, :self.filters, None]
            bias2 = latent_2[:, self.filters:, None]

            if latent.size(0) < x.size(0):

                x = self.conv1(x)
                x = x.reshape(latent.size(0), -1, *x.shape[-2:])
                x = gain[:, None] * x + bias[:, None]
                x = swish(x)

                x = x.flatten(0,1)
                x = self.conv2(x)
                x = x.reshape(latent.size(0), -1, *x.shape[-2:])
                x = gain2[:, None] * x + bias2[:, None]
                x = swish(x)
                x = x.flatten(0,1)
            else:
                x = self.conv1(x)
                x = gain * x + bias
                x = swish(x)


                x = self.conv2(x)
                x = gain2 * x + bias2
                x = swish(x)

        else:
            x = self.conv1(x)
            x = swish(x)
            x = self.conv2(x)
            x = swish(x)

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

        latent_1 = self.latent_fc1(latent)
        latent_2 = self.latent_fc2(latent)

        gain = latent_1[:, None, :self.filters]
        bias = latent_1[:, None, self.filters:]

        gain2 = latent_2[:, None, :self.filters]
        bias2 = latent_2[:, None, self.filters:]


        x = self.fc1(x)
        x = gain * x + bias
        x = swish(x)


        x = self.fc2(x)
        x = gain2 * x + bias2
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
