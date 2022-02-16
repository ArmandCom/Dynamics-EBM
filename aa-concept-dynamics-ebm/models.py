from torch.nn import ModuleList
import math
import torch.nn.functional as F
import torch.nn as nn
import torch
from easydict import EasyDict
from downsample import Downsample
import warnings
from torch.nn.utils import spectral_norm as sn
from encoder_models import MLPLatentEncoder, CNNLatentEncoder, my_softmax

warnings.filterwarnings("ignore")

def swish(x):
    return x * torch.sigmoid(x)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class TrajEBM(nn.Module):
    def __init__(self, args, dataset):

        # TODO: Node encoding appended to z + encoded in EBM for matching. Explicit matching?
        super(TrajEBM, self).__init__()

        filter_dim = args.filter_dim
        self.filter_dim = filter_dim
        latent_dim_expand = args.latent_dim * args.components
        latent_dim = args.latent_dim
        spectral_norm = args.spectral_norm

        self.components = args.components

        # self.conv1 = nn.Conv1d(4, filter_dim, kernel_size=5, stride=1, padding=2, bias=True)
        if spectral_norm:
            self.conv1 = sn(nn.Conv1d(4 * args.n_objects, filter_dim, kernel_size=5, stride=1, padding=2, bias=True))
            self.conv2 = sn(nn.Conv1d(filter_dim, filter_dim, kernel_size=5, stride=1, padding=2, bias=True))
            self.energy_map = sn(nn.Linear(filter_dim, 1))
        else:
            self.conv1 = nn.Conv1d(4 * args.n_objects, filter_dim, kernel_size=5, stride=1, padding=2, bias=True)
            self.conv2 = nn.Conv1d(filter_dim, filter_dim, kernel_size=5, stride=1, padding=2, bias=True)
            self.energy_map = nn.Linear(filter_dim, 1)

        self.layer_encode = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, rescale=False, spectral_norm=spectral_norm)
        self.layer1 = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, downsample=False, rescale=False, spectral_norm=spectral_norm)
        self.layer2 = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, rescale=False, spectral_norm=spectral_norm)
        self.layer3 = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, downsample = False, rescale=False, spectral_norm=spectral_norm)

        self.avg_pool = nn.AvgPool1d(3, stride=2, padding=1)

        # New
        if args.forecast:
            timesteps_enc = args.num_fixed_timesteps
        else: timesteps_enc = args.timesteps
        # self.latent_encoder = MLPLatentEncoder(timesteps_enc * args.input_dim, args.latent_hidden_dim, args.latent_dim,
        #                                        do_prob=args.dropout, factor=True)
        self.latent_encoder = CNNLatentEncoder(args.input_dim, args.latent_hidden_dim, args.latent_dim,
                                               do_prob=args.dropout, factor=True)

    def embed_latent(self, traj, rel_rec, rel_send):
        '''
        im: [24, 3, 64, 64]
        traj: [B, n_obj, T, dim]
        '''
        return self.latent_encoder(traj, rel_rec, rel_send)

    def forward(self, x, latent):

        # TODO: Smoothing trajectory

        # Note: TESTING FEATURE
        # nodes = x.view(x.size(0), x.size(1), -1)
        # node_embs = self.latent_encoder.mlp1(nodes).unsqueeze(2)\
        #     .repeat_interleave(x.shape[2], dim=2)[..., :4]  # 2-layer ELU net per node (we only keep the last N features for matching)
        # x = torch.cat([x, node_embs], dim=-1)

        # For the first attempt we treat objects jointly. We could also consider to embed the distances in the edges. Or treat it as a graph.
        x = x.permute(0,1,3,2).flatten(1,2)
        # x = x.contiguous()
        # Note: x = inputs.view(inputs.size(0), inputs.size(1), -1)

        # Future work
        # self.latent_encoder.mlp1(x.) # Canviar in filters de

        inter = self.conv1(x)
        inter = swish(inter)
        inter = self.conv2(inter)
        x = swish(inter)

        # x = self.avg_pool(inter)

        x = self.layer_encode(x, latent)

        # if self.args.self_attn:
        #     x, _ = self.self_attn(x)

        x = self.layer1(x, latent) # CondResBlock
        x = self.layer2(x, latent)
        x = self.layer3(x, latent)

        x = x.mean(dim=2)
        x = x.view(x.size(0), -1)

        energy = self.energy_map(x)

        return energy

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

        state_dim = 4
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
        self.latent_encoder = CNNLatentEncoder(args.input_dim, args.latent_hidden_dim, args.latent_dim,
                                               do_prob=args.dropout, factor=True)

        self.rel_rec = None
        self.rel_send = None

    def embed_latent(self, traj, rel_rec, rel_send):
        if self.rel_rec is None and self.rel_send is None:
            self.rel_rec, self.rel_send = rel_rec[0:1], rel_send[0:1]
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

        state_dim = 4
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
        self.latent_encoder = CNNLatentEncoder(args.input_dim, args.latent_hidden_dim, args.latent_dim,
                                               do_prob=args.dropout, factor=True)

        self.rel_rec = None
        self.rel_send = None

    def embed_latent(self, traj, rel_rec, rel_send):
        if self.rel_rec is None and self.rel_send is None:
            self.rel_rec, self.rel_send = rel_rec[0:1], rel_send[0:1]
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

            x = self.node2edge(x, rel_rec, rel_send) # [N, F] --> [R, 2F] # marshalling
            x = torch.cat((x, x_skip), dim=2)  # [R, 2F] --> [R, 3F] # Skip connection
            x = swish(self.mlp3(x)) # [R, 3F] --> [R, F]
            x = self.layer2(x, latent) # [R, F] --> [R, F] # Conditioning layer
            x = self.layer3(x, latent) # [R, F] --> [R, F] # Conditioning layer

        x = x.mean(1) # Avg across nodes
        x = x.view(x.size(0), -1)
        energy = self.energy_map(x) # [F] --> [1] # Project features to scalar

        return energy

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
