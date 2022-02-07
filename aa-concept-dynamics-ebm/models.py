from torch.nn import ModuleList
import torch.nn.functional as F
import torch.nn as nn
import torch
from easydict import EasyDict
from downsample import Downsample
import warnings
from torch.nn.utils import spectral_norm as sn

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


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

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

        n_instance = len(dataset)

        # self.conv1 = nn.Conv1d(4, filter_dim, kernel_size=5, stride=1, padding=2, bias=True)
        if spectral_norm:
            self.conv1 = sn(nn.Conv1d(4 * args.n_objects, filter_dim, kernel_size=5, stride=1, padding=2, bias=True))
            self.conv2 = sn(nn.Conv1d(filter_dim, filter_dim, kernel_size=5, stride=1, padding=2, bias=True))
            self.energy_map = sn(nn.Linear(filter_dim * 2, 1))
        else:
            self.conv1 = nn.Conv1d(4 * args.n_objects, filter_dim, kernel_size=5, stride=1, padding=2, bias=True)
            self.conv2 = nn.Conv1d(filter_dim, filter_dim, kernel_size=5, stride=1, padding=2, bias=True)
            self.energy_map = nn.Linear(filter_dim * 2, 1)

        self.layer_encode = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, rescale=False, spectral_norm=spectral_norm)
        self.layer1 = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, rescale=False, spectral_norm=spectral_norm)
        self.layer2 = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)

        self.avg_pool = nn.AvgPool1d(3, stride=2, padding=1)

        # New
        self.latent_encoder = MLPLatentEncoder(args.timesteps * args.input_dim, args.latent_hidden_dim, args.latent_dim,
                                               do_prob=args.dropout, factor=True)

    def embed_latent(self, traj, rel_rec, rel_send):
        '''
        im: [24, 3, 64, 64]
        traj: [B, n_obj, T, dim]
        '''

        return self.latent_encoder(traj, rel_rec, rel_send)

        # TODO: Object centric trajectory decomposition?
    def forward(self, x, latent):

        # TODO: Smoothing trajectory

        # Note: TESTING FEATURE
        # nodes = x.view(x.size(0), x.size(1), -1)
        # node_embs = self.latent_encoder.mlp1(nodes).unsqueeze(2)\
        #     .repeat_interleave(x.shape[2], dim=2)[..., :4]  # 2-layer ELU net per node (we only keep the last N features for matching)
        # # TODO try swish. We are repeating the computation. We already have it in the latent.
        # x = torch.cat([x, node_embs], dim=-1)

        # For the first attempt we treat objects jointly. We could also consider to embed the distances in the edges. Or treat it as a graph.
        x = x.permute(0,1,3,2).flatten(1,2)
        # x = x.contiguous()
        # Note: x = inputs.view(inputs.size(0), inputs.size(1), -1)

        # Future work
        # self.latent_encoder.mlp1(x.) # Canviar in filters de

        inter = self.conv1(x)
        inter = swish(inter)

        x = self.avg_pool(inter)

        # TODO: Consider keeping positional embedding as a temporal encoding.
        x = self.layer_encode(x, latent)

        # if self.args.self_attn: # TODO: attention?
        #     x, _ = self.self_attn(x)

        x = self.layer1(x, latent) # CondResBlock
        x = self.layer2(x, latent)

        x = x.mean(dim=2)
        x = x.view(x.size(0), -1)

        # TODO: Add rnn?
        energy = self.energy_map(x)

        # TODO: condition on X(t=0).
        return energy

class MLPLatentEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        super(MLPLatentEncoder, self).__init__()

        self.factor = factor

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.permute(0,2,1), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        # Note: En el original tambe es miren tota la trajectoria de una?
        obj = self.mlp1(x)  # 2-layer ELU net per node

        x = self.node2edge(obj, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)

        return self.fc_out(x)

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

            self.avg_pool = nn.AvgPool1d(3, stride=2, padding=1)

    def forward(self, x, latent):

        # TODO: object-aware conditioning. Not only timewise. Could objects be treated as features?
        #   Try "A simple neural network module for relational reasoning" https://github.com/moduIo/Relation-Networks?utm_source=catalyzex.com.
        #   Intuition treat relations pairwise (all permutations), sum all scores.
        #   That might not work well, as relations are matched to all pairs and aggregated. Should rel(O1, O2) affect O3, O4 pair?

        x_orig = x

        latent_1 = self.latent_fc1(latent)
        latent_2 = self.latent_fc2(latent)

        gain = latent_1[:, :self.filters, None]
        bias = latent_1[:, self.filters:, None]

        gain2 = latent_2[:, :self.filters, None]
        bias2 = latent_2[:, self.filters:, None]

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

class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
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
        x = F.elu(self.fc1(inputs)) # TODO: Switch to swish? (or only in ebm)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return x # Note: removed normalization self.batch_norm(x)

# Other classes
class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False):
        super(MLPDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        pre_msg.size(2), self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.

        inputs = inputs.transpose(1, 2).contiguous()

        sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1),
                 rel_type.size(2)]
        rel_type = rel_type.unsqueeze(1).expand(sizes)

        time_steps = inputs.size(1)
        assert (pred_steps <= time_steps)
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]
        curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        for step in range(0, pred_steps):
            last_pred = self.single_step_forward(last_pred, rel_rec, rel_send,
                                                 curr_rel_type)
            preds.append(last_pred)

        sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                 preds[0].size(2), preds[0].size(3)]

        output = Variable(torch.zeros(sizes))
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()

class CondResBlock(nn.Module):
    def __init__(self, downsample=True, rescale=True, filters=64, latent_dim=64, im_size=64, latent_grid=False):
        super(CondResBlock, self).__init__()

        self.filters = filters
        self.latent_dim = latent_dim
        self.im_size = im_size
        self.downsample = downsample
        self.latent_grid = latent_grid

        if filters <= 128: # Q: Why this condition?
            self.bn1 = nn.InstanceNorm2d(filters, affine=False)
        else:
            self.bn1 = nn.GroupNorm(32, filters, affine=False)

        self.conv1 = nn.Conv2d(filters, filters, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=5, stride=1, padding=2)

        if filters <= 128:
            self.bn2 = nn.InstanceNorm2d(filters, affine=False)
        else:
            self.bn2 = nn.GroupNorm(32, filters, affine=False)


        torch.nn.init.normal_(self.conv2.weight, mean=0.0, std=1e-5)

        # Upscale to an mask of image
        self.latent_fc1 = nn.Linear(latent_dim, 2*filters)
        self.latent_fc2 = nn.Linear(latent_dim, 2*filters)

        # Upscale to mask of image
        if downsample:
            if rescale:
                self.conv_downsample = nn.Conv2d(filters, 2 * filters, kernel_size=5, stride=1, padding=2)
            else:
                self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=5, stride=1, padding=2)

            self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x, latent):
        x_orig = x

        latent_1 = self.latent_fc1(latent)
        latent_2 = self.latent_fc2(latent)

        gain = latent_1[:, :self.filters, None, None]
        bias = latent_1[:, self.filters:, None, None]

        gain2 = latent_2[:, :self.filters, None, None]
        bias2 = latent_2[:, self.filters:, None, None]

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

class CondResBlockNoLatent(nn.Module):
    def __init__(self, downsample=True, rescale=True, filters=64, upsample=False):
        super(CondResBlockNoLatent, self).__init__()

        self.filters = filters
        self.downsample = downsample

        if filters <= 128:
            self.bn1 = nn.GroupNorm(int(32  * filters / 128), filters, affine=True)
        else:
            self.bn1 = nn.GroupNorm(32, filters, affine=False)

        self.conv1 = nn.Conv2d(filters, filters, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=5, stride=1, padding=2)

        if filters <= 128:
            self.bn2 = nn.GroupNorm(int(32 * filters / 128), filters, affine=True)
        else:
            self.bn2 = nn.GroupNorm(32, filters, affine=True)

        self.upsample = upsample
        self.upsample_module = nn.Upsample(scale_factor=2)
        # Upscale to mask of image
        if downsample:
            if rescale:
                self.conv_downsample = nn.Conv2d(filters, 2 * filters, kernel_size=5, stride=1, padding=2)
            else:
                self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=5, stride=1, padding=2)

            self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

        if upsample:
            self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x_orig = x


        x = self.conv1(x)
        x = swish(x)

        x = self.conv2(x)
        x = swish(x)

        x_out = x_orig + x

        if self.upsample:
            x_out = self.upsample_module(x_out)
            x_out = swish(self.conv_downsample(x_out))

        if self.downsample:
            x_out = swish(self.conv_downsample(x_out))
            x_out = self.avg_pool(x_out)

        return x_out

class BroadcastConvDecoder(nn.Module):
    def __init__(self, im_size, latent_dim):
        super().__init__()
        self.im_size = im_size + 8
        self.latent_dim = latent_dim
        self.init_grid()

        self.g = nn.Sequential(
                    nn.Conv2d(self.latent_dim+2, 32, 3, 1, 0),
                    nn.ReLU(True),
                    nn.Conv2d(32, 32, 3, 1, 0),
                    nn.ReLU(True),
                    nn.Conv2d(32, 32, 3, 1, 0),
                    nn.ReLU(True),
                    nn.Conv2d(32, 32, 3, 1, 0),
                    nn.ReLU(True),
                    nn.Conv2d(32, self.latent_dim, 1, 1, 0)
                    )

    def init_grid(self):
        x = torch.linspace(0, 1, self.im_size)
        y = torch.linspace(0, 1, self.im_size)
        self.x_grid, self.y_grid = torch.meshgrid(x, y)


    def broadcast(self, z):
        b = z.size(0)
        x_grid = self.x_grid.expand(b, 1, -1, -1).to(z.device)
        y_grid = self.y_grid.expand(b, 1, -1, -1).to(z.device)
        z = z.view((b, -1, 1, 1)).expand(-1, -1, self.im_size, self.im_size)
        z = torch.cat((z, x_grid, y_grid), dim=1)
        return z

    def forward(self, z):
        z = self.broadcast(z)
        x = self.g(z)
        return x

class DisentangleModel(nn.Module):
    def __init__(self):
        super(DisentangleModel, self).__init__()

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10, nc=3):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, return_z=False):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        if return_z:
            return x_recon, mu, logvar, z
        else:
            return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

    def reconstruction_loss(self, x, x_recon, distribution):
        batch_size = x.size(0)
        assert batch_size != 0

        if distribution == 'bernoulli':
            recon_loss = F.binary_cross_entropy_with_logits(
                x_recon, x, size_average=False).div(batch_size)
        elif distribution == 'gaussian':
            x_recon = F.sigmoid(x_recon)
            recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
        else:
            recon_loss = None

        return recon_loss

    def compute_cross_ent_normal(self, mu, logvar):
        return 0.5 * (mu**2 + torch.exp(logvar)) + np.log(np.sqrt(2 * np.pi))

    def compute_ent_normal(self, logvar):
        return 0.5 * (logvar + np.log(2 * np.pi * np.e))


if __name__ == "__main__":
    args = EasyDict()
    args.filter_dim = 64
    args.latent_dim = 64
    args.im_size = 256

    model = LatentEBM(args).cuda()
    x = torch.zeros(1, 3, 256, 256).cuda()
    latent = torch.zeros(1, 64).cuda()
    model(x, latent)
