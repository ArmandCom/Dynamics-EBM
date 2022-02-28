from torch.nn import ModuleList
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from easydict import EasyDict
from downsample import Downsample
import warnings
import math
from torch.nn.utils import spectral_norm as sn

warnings.filterwarnings("ignore")

def reparametrize(mu, logvar):
	std = logvar.div(2).exp()
	eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
	return mu + std*eps

def my_softmax(input, axis=1):
	trans_input = input.transpose(axis, 0).contiguous()
	soft_max_1d = F.softmax(trans_input)
	return soft_max_1d.transpose(axis, 0)

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

class CNN(nn.Module):
	def __init__(self, n_in, n_hid, n_out, do_prob=0., activation = 'relu'):
		super(CNN, self).__init__()
		self.pool = nn.MaxPool1d(kernel_size=2, stride=None, padding=0,
								 dilation=1, return_indices=False,
								 ceil_mode=False)

		self.conv1 = nn.Conv1d(n_in, n_hid, kernel_size=5, stride=1, padding=0)
		self.bn1 = nn.BatchNorm1d(n_hid)
		self.conv2 = nn.Conv1d(n_hid, n_hid, kernel_size=5, stride=1, padding=0)
		self.bn2 = nn.BatchNorm1d(n_hid)
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

		x = F.relu(self.conv1(inputs))
		# x = self.bn1(x)
		x = F.dropout(x, self.dropout_prob, training=self.training)
		# x = self.pool(x)
		x = F.relu(self.conv2(x))
		# x = self.bn2(x)
		pred = self.conv_predict(x)

		# attention = my_softmax(self.conv_attention(x), axis=2)
		# edge_prob = (pred * attention).mean(dim=2)

		edge_prob = pred
		# edge_prob = pred
		return edge_prob

class MLPLatentEncoder(nn.Module):
	def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
		super(MLPLatentEncoder, self).__init__()

		self.factor = factor

		self.mlp1 = MLP(n_in, n_hid // 2, n_hid, do_prob)
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

class CNNLatentEncoder(nn.Module):
	def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
		super(CNNLatentEncoder, self).__init__()
		self.dropout_prob = do_prob

		self.factor = factor

		self.cnn = CNN(n_in * 2, n_hid // 2, n_hid, do_prob)
		self.mlp1 = MLP(n_hid, n_hid, n_hid, do_prob)
		self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob)
		self.mlp3 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
		self.fc_out = nn.Linear(n_hid, n_out)

		if self.factor:
			print("Using factor graph CNN encoder.")
		else:
			print("Using CNN encoder.")

		self.init_weights()

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

	def forward(self, inputs, rel_rec, rel_send):

		# Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
		edges = self.node2edge_temporal(inputs, rel_rec, rel_send)
		x = self.cnn(edges)
		x = x.reshape(inputs.size(0), (inputs.size(1) - 1) * inputs.size(1), -1)
		x = self.mlp1(x)
		x_skip = x

		if self.factor:
			x = self.edge2node(x, rel_rec, rel_send)
			x = self.mlp2(x)

			x = self.node2edge(x, rel_rec, rel_send)
			x = torch.cat((x, x_skip), dim=2)  # Skip connection
			x = self.mlp3(x)

		return self.fc_out(x)