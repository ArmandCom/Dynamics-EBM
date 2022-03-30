## Standard libraries
import os
import json
import math
import numpy as np
import random

## Imports for plotting
import matplotlib.pyplot as plt
from matplotlib import cm
# %matplotlib inline
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import MNIST, EMNIST, KMNIST
from torchvision import transforms
# PyTorch Lightning
# try:
# 	import pytorch_lightning as pl
# except ModuleNotFoundError: # Google Colab does not have PyTorch Lightning installed by default. Hence, we do it here if necessary
# 	!pip install --quiet pytorch-lightning>=1.4
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# CHECKPOINT_PATH = "./saved_models/tutorial8_dt_test" #ls_kernel_optimized
# asdf
n_workers = 4
device_id = 1
order = 4
dim = 10
num_samples = 500
pretrained = False
freeze = False
kernel = False
indep = True

if indep: from SoS_batch import SoS_loss
else: from SoS import SoS_loss

# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models/dt_dim{}_ord{}_indep{}_kernel{}_pretrained{}".format(dim, order, int(indep), int(kernel), int(pretrained)) #ls_kernel_optimized
# CHECKPOINT_PATH = "./saved´_models/debug" #ls_kernel_optimized
print(CHECKPOINT_PATH)

# Setting the seed
pl.seed_everything(42)

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "./data"

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device_id = 0
device = torch.device("cuda:"+str(device_id)) if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

import urllib.request
from urllib.error import HTTPError
# Github URL where saved models are stored for this tutorial
base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial8/"
# Files to download
pretrained_files = ["MNIST.ckpt", "tensorboards/events.out.tfevents.MNIST"]

# Create checkpoint path if it doesn't exist yet
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# For each file, check whether it already exists. If not, try downloading it.
for file_name in pretrained_files:
	file_path = os.path.join(CHECKPOINT_PATH, file_name)
	if "/" in file_name:
		os.makedirs(file_path.rsplit("/",1)[0], exist_ok=True)
	if not os.path.isfile(file_path):
		file_url = base_url + file_name
		print(f"Downloading {file_url}...")
		try:
			urllib.request.urlretrieve(file_url, file_path)
		except HTTPError as e:
			print("Something went wrong. Please try to download the file from the GDrive folder, "
				  "or contact the author with the full output including the following error:\n", e)


### Dataset ###
# Transformations applied on each image => make them a tensor and normalize between -1 and 1
transform = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize((0.5,), (0.5,))
								])

# Loading the training dataset. We need to split it into a training and validation part
train_set = MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
# test_set = MNIST(root=DATASET_PATH, train=False, transform=transform, download=True)

# Loading the test set
# train_set = EMNIST(root=DATASET_PATH, train=True, split='letters', transform=transform, download=True)
test_set = EMNIST(root=DATASET_PATH, train=False, split='letters', transform=transform, download=True)

# We define a set of data loaders that we can use for various purposes later.
# Note that for actually training a model, we will use different data loaders
# with a lower batch size.
train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True,  drop_last=True,  num_workers=n_workers, pin_memory=True)
test_loader  = data.DataLoader(test_set,  batch_size=256, shuffle=False, drop_last=False, num_workers=n_workers)

### Note: Model ###
class Swish(nn.Module):

	def forward(self, x):
		return x * torch.sigmoid(x)


class CNNModel(nn.Module):

	def __init__(self, hidden_features=32, num_samples_sos=1001, **kwargs):
		# Note: Same depth but last layer is of higher dimension
		super().__init__()
		# We increase the hidden dimension over layers. Here pre-calculated for simplicity.
		c_hid1 = hidden_features//2
		c_hid2 = hidden_features
		c_hid3 = hidden_features*2

		out_dim = kwargs['out_dim']

		self.freeze = freeze
		self.kernelized = kernel
		# Series of convolutions and Swish activation functions
		self.cnn_layers = nn.Sequential(
			nn.Conv2d(1, c_hid1, kernel_size=5, stride=2, padding=4), # [16x16] - Larger padding to get 32x32 image
			Swish(),
			nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1), #  [8x8]
			Swish(),
			nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1), # [4x4]
			Swish(),
			nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1), # [2x2]
			Swish(),
			nn.Flatten(),
			nn.Linear(c_hid3*4, c_hid3),
			Swish(), # Only keep the las
		)
		self.cnn_projection = nn.Sequential(
			nn.Linear(c_hid3, c_hid3),
			Swish(),
			nn.Linear(c_hid3, out_dim),
			nn.Softplus()) # Note: Seems like all values must be positive for the Veronesse map current formulation.

		self.sos = SoS_loss(out_dim, mord=order, num_samples=num_samples_sos, gpu_id=device)
		self.register_buffer('V', self.sos.vmap(torch.randn(self.sos.s, out_dim).abs().to(device)))
		self.update_Minv()

	def update_V(self, X):
		self.V = self.sos.vmap(X).detach().to(device)

	def update_Minv(self):
		self.Minv = self.sos.calcMinv(self.V, kernel=self.kernelized)

	def forward(self, x, sample_features=False):
		if self.freeze:
			with torch.no_grad():
				x = self.cnn_layers(x)
		else: x = self.cnn_layers(x)
		x = self.cnn_projection(x)
		if self.V.device != x.device:
			self.V = self.V.to(device) # TODO: This is very inefficient.
		if self.Minv.device != x.device: self.Minv = self.Minv.to(device)
		# Note: precompute Minv
		e = self.sos.calcQ(self.V, self.sos.vmap(x), self.Minv, kernelized=self.kernelized)
		if sample_features:
			return x, e
		else: return e

### Note: Sampler / Buffer ###
class Sampler:

	def __init__(self, model, img_shape, sample_size, max_len=8192, initialize_list=True):
		"""
		Inputs:
			model - Neural network to use for modeling E_theta
			img_shape - Shape of the images to model
			sample_size - Batch size of the samples
			max_len - Maximum number of data points to keep in the buffer
		"""
		super().__init__()
		self.model = model
		self.img_shape = img_shape
		self.sample_size = sample_size
		self.max_len = max_len
		if initialize_list:
			self.examples = [(torch.rand((1,)+img_shape)*2-1) for _ in range(self.sample_size)]
		else: self.examples = []

	def sample_new_exmps(self, steps=60, step_size=10):
		"""
		Function for getting a new batch of "fake" images.
		Inputs:
			steps - Number of iterations in the MCMC algorithm
			step_size - Learning rate nu in the algorithm above
		"""
		# Choose 95% of the batch from the buffer, 5% generate from scratch
		n_new = np.random.binomial(self.sample_size, 0.05)
		rand_imgs = torch.rand((n_new,) + self.img_shape) * 2 - 1
		old_imgs = torch.cat(random.choices(self.examples, k=self.sample_size-n_new), dim=0)
		inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to(device)

		# Perform MCMC sampling
		inp_imgs = Sampler.generate_samples(self.model, inp_imgs, steps=steps, step_size=step_size)

		# Add new images to the buffer and remove old ones if needed
		self.examples = list(inp_imgs.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.examples
		self.examples = self.examples[:self.max_len]
		return inp_imgs

	def save_features(self, inputs=None, feats=None):
		"""
		Function for getting a new batch of "fake" images.
		Inputs:
			steps - Number of iterations in the MCMC algorithm
			step_size - Learning rate nu in the algorithm above
		"""
		if feats is None and inputs is not None:
			feats = self.model(inputs, sample_features=True)[0]
		elif feats is not None and inputs is None: pass
		elif feats is not None and inputs is not None: raise NotImplementedError
		elif feats is None and inputs is None: raise NotImplementedError

		# Add new images to the buffer and remove old ones if needed
		self.examples = list(feats.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.examples
		self.examples = self.examples[:self.max_len]

	@staticmethod
	def generate_samples(model, inp_imgs, steps=60, step_size=10, return_img_per_step=False):
		"""
		Function for sampling images for a given model.
		Inputs:
			model - Neural network to use for modeling E_theta
			inp_imgs - Images to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
			steps - Number of iterations in the MCMC algorithm.
			step_size - Learning rate nu in the algorithm above
			return_img_per_step - If True, we return the sample at every iteration of the MCMC
		"""
		# Before MCMC: set model parameters to "required_grad=False"
		# because we are only interested in the gradients of the input.
		is_training = model.training
		model.eval()
		for p in model.parameters():
			p.requires_grad = False
		inp_imgs.requires_grad = True

		# Enable gradient calculation if not already the case
		had_gradients_enabled = torch.is_grad_enabled()
		torch.set_grad_enabled(True)

		# We use a buffer tensor in which we generate noise each loop iteration.
		# More efficient than creating a new tensor every iteration.
		noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)

		# List for storing generations at each step (for later analysis)
		imgs_per_step = []

		# Loop over K (steps)
		for _ in range(steps):
			# Part 1: Add noise to the input.
			noise.normal_(0, 0.005)
			inp_imgs.data.add_(noise.data)
			inp_imgs.data.clamp_(min=-1.0, max=1.0)

			# Part 2: calculate gradients for the current input.
			if model.freeze:
				model.freeze = False
				out_imgs = model(inp_imgs)
				model.freeze = True
			else: out_imgs = model(inp_imgs)

			out_imgs.sum().backward()
			inp_imgs.grad.data.clamp_(-0.03, 0.03) # For stabilizing and preventing too high gradients

			# Apply gradients to our current samples
			inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
			inp_imgs.grad.detach_()
			inp_imgs.grad.zero_()
			inp_imgs.data.clamp_(min=-1.0, max=1.0)

			if return_img_per_step:
				imgs_per_step.append(inp_imgs.clone().detach())

		# Reactivate gradients for parameters for training
		for p in model.parameters():
			p.requires_grad = True
		model.train(is_training)

		# Reset gradient calculation to setting before this function
		torch.set_grad_enabled(had_gradients_enabled)

		if return_img_per_step:
			return torch.stack(imgs_per_step, dim=0)
		else:
			return inp_imgs

### Note: Training / Overall Model ###
class DeepEnergyModel(pl.LightningModule):

	def __init__(self, img_shape, batch_size, num_samples_sos=1000, m_interval = 250, alpha=0.1, lr=1e-4, beta1=0.0, **CNN_args):
		super().__init__()
		self.save_hyperparameters()

		self.m_interval = m_interval
		self.num_samples_sos = num_samples_sos

		self.cnn = CNNModel(num_samples_sos = num_samples_sos, **CNN_args)
		self.sampler = Sampler(self.cnn, img_shape=img_shape, sample_size=batch_size)
		self.feat_buffer = Sampler(self.cnn, img_shape=(CNN_args['out_dim'],), sample_size=batch_size,
								   max_len=num_samples_sos, initialize_list=False) # Max len here will be the number of samples to calculate
		self.feat_buffer_test = Sampler(self.cnn, img_shape=(CNN_args['out_dim'],), sample_size=test_loader.batch_size,
								   max_len=num_samples_sos, initialize_list=False)
		self.example_input_array = torch.zeros(1, *img_shape)

	def forward(self, x):
		z = self.cnn(x)
		return z

	def load_partial_state_dict(self, state_dict, freeze=True):

		own_state = self.state_dict()#
		state_dict = torch.load(state_dict)['state_dict']
		for name, param in state_dict.items():
			if name not in own_state:
				print(name)
				print()
				continue
			if isinstance(param, nn.Parameter):
				# backwards compatibility for serialized parameters
				param = param.data
			own_state[name].copy_(param)
		self.load_state_dict(own_state)

	def configure_optimizers(self):
		# Energy models can have issues with momentum as the loss surfaces changes with its parameters.
		# Hence, we set it to 0 by default.
		optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
		scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97) # Exponential decay over epochs
		return [optimizer], [scheduler]

	def training_step(self, batch, batch_idx):
		# We add minimal noise to the original images to prevent the model from focusing on purely "clean" inputs
		real_imgs, _ = batch
		small_noise = torch.randn_like(real_imgs) * 0.005
		real_imgs.add_(small_noise).clamp_(min=-1.0, max=1.0)

		# Obtain samples
		fake_imgs = self.sampler.sample_new_exmps(steps=60, step_size=10)

		# Predict energy score for all images
		inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)

		feat, es = self.cnn(inp_imgs, sample_features=True)
		real_out, fake_out = es.chunk(2, dim=0)
		self.feat_buffer.save_features(feats=feat.chunk(2, dim=0)[0]) # Add the real features to the buffer

		if (len(self.feat_buffer.examples) >= self.num_samples_sos
			and batch_idx % self.m_interval == 0):
			self.cnn.update_V( torch.cat(self.feat_buffer.examples) )
			self.cnn.update_Minv()
			# print('Regenerating V \n')


		# Calculate losses
		reg_loss = self.hparams.alpha * (real_out ** 2 + fake_out ** 2).mean()
		cdiv_loss = real_out.mean() - fake_out.mean()
		loss = reg_loss + cdiv_loss

		# Logging
		self.log('loss', loss)
		self.log('loss_regularization', reg_loss)
		self.log('loss_contrastive_divergence', cdiv_loss)
		self.log('metrics_avg_real', real_out.mean())
		self.log('metrics_avg_fake', fake_out.mean())
		return loss

	def validation_step(self, batch, batch_idx):
		# For validating, we calculate the contrastive divergence between purely random images and unseen examples
		# Note that the validation/test step of energy-based models depends on what we are interested in the model
		real_imgs, _ = batch
		fake_imgs = torch.rand_like(real_imgs) * 2 - 1

		inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
		real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

		cdiv = real_out.mean() - fake_out.mean()
		self.log('val_contrastive_divergence', cdiv)
		self.log('val_fake_out', fake_out.mean())
		self.log('val_real_out', real_out.mean())

### Note: Callbacks ###
class GenerateCallback(pl.Callback):

	def __init__(self, batch_size=8, vis_steps=8, num_steps=256, every_n_epochs=5):
		super().__init__()
		self.batch_size = batch_size         # Number of images to generate
		self.vis_steps = vis_steps           # Number of steps within generation to visualize
		self.num_steps = num_steps           # Number of steps to take during generation
		self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

	def on_epoch_end(self, trainer, pl_module):
		# Skip for all other epochs
		if trainer.current_epoch % self.every_n_epochs == 0:
			# Generate images
			imgs_per_step = self.generate_imgs(pl_module)
			# Plot and add to tensorboard
			for i in range(imgs_per_step.shape[1]):
				step_size = self.num_steps // self.vis_steps
				imgs_to_plot = imgs_per_step[step_size-1::step_size,i]
				grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True, range=(-1,1))
				trainer.logger.experiment.add_image(f"generation_{i}", grid, global_step=trainer.current_epoch)

	def generate_imgs(self, pl_module):
		pl_module.eval()
		start_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(pl_module.device)
		start_imgs = start_imgs * 2 - 1
		torch.set_grad_enabled(True)  # Tracking gradients for sampling necessary
		imgs_per_step = Sampler.generate_samples(pl_module.cnn, start_imgs, steps=self.num_steps, step_size=10, return_img_per_step=True)
		torch.set_grad_enabled(False)
		pl_module.train()
		return imgs_per_step

class SamplerCallback(pl.Callback):

	def __init__(self, num_imgs=32, every_n_epochs=5):
		super().__init__()
		self.num_imgs = num_imgs             # Number of images to plot
		self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

	def on_epoch_end(self, trainer, pl_module):
		if trainer.current_epoch % self.every_n_epochs == 0:
			exmp_imgs = torch.cat(random.choices(pl_module.sampler.examples, k=self.num_imgs), dim=0)
			grid = torchvision.utils.make_grid(exmp_imgs, nrow=4, normalize=True, range=(-1,1))
			trainer.logger.experiment.add_image("sampler", grid, global_step=trainer.current_epoch)

class DataCallback(pl.Callback):

	def __init__(self, dataloader, num_imgs=32, every_n_epochs=5):
		super().__init__()
		self.num_imgs = num_imgs             # Number of images to plot
		self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)
		self.dataloader = dataloader

	def on_epoch_end(self, trainer, pl_module):
		if trainer.current_epoch % self.every_n_epochs == 0:
			len = 0
			all_imgs = []
			while len < self.num_imgs:
				data_imgs,_ = next(iter(self.dataloader))
				all_imgs.append(data_imgs)
				len += data_imgs.shape[0]
			exmp_imgs = torch.cat(all_imgs, dim=0)[:self.num_imgs]
			grid = torchvision.utils.make_grid(exmp_imgs, nrow=4, normalize=True, range=(-1,1))
			trainer.logger.experiment.add_image("original_imgs_train{}".format(int(self.dataloader.dataset.train)), grid, global_step=trainer.current_epoch)

class OutlierCallback(pl.Callback):

	def __init__(self, batch_size=1024):
		super().__init__()
		self.batch_size = batch_size

	def on_epoch_end(self, trainer, pl_module):
		with torch.no_grad():
			pl_module.eval()
			rand_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(pl_module.device)
			rand_imgs = rand_imgs * 2 - 1.0
			rand_out = pl_module.cnn(rand_imgs).mean()
			pl_module.train()

		trainer.logger.experiment.add_scalar("rand_out", rand_out, global_step=trainer.current_epoch)

class TransferCallback(pl.Callback):

	def __init__(self, dataloader, num_samples=1001, batch_size=8, vis_steps=8, num_steps=256, every_n_epochs=5):
		super().__init__()
		self.dataloader = dataloader
		self.batch_size = batch_size         # Number of images to generate
		self.vis_steps = vis_steps           # Number of steps within generation to visualize
		self.num_steps = num_steps           # Number of steps to take during generation
		self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)
		self.num_samples = num_samples

	def on_epoch_end(self, trainer, pl_module):
		# Skip for all other epochs
		if trainer.current_epoch % self.every_n_epochs == 0:

			# Modify V and Minv with the new data and sample according to it.
			with torch.no_grad():
				len = 0
				# test_imgs = []
				while len < self.num_samples:
					data_imgs,_ = next(iter(self.dataloader))
					pl_module.feat_buffer_test.save_features(inputs=data_imgs.to(device))
					len += data_imgs.shape[0]
				pl_module.cnn.update_V( torch.cat(pl_module.feat_buffer_test.examples) )
				pl_module.cnn.update_Minv()
			# Generate images
			imgs_per_step = self.generate_imgs(pl_module)

			# Plot and add to tensorboard
			for i in range(imgs_per_step.shape[1]):
				step_size = self.num_steps // self.vis_steps
				imgs_to_plot = imgs_per_step[step_size-1::step_size,i]
				grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True, range=(-1,1))
				trainer.logger.experiment.add_image(f"generation_letter_{i}", grid, global_step=trainer.current_epoch)

			# V and Minv back to training mode
			if trainer.current_epoch > 0:
				pl_module.cnn.update_V( torch.cat(pl_module.feat_buffer.examples) )
				pl_module.cnn.update_Minv()


	def generate_imgs(self, pl_module):
		pl_module.eval()
		start_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(pl_module.device)
		start_imgs = start_imgs * 2 - 1
		torch.set_grad_enabled(True)  # Tracking gradients for sampling necessary
		imgs_per_step = Sampler.generate_samples(pl_module.cnn, start_imgs, steps=self.num_steps, step_size=10, return_img_per_step=True)
		torch.set_grad_enabled(False)
		pl_module.train()
		return imgs_per_step

class TrainTransferCompCallback(pl.Callback):

	def __init__(self, train_dataloader, test_dataloader, num_samples=1001, batch_size=8, vis_steps=8, num_steps=256, every_n_epochs=5):
		super().__init__()
		self.train_dataloader = train_dataloader
		self.test_dataloader  = test_dataloader
		self.batch_size = batch_size         # Number of images to generate
		self.vis_steps = vis_steps           # Number of steps within generation to visualize
		self.num_steps = num_steps           # Number of steps to take during generation
		self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)
		self.num_samples = num_samples

	def on_epoch_end(self, trainer, pl_module):
		# Skip for all other epochs
		if trainer.current_epoch % self.every_n_epochs == 0:


			start_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(pl_module.device)
			start_imgs = start_imgs * 2 - 1

			# Modify V and Minv with the new data and sample according to it.
			with torch.no_grad():
				len = 0
				# test_imgs = []
				while len < self.num_samples:
					data_imgs,_ = next(iter(self.test_dataloader))
					pl_module.feat_buffer_test.save_features(inputs=data_imgs.to(device))
					len += data_imgs.shape[0]
				pl_module.cnn.update_V( torch.cat(pl_module.feat_buffer_test.examples) )
				pl_module.cnn.update_Minv()
			# Generate images
			imgs_per_step_test = self.generate_imgs(pl_module, start_imgs)

			# Modify V and Minv with the new data and sample according to it.
			with torch.no_grad():
				len = 0
				# test_imgs = []
				while len < self.num_samples:
					data_imgs,_ = next(iter(self.train_dataloader))
					pl_module.feat_buffer.save_features(inputs=data_imgs.to(device))
					len += data_imgs.shape[0]
				pl_module.cnn.update_V( torch.cat(pl_module.feat_buffer.examples) )
				pl_module.cnn.update_Minv()
			# Generate images
			imgs_per_step_train = self.generate_imgs(pl_module, start_imgs)

			imgs_per_step = torch.cat([imgs_per_step_train, imgs_per_step_test], dim=-2)

			# V and Minv back to training mode
			if trainer.current_epoch > 0:
				pl_module.cnn.update_V( torch.cat(pl_module.feat_buffer.examples) )
				pl_module.cnn.update_Minv()

			# Plot and add to tensorboard
			for i in range(imgs_per_step.shape[1]):
				step_size = self.num_steps // self.vis_steps
				imgs_to_plot = imgs_per_step[step_size-1::step_size,i]
				grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True, range=(-1,1))
				trainer.logger.experiment.add_image(f"generation_train-test_{i}", grid, global_step=trainer.current_epoch)


	def generate_imgs(self, pl_module, start_imgs):
		pl_module.eval()
		torch.set_grad_enabled(True)  # Tracking gradients for sampling necessary
		imgs_per_step = Sampler.generate_samples(pl_module.cnn, start_imgs, steps=self.num_steps, step_size=10, return_img_per_step=True)
		torch.set_grad_enabled(False)
		pl_module.train()
		return imgs_per_step

### Note: Running model ###
def train_model(**kwargs):
	# Create a PyTorch Lightning trainer with the generation callback
	trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "MNIST"),
						 gpus=[device_id] if str(device).startswith("cuda") else 0,
						 max_epochs=100,
						 gradient_clip_val=0.1,
						 callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor='val_contrastive_divergence'),
									# GenerateCallback(every_n_epochs=5),
									SamplerCallback(every_n_epochs=5),
									OutlierCallback(),
									# TransferCallback(test_loader,
									# 				 num_samples=kwargs['num_samples_sos'],
									# 				 every_n_epochs=1, num_steps=512),
									TrainTransferCompCallback(train_loader, test_loader,
															  num_samples=kwargs['num_samples_sos'],
															  every_n_epochs=1, num_steps=256),
									DataCallback(train_loader, num_imgs=32, every_n_epochs=20),
									DataCallback(test_loader, num_imgs=32, every_n_epochs=20),
									LearningRateMonitor("epoch")
									],
						 progress_bar_refresh_rate=1)
	# else:
	# Note: Do training
	pl.seed_everything(42)
	model = DeepEnergyModel(**kwargs)

	# Note: Trying partial feature
	# Check whether pretrained model exists. If yes, load it and skip training
	if pretrained:
		pretrained_filename = os.path.join(CHECKPOINT_PATH, "MNIST.ckpt")
		if os.path.isfile(pretrained_filename):
			print("Found pretrained model, loading...")
			model.load_partial_state_dict(pretrained_filename)

	trainer.fit(model, train_loader, test_loader)
	# model = DeepEnergyModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
	# No testing as we are more interested in other properties
	return model

if __name__ == '__main__':
	### Note: Create model ###
	model = train_model(img_shape=(1,28,28),
						batch_size=train_loader.batch_size,
						num_samples_sos = num_samples, #Num of samles needed for matrix M
						lr=1e-4,
						beta1=0.0,
						out_dim=dim
						)

	print('Train done!')
	### Note: Image generation ###
	# model.to(device)
	# pl.seed_everything(43)
	# callback = GenerateCallback(batch_size=4, vis_steps=8, num_steps=256)
	# imgs_per_step = callback.generate_imgs(model)
	# imgs_per_step = imgs_per_step.cpu()
	#
	# for i in range(imgs_per_step.shape[1]):
	# 	step_size = callback.num_steps // callback.vis_steps
	# 	imgs_to_plot = imgs_per_step[step_size-1::step_size,i]
	# 	imgs_to_plot = torch.cat([imgs_per_step[0:1,i],imgs_to_plot], dim=0)
	# 	grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True, range=(-1,1), pad_value=0.5, padding=2)
	# 	grid = grid.permute(1, 2, 0)
	# 	plt.figure(figsize=(8,8))
	# 	plt.imshow(grid)
	# 	plt.xlabel("Generation iteration")
	# 	plt.xticks([(imgs_per_step.shape[-1]+2)*(0.5+j) for j in range(callback.vis_steps+1)],
	# 			   labels=[1] + list(range(step_size,imgs_per_step.shape[0]+1,step_size)))
	# 	plt.yticks([])
	# 	plt.show()

	### Note: Domain Transfer generation ###

	### Note: OOD detection ###
	# with torch.no_grad():
	# 	rand_imgs = torch.rand((128,) + model.hparams.img_shape).to(model.device)
	# 	rand_imgs = rand_imgs * 2 - 1.0
	# 	rand_out = model.cnn(rand_imgs).mean()
	# 	print(f"Average score for random images: {rand_out.item():4.2f}")
	#
	# with torch.no_grad():
	# 	train_imgs,_ = next(iter(train_loader))
	# 	train_imgs = train_imgs.to(model.device)
	# 	train_out = model.cnn(train_imgs).mean()
	# 	print(f"Average score for training images: {train_out.item():4.2f}")
	#
	# @torch.no_grad()
	# def compare_images(img1, img2):
	# 	imgs = torch.stack([img1, img2], dim=0).to(model.device)
	# 	score1, score2 = model.cnn(imgs).cpu().chunk(2, dim=0)
	# 	grid = torchvision.utils.make_grid([img1.cpu(), img2.cpu()], nrow=2, normalize=True, range=(-1,1), pad_value=0.5, padding=2)
	# 	grid = grid.permute(1, 2, 0)
	# 	plt.figure(figsize=(4,4))
	# 	plt.imshow(grid)
	# 	plt.xticks([(img1.shape[2]+2)*(0.5+j) for j in range(2)],
	# 			   labels=["Original image", "Transformed image"])
	# 	plt.yticks([])
	# 	plt.show()
	# 	print(f"Score original image: {score1:4.2f}")
	# 	print(f"Score transformed image: {score2:4.2f}")
	#
	# test_imgs, _ = next(iter(test_loader))
	# exmp_img = test_imgs[0].to(model.device)
	#
	# img_noisy = exmp_img + torch.randn_like(exmp_img) * 0.3
	# img_noisy.clamp_(min=-1.0, max=1.0)
	# compare_images(exmp_img, img_noisy)
	#
	# img_flipped = exmp_img.flip(dims=(1,2))
	# compare_images(exmp_img, img_flipped)
	#
	# img_tiny = torch.zeros_like(exmp_img)-1
	# img_tiny[:,exmp_img.shape[1]//2:,exmp_img.shape[2]//2:] = exmp_img[:,::2,::2]
	# compare_images(exmp_img, img_tiny)