import os
import os.path as osp
import numpy as np
import json

import torchvision.transforms.functional as TF
import random

from PIL import Image
import torch.utils.data as data
import torch
import cv2
from torchvision import transforms
import glob
from third_party.utils import visualize_trajectories

try:
    import multi_dsprites
    import tetrominoes
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    pass

from glob import glob

from imageio import imread
from skimage.transform import resize as imresize

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

class GaussianBlur(object):

    def __init__(self, min=0.1, max=2.0, kernel_size=9):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

class ChargedParticlesSim(data.Dataset):
    def __init__(self, args, split):
        self.args = args

class SpringsParticles(data.Dataset):
    def __init__(self, args, split):
        # n_balls=5, box_size=5., loc_std=1., vel_norm=0.5,
        # interaction_strength=1., noise_var=0.):
        self.args = args
        old = False
        self.n_objects = args.n_objects
        suffix = '_springs'
        # suffix += '-random-temp'
        suffix += str(self.n_objects)
        # suffix += 'inter0.1_nowalls_sf100_len5000'
        if old:
            suffix += 'inter0.1_sf50_lentrain5000_nstrain50000'
        # suffix += 'inter0.1_sf100_lentrain10000_nstrain50000'
        # suffix += 'inter_s0.1-1.0_sf50_lentrain5000_nstrain50000'
        # print('Dataset: SPRINGS with random edge value 0.1 - 1')
        # suffix += 'inter0.1_nowalls_sf20_lentrain5000_nstrain50000' # 249 samples # TODO: Change velocity normalization

        if old:
            feat, edges, stats = self._load_data_old(suffix=suffix, split=split)
        else: feat, edges, stats = self._load_data(suffix=suffix, split=split)

        # TODO: loc_max, loc_min, vel_max, vel_min
        assert self.n_objects == feat.shape[1]
        self.length = feat.shape[0]
        self.timesteps = args.num_timesteps #feat.shape[2]

        # visualize_trajectories(torch.tensor(feat), None, torch.tensor(edges))

        # self.box_size = 5.
        # self.loc_std = 1.
        # self.vel_norm = 0.5
        # self.interaction_strength = args.interaction_strength
        # self.noise_var = args.noise_var
        # self.sample_freq = args.sample_freq # default: 100
        # self.sequence_length = args.sequence_length # default: 1000

        # print('-Randomly placed trajectory.')
        # ini_id = np.random.randint(0, feat.shape[2]-self.timesteps, (feat.shape[0],))[:, None].repeat(self.timesteps, 1)
        # batch_id = np.arange(0, feat.shape[0])[:, None].repeat(self.timesteps, 1)
        # ini_id += np.arange(0,self.timesteps)[None].repeat(feat.shape[0], 0)
        # self.feat, self.edges = np.transpose(feat[batch_id, :, ini_id],(0,2,1,3)), edges

        print('-Trajectory begins at start.')
        self.feat, self.edges = feat[:, :, :self.timesteps], edges

        off_diag = np.ones([args.n_objects, args.n_objects]) - np.eye(args.n_objects)
        self.rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        self.rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing
        """
        # if self.datasource == 'default':
        #     im_corrupt = im + 0.3 * torch.randn(self.image_size, self.image_size, 3)
        # elif self.datasource == 'random':
        #     im_corrupt = 0.5 + 0.5 * torch.randn(self.image_size, self.image_size, 3)
        # else:
        #     raise NotImplementedError
        return (torch.tensor(self.feat[index]), torch.tensor(self.edges[index])), \
               (torch.tensor(self.rel_rec), torch.tensor(self.rel_send)), index

    def __len__(self):
        """Return the total number of trajectories in the dataset."""
        return self.length

    def _load_data(self, batch_size=1, suffix='', split='train'):
        loc_train = np.load('/data/Armand/NRIori/loc_train' + suffix + '.npy')
        vel_train = np.load('/data/Armand/NRIori/vel_train' + suffix + '.npy')
        edges_train = np.load('/data/Armand/NRIori/edges_train' + suffix + '.npy')

        loc_valid = np.load('/data/Armand/NRIori/loc_valid' + suffix + '.npy')
        vel_valid = np.load('/data/Armand/NRIori/vel_valid' + suffix + '.npy')
        edges_valid = np.load('/data/Armand/NRIori/edges_valid' + suffix + '.npy')

        loc_test = np.load('/data/Armand/NRIori/loc_test' + suffix + '.npy')
        vel_test = np.load('/data/Armand/NRIori/vel_test' + suffix + '.npy')
        edges_test = np.load('/data/Armand/NRIori/edges_test' + suffix + '.npy')

        # [num_samples, num_timesteps, num_dims, num_atoms]
        num_atoms = loc_train.shape[3]

        loc_max = loc_train.max()
        loc_min = loc_train.min()
        vel_max = vel_train.max()
        vel_min = vel_train.min()

        # Normalize to [-1, 1]
        loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
        vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

        loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
        vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

        loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
        vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

        # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
        loc_train = np.transpose(loc_train, [0, 3, 1, 2])
        vel_train = np.transpose(vel_train, [0, 3, 1, 2])
        feat_train = np.concatenate([loc_train, vel_train], axis=3)
        edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
        edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

        loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
        vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
        feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
        edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
        edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

        loc_test = np.transpose(loc_test, [0, 3, 1, 2])
        vel_test = np.transpose(vel_test, [0, 3, 1, 2])
        feat_test = np.concatenate([loc_test, vel_test], axis=3)
        edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
        edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

        feat_train = torch.FloatTensor(feat_train)
        edges_train = torch.LongTensor(edges_train)
        feat_valid = torch.FloatTensor(feat_valid)
        edges_valid = torch.LongTensor(edges_valid)
        feat_test = torch.FloatTensor(feat_test)
        edges_test = torch.LongTensor(edges_test)

        # Exclude self edges
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
            [num_atoms, num_atoms])
        edges_train = edges_train[:, off_diag_idx]
        edges_valid = edges_valid[:, off_diag_idx]
        edges_test = edges_test[:, off_diag_idx]

        if split == 'train':
            feat = feat_train
            edges = edges_train
        elif split == 'valid':
            feat = feat_valid
            edges = edges_valid
        elif split == 'test':
            feat = feat_test
            edges = edges_test
        else: raise NotImplementedError
        return feat, edges, (loc_max, loc_min, vel_max, vel_min)

    def _load_data_old(self, suffix='', split='train'):
        loc = np.load('/data/Armand/NRI/loc_' + split + suffix + '.npy')
        vel = np.load('/data/Armand/NRI/vel_' + split + suffix + '.npy')
        edges = np.load('/data/Armand/NRI/edges_' + split + suffix + '.npy')

        # [num_samples, num_timesteps, num_dims, num_atoms]
        num_atoms = loc.shape[3]

        # Note: unnormalize

        loc_max = loc.max()
        loc_min = loc.min()
        vel = vel / 20
        # Note: In simulation our increase in T (delta T) is 0.001.
        #  Then we sample 1/100 generated samples.
        #  Therefore the ratio between loc and velocity is vel/(incrLoc) = 10
        vel_max = vel.max()
        vel_min = vel.min()

        print("Normalized Springs Dataset")
        # Normalize to [-1, 1]
        loc = (loc - loc_min) * 2 / (loc_max - loc_min) - 1
        # vel = (vel - vel_min) * 2 / (vel_max - vel_min) - 1
        vel = vel * 2 / (loc_max - loc_min)

        # print("Standardized Springs Dataset")
        # loc_mean = loc.mean()
        # loc_std = loc.std()
        # loc = (loc - loc_mean) / loc_std
        # vel = vel / loc_std

        # print("Unnormalized Spring Dataset")
        # loc_max = None
        # loc_min = None
        # vel_max = None
        # vel_min = None
        # loc = loc / 5.
        # vel = vel / 5.

        # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
        loc = np.transpose(loc, [0, 3, 1, 2])
        vel = np.transpose(vel, [0, 3, 1, 2])
        feat = np.concatenate([loc, vel], axis=3)
        edges = np.reshape(edges, [-1, num_atoms ** 2])
        edges = np.array((edges + 1) / 2, dtype=np.int64)

        feat = torch.FloatTensor(feat)
        edges = torch.LongTensor(edges)

        # Exclude self edges
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
            [num_atoms, num_atoms])
        edges = edges[:, off_diag_idx]

        # data = TensorDataset(feat, edges)
        # data_loader = DataLoader(data, batch_size=batch_size)
        # TODO: we need a way to encode the walls. Maybe add the minmax.
        return feat, edges, (loc_max, loc_min, vel_max, vel_min) # TODO: Check how mins and maxes are used

class ChargedParticles(data.Dataset):
    def __init__(self, args, split):
        # n_balls=5, box_size=5., loc_std=1., vel_norm=0.5,
        # interaction_strength=1., noise_var=0.):
        self.args = args
        self.n_objects = args.n_objects
        old = False
        suffix = '_charged'+str(self.n_objects)
        # suffix += '_nobox_05int-strength'
        # suffix += 'inter0.5_nowalls_sf100_len5000'
        if old:
            suffix += 'inter0.5_sf50_lentrain5000_nstrain50000'
        # suffix += ''
        # suffix += 'inter0.5_nowalls_sf20_lentrain5000_nstrain50000' # 249 samples # TODO: Change velocity normalization

        if old:
            feat, edges, stats = self._load_data_old(suffix=suffix, split=split)

        else: feat, edges, stats = self._load_data(suffix=suffix, split=split)
        # TODO: loc_max, loc_min, vel_max, vel_min
        assert self.n_objects == feat.shape[1]
        self.length = feat.shape[0]
        self.timesteps = args.num_timesteps #feat.shape[2]

        # visualize_trajectories(torch.tensor(feat), None, torch.tensor(edges))

        # self.box_size = 5.
        # self.loc_std = 1.
        # self.vel_norm = 0.5
        # self.interaction_strength = args.interaction_strength
        # self.noise_var = args.noise_var
        # self.sample_freq = args.sample_freq # default: 100
        # self.sequence_length = args.sequence_length # default: 1000

        # print('-Randomly placed trajectory.')
        # ini_id = np.random.randint(0, feat.shape[2]-self.timesteps, (feat.shape[0],))[:, None].repeat(self.timesteps, 1)
        # batch_id = np.arange(0, feat.shape[0])[:, None].repeat(self.timesteps, 1)
        # ini_id += np.arange(0,self.timesteps)[None].repeat(feat.shape[0], 0)
        # self.feat, self.edges = np.transpose(feat[batch_id, :, ini_id],(0,2,1,3)), edges

        print('-Trajectory begins at start.')
        self.feat, self.edges = feat[:, :, :self.timesteps], edges

        off_diag = np.ones([args.n_objects, args.n_objects]) - np.eye(args.n_objects)
        self.rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        self.rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing
        """
        # if self.datasource == 'default':
        #     im_corrupt = im + 0.3 * torch.randn(self.image_size, self.image_size, 3)
        # elif self.datasource == 'random':
        #     im_corrupt = 0.5 + 0.5 * torch.randn(self.image_size, self.image_size, 3)
        # else:
        #     raise NotImplementedError

        return (torch.tensor(self.feat[index]), torch.tensor(self.edges[index])), \
               (torch.tensor(self.rel_rec), torch.tensor(self.rel_send)), index

    def __len__(self):
        """Return the total number of trajectories in the dataset."""
        return self.length

    def _load_data(self, batch_size=1, suffix='', split='train'):
        loc_train = np.load('/data/Armand/NRIori/loc_train' + suffix + '.npy')
        vel_train = np.load('/data/Armand/NRIori/vel_train' + suffix + '.npy')
        edges_train = np.load('/data/Armand/NRIori/edges_train' + suffix + '.npy')

        loc_valid = np.load('/data/Armand/NRIori/loc_valid' + suffix + '.npy')
        vel_valid = np.load('/data/Armand/NRIori/vel_valid' + suffix + '.npy')
        edges_valid = np.load('/data/Armand/NRIori/edges_valid' + suffix + '.npy')

        loc_test = np.load('/data/Armand/NRIori/loc_test' + suffix + '.npy')
        vel_test = np.load('/data/Armand/NRIori/vel_test' + suffix + '.npy')
        edges_test = np.load('/data/Armand/NRIori/edges_test' + suffix + '.npy')

        # [num_samples, num_timesteps, num_dims, num_atoms]
        num_atoms = loc_train.shape[3]

        loc_max = loc_train.max()
        loc_min = loc_train.min()
        vel_max = vel_train.max()
        vel_min = vel_train.min()

        # Normalize to [-1, 1]
        loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
        vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

        loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
        vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

        loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
        vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

        # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
        loc_train = np.transpose(loc_train, [0, 3, 1, 2])
        vel_train = np.transpose(vel_train, [0, 3, 1, 2])
        feat_train = np.concatenate([loc_train, vel_train], axis=3)
        edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
        edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

        loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
        vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
        feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
        edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
        edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

        loc_test = np.transpose(loc_test, [0, 3, 1, 2])
        vel_test = np.transpose(vel_test, [0, 3, 1, 2])
        feat_test = np.concatenate([loc_test, vel_test], axis=3)
        edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
        edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

        feat_train = torch.FloatTensor(feat_train)
        edges_train = torch.LongTensor(edges_train)
        feat_valid = torch.FloatTensor(feat_valid)
        edges_valid = torch.LongTensor(edges_valid)
        feat_test = torch.FloatTensor(feat_test)
        edges_test = torch.LongTensor(edges_test)

        # Exclude self edges
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
            [num_atoms, num_atoms])
        edges_train = edges_train[:, off_diag_idx]
        edges_valid = edges_valid[:, off_diag_idx]
        edges_test = edges_test[:, off_diag_idx]

        if split == 'train':
            feat = feat_train
            edges = edges_train
        elif split == 'valid':
            feat = feat_valid
            edges = edges_valid
        elif split == 'test':
            feat = feat_test
            edges = edges_test
        else: raise NotImplementedError
        return feat, edges, (loc_max, loc_min, vel_max, vel_min)


    def _load_data_old(self, suffix='', split='train'):
        loc = np.load('/data/Armand/NRI/loc_' + split + suffix + '.npy')
        vel = np.load('/data/Armand/NRI/vel_' + split + suffix + '.npy')
        edges = np.load('/data/Armand/NRI/edges_' + split + suffix + '.npy')

        # [num_samples, num_timesteps, num_dims, num_atoms]
        num_atoms = loc.shape[3]

        # Note: unnormalize

        loc_max = loc.max()
        loc_min = loc.min()
        vel = vel / 20 # 10 if sf 100, 50 if sf 20
        # Note: In simulation our increase in T (delta T) is 0.001.
        #  Then we sample 1/100 generated samples.
        #  Therefore the ratio between loc and velocity is vel/(incrLoc) = 10
        vel_max = vel.max()
        vel_min = vel.min()

        print("Normalized Charged Dataset")
        # Normalize to [-1, 1]
        loc = (loc - loc_min) * 2 / (loc_max - loc_min) - 1
        vel = vel * 2 / (loc_max - loc_min)
        # vel = (vel - vel_min) * 2 / (vel_max - vel_min) - 1


        # print("Standardized Charged Dataset")
        # loc_mean = loc.mean()
        # loc_std = loc.std()
        # loc = (loc - loc_mean) / loc_std
        # vel = vel / loc_std

        # print("Unnormalized Charged Dataset")

        # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
        loc = np.transpose(loc, [0, 3, 1, 2])
        vel = np.transpose(vel, [0, 3, 1, 2])
        feat = np.concatenate([loc, vel], axis=3)
        edges = np.reshape(edges, [-1, num_atoms ** 2])
        edges = np.array((edges + 1) / 2, dtype=np.int64)

        feat = torch.FloatTensor(feat)
        edges = torch.LongTensor(edges)

        # Exclude self edges
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
            [num_atoms, num_atoms])
        edges = edges[:, off_diag_idx]

        # data = TensorDataset(feat, edges)
        # data_loader = DataLoader(data, batch_size=batch_size)
        # TODO: we need a way to encode the walls. Maybe add the minmax.
        return feat, edges, (loc_max, loc_min, vel_max, vel_min) # TODO: Check how mins and maxes are used

class ChargedSpringsParticles(data.Dataset):
    def __init__(self, args, split):
        # n_balls=5, box_size=5., loc_std=1., vel_norm=0.5,
        # interaction_strength=1., noise_var=0.):
        self.args = args
        self.n_objects = args.n_objects

        suffix = '_charged-springs'+str(self.n_objects)
        # suffix += '_nobox_05int-strength'
        print('------TRAINING with 1000 SAMPLES-------')
        suffix +='1000samples_inter_s0.1_c0.5_sf50_lentrain5000_nstrain1000_mixedbynode'

        # suffix += 'inter_s0.1_c0.5_nowalls_sf100_lentrain5000_nstrain1_mixedbynode' # Only for test.

        # suffix += 'inter0.5_nowalls_sf100_len5000_test-mixed'

        # suffix += 'inter0.5_nowalls_sf10'

        feat, edges, stats = self._load_data(suffix=suffix, split=split)
        # TODO: loc_max, loc_min, vel_max, vel_min
        assert self.n_objects == feat.shape[1]
        self.length = feat.shape[0]
        self.timesteps = args.num_timesteps #feat.shape[2]

        # visualize_trajectories(torch.tensor(feat), None, torch.tensor(edges))

        # self.box_size = 5.
        # self.loc_std = 1.
        # self.vel_norm = 0.5
        # self.interaction_strength = args.interaction_strength
        # self.noise_var = args.noise_var
        # self.sample_freq = args.sample_freq # default: 100
        # self.sequence_length = args.sequence_length # default: 1000

        # Generate off-diagonal interaction graph
        self.feat, self.edges = feat[:, :, :self.timesteps], edges


        off_diag = np.ones([args.n_objects, args.n_objects]) - np.eye(args.n_objects)
        self.rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        self.rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing
        """
        # if self.datasource == 'default':
        #     im_corrupt = im + 0.3 * torch.randn(self.image_size, self.image_size, 3)
        # elif self.datasource == 'random':
        #     im_corrupt = 0.5 + 0.5 * torch.randn(self.image_size, self.image_size, 3)
        # else:
        #     raise NotImplementedError
        return (torch.tensor(self.feat[index]), torch.tensor(self.edges[index])), \
               (torch.tensor(self.rel_rec), torch.tensor(self.rel_send)), index

    def __len__(self):
        """Return the total number of trajectories in the dataset."""
        return self.length

    def _load_data(self, suffix='', split='train'):
        loc = np.load('/data/Armand/NRI/loc_' + split + suffix + '.npy')
        vel = np.load('/data/Armand/NRI/vel_' + split + suffix + '.npy')
        edges = np.load('/data/Armand/NRI/edges_' + split + suffix + '.npy')

        # [num_samples, num_timesteps, num_dims, num_atoms]
        num_atoms = loc.shape[3]

        # Note: unnormalize

        loc_max = loc.max()
        loc_min = loc.min()
        vel = vel / 20
        # Note: In simulation our increase in T (delta T) is 0.001.
        #  Then we sample 1/10 generated samples.
        #  Therefore the ratio between loc and velocity is vel/(incrLoc) = 10
        vel_max = vel.max()
        vel_min = vel.min()

        print("Normalized Charged Dataset")
        # Normalize to [-1, 1]
        loc = (loc - loc_min) * 2 / (loc_max - loc_min) - 1
        vel = vel * 2 / (loc_max - loc_min)
        # vel = (vel - vel_min) * 2 / (vel_max - vel_min) - 1

        # print("Unnormalized Charged Dataset")

        # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
        loc = np.transpose(loc, [0, 3, 1, 2])
        vel = np.transpose(vel, [0, 3, 1, 2])
        feat = np.concatenate([loc, vel], axis=3)


        # Note: Here, we use edges differently, as we want to know which of them has been modified
        # edges = np.reshape(edges, [-1, num_atoms ** 2])
        # edges = np.array((edges + 1) / 2, dtype=np.int64)

        feat = torch.FloatTensor(feat)
        edges = torch.LongTensor(edges)

        # Exclude self edges
        # off_diag_idx = np.ravel_multi_index(
        #     np.where(np.ones((num isabella steward - Tour MIT_atoms, num_atoms)) - np.eye(num_atoms)),
        #     [num_atoms, num_atoms])
        # edges = edges[:, off_diag_idx]

        # data = TensorDataset(feat, edges)
        # data_loader = DataLoader(data, batch_size=batch_size)
        # TODO: we need a way to encode the walls. Maybe add the minmax.
        return feat, edges, (loc_max, loc_min, vel_max, vel_min) # TODO: Check how mins and maxes are used

class NBADataset(data.Dataset):
    def __init__(self, args, split):
        self.args = args
        self.datadir = '/data/Armand/nba-data-large/processed/'
        suffix = '_NBA_11'
        length, sample_freq = 100, 2 # 60, 10; 100, 1
        print('NBA dataset with len{}, sf{}'.format(length, sample_freq))
        suffix += '_len{}_sf{}'.format(length, sample_freq)


        feat, _, _ = self._load_data(suffix=suffix, split=split)

        assert args.n_objects == feat.shape[1]
        self.length = feat.shape[0]
        self.timesteps = args.num_timesteps #feat.shape[2]

        self.feat, self.nothing = feat[:, :, :self.timesteps], np.ones((feat.shape[0],1))

        off_diag = np.ones([args.n_objects, args.n_objects]) - np.eye(args.n_objects)
        self.rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        self.rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing
        """
        # if self.datasource == 'default':
        #     im_corrupt = im + 0.3 * torch.randn(self.image_size, self.image_size, 3)
        # elif self.datasource == 'random':
        #     im_corrupt = 0.5 + 0.5 * torch.randn(self.image_size, self.image_size, 3)
        # else:
        #     raise NotImplementedError
        return (torch.tensor(self.feat[index]), torch.tensor(self.nothing[index])), \
               (torch.tensor(self.rel_rec), torch.tensor(self.rel_send)), index

    def __len__(self):
        """Return the total number of trajectories in the dataset."""
        return self.length

    # def _load_data(self, suffix='', split='train'):
    #     loc = np.load(self.datadir + 'loc_' + split + suffix + '.npy')[..., :2]
    #
    #     # Note: unnormalize
    #     print("Normalized in my way with Nans")
    #     loc_max = loc.max()
    #     loc_min = loc.min()
    #
    #     print("Normalized NBA Dataset")
    #     # Normalize to [-1, 1]
    #     feat = (loc - loc_min) * 2 / (loc_max - loc_min) - 1
    #     if self.args.input_dim == 4:
    #         vel = np.zeros_like(feat)
    #         vel[:, :, :-1] = feat[:, :, 1:] - feat[:, :, :-1]
    #         vel[:, :, -1] = np.nan
    #         feat = np.concatenate([feat, vel], -1)
    #
    #     # print("Unnormalized Charged Dataset")
    #
    #     # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    #     # feat = np.transpose(loc, [0, 3, 1, 2])
    #
    #
    #     feat = torch.FloatTensor(feat)
    #
    #     return feat, None, (loc_max, loc_min) # TODO: Check how mins and maxes are used

    def _load_data(self, batch_size=1, suffix='', split='train', length = 100, sample_freq=2, args=None):
        data_root = '/data/Armand/nba-data-large/processed/'

        suffix = '_NBA_11'
        # length, sample_freq = 100, 4 # 60, 10; 100, 1
        suffix += '_len{}_sf{}'.format(length, sample_freq)

        print("Normalized Dataset as in NRI")
        # Normalize to [-1, 1]
        # loc = (loc - loc_min) * 2 / (loc_max - loc_min) - 1
        if self.args.input_dim != 4:
            raise NotImplementedError


        # data_root = '/data/Armand/NRI'

        # dset = suffix[1:-1]
        # if dset == 'springs':
        #     # suffix += 'inter0.1_sf50_lentrain5000_nstrain50000'
        #     suffix += 'inter0.1_sf100_lentrain10000_nstrain50000'
        # elif dset == 'charged':
        #     # suffix += 'inter0.5_sf50_lentrain5000_nstrain50000'
        #     suffix += 'inter0.5_sf100_lentrain10000_nstrain50000'
        # elif dset == 'charged-springs':
        #     suffix += 'inter_s0.1_c0.5_sf50_lentrain5000_nstrain50000_mixedbynode'
        # else: raise NotImplementedError
        # sf = 100
        # norm_cte = 1000/sf

        loc_train = np.load(data_root + '/loc_train' + suffix + '.npy')
        vel_train = np.zeros_like(loc_train)
        vel_train[:, :, :-1] = loc_train[:, :, 1:] - loc_train[:, :, :-1]
        # vel_train = np.load(data_root + '/vel_train' + suffix + '.npy') / norm_cte
        # edges_train = np.load(data_root + '/edges_train' + suffix + '.npy')

        loc_valid = np.load(data_root + '/loc_valid' + suffix + '.npy')
        vel_valid = np.zeros_like(loc_valid)
        vel_valid[:, :, :-1] = loc_valid[:, :, 1:] - loc_valid[:, :, :-1]
        # vel_valid = np.load(data_root + '/vel_valid' + suffix + '.npy') / norm_cte
        # edges_valid = np.load(data_root + '/edges_valid' + suffix + '.npy')

        loc_test = np.load(data_root + '/loc_test' + suffix + '.npy')
        vel_test = np.zeros_like(loc_test)
        vel_test[:, :, :-1] = loc_test[:, :, 1:] - loc_test[:, :, :-1]
        # vel_test = np.load(data_root + '/vel_test' + suffix + '.npy') / norm_cte
        # edges_test = np.load(data_root + '/edges_test' + suffix + '.npy')

        # [num_samples, num_timesteps, num_dims, num_atoms]
        num_atoms = loc_train.shape[1]

        loc_max = loc_train.max()
        loc_min = loc_train.min()
        vel_max = vel_train.max()
        vel_min = vel_train.min()

        loc_max_test = loc_test.max()
        loc_min_test = loc_test.min()
        loc_max_val = loc_valid.max()
        loc_min_val = loc_valid.min()

        # Normalize to [-1, 1]
        loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
        # vel_train = vel_train * 2 / (loc_max - loc_min)
        vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

        loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
        # vel_valid = (vel_valid) * 2 / (loc_max_val - loc_min_val) - 1
        vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

        loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
        # vel_test = (vel_test) * 2 / (loc_max_test - loc_min_test) - 1
        vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

        # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]\
        #  timesteps = args.obs_length+args.pred_length
        # feat = loc[:, :, :timesteps]
        # feat = feat.transpose(0,2,1,3)


        loc_train = loc_train[:,:,:,:2]
        vel_train = vel_train[:,:,:,:2]
        feat_train = np.concatenate([loc_train, vel_train], axis=3)
        # edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
        # edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

        loc_valid = loc_valid[:,:,:,:2]
        vel_valid = vel_valid[:,:,:,:2]
        feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
        # edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
        # edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

        loc_test = loc_test[:,:,:,:2]
        vel_test = vel_test[:,:,:,:2]
        feat_test = np.concatenate([loc_test, vel_test], axis=3)
        # edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
        # edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

        # print(feat_train.shape, feat_test.shape, edges_train.shape)
        feat_train = torch.FloatTensor(feat_train[:, :, :61, :])
        # edges_train = torch.LongTensor(edges_train)
        feat_valid = torch.FloatTensor(feat_valid[:, :, :61, :])
        # edges_valid = torch.LongTensor(edges_valid)
        feat_test = torch.FloatTensor(feat_test[:, :, :61, :])
        # edges_test = torch.LongTensor(edges_test)

        # Exclude self edges
        # off_diag_idx = np.ravel_multi_index(
        #     np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        #     [num_atoms, num_atoms])
        # edges_train = edges_train[:, off_diag_idx]
        # edges_valid = edges_valid[:, off_diag_idx]
        # edges_test = edges_test[:, off_diag_idx]

        # train_edges = torch.ones(feat_train.shape[0], num_atoms*(num_atoms-1))
        # valid_edges = torch.ones(feat_valid.shape[0], num_atoms*(num_atoms-1))
        # test_edges = torch.ones(feat_test.shape[0],num_atoms**2)

        if split == 'train':
            feat = feat_train
        elif split == 'valid':
            feat = feat_valid
        elif split == 'test':
            feat = feat_test
        else: raise NotImplementedError

        return feat, None, (loc_max, loc_min)

class TrajnetDataset(data.Dataset):
    def __init__(self, args, split):
        # n_balls=5, box_size=5., loc_std=1., vel_norm=0.5,
        # interaction_strength=1., noise_var=0.):
        self.args = args
        self.n_objects = args.n_objects

        suffix = '_{}_T{}_Nobj{}'.format(args.dataset.split('_', 1)[-1], args.num_timesteps, self.n_objects)

        feat, scene_goal, stats = self._load_data(suffix=suffix, split=split)
        # TODO: loc_max, loc_min, vel_max, vel_min
        assert self.n_objects == feat.shape[1]
        self.length = feat.shape[0]
        self.timesteps = args.num_timesteps #feat.shape[2]

        # Generate off-diagonal interaction graph
        # ini_id = np.random.randint(0, feat.shape[2]-self.timesteps, (feat.shape[0],))[:, None].repeat(self.timesteps, 1)
        # batch_id = np.arange(0, feat.shape[0])[:, None].repeat(self.timesteps, 1)
        # ini_id += np.arange(0,self.timesteps)[None].repeat(feat.shape[0], 0)
        # self.feat, self.scene_goal = np.transpose(feat[batch_id, :, ini_id],(0,2,1,3)), scene_goal
        self.feat, self.scene_goal = feat[:, :, :self.timesteps], scene_goal

        off_diag = np.ones([args.n_objects, args.n_objects]) - np.eye(args.n_objects)
        self.rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        self.rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing
        """

        return (torch.tensor(self.feat[index]), torch.tensor(self.scene_goal[index])), \
               (torch.tensor(self.rel_rec), torch.tensor(self.rel_send)), index

    def __len__(self):
        """Return the total number of trajectories in the dataset."""
        return self.length

    def _load_data(self, suffix='', split='train'):
        if split == 'test': split = 'val'
        scene = np.load('/data/Armand/TrajNet/scene_' + split + suffix + '.npy')
        scene_goal = np.load('/data/Armand/TrajNet/scene_goal_' + split + suffix + '.npy')
        # split = ...

        # [num_samples, num_timesteps, num_dims, num_atoms]
        num_atoms = self.n_objects

        # Note: unnormalize
        nonans = scene == scene
        # scene[nans] = 0 # TODO: Is that correct?
        loc_max = scene[nonans].max()
        loc_min = scene[nonans].min()

        print("Normalized Charged Dataset")
        # Normalize to [-1, 1]
        scene = (scene - loc_min) * 2 / (loc_max - loc_min) - 1

        # print("Unnormalized Charged Dataset")

        # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
        scene = np.transpose(scene, [0, 2, 1, 3])
        scene = torch.FloatTensor(scene)

        # data = TensorDataset(feat, edges)
        # data_loader = DataLoader(data, batch_size=batch_size)
        # TODO: we need a way to encode the walls. Maybe add the minmax.
        return scene, scene_goal, (loc_max, loc_min) # TODO: Check how mins and maxes are used



if __name__ == "__main__":
    exit()
    # for data in loader:
    #     print("here")
