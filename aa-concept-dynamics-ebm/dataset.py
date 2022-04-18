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
    def __init__(self, args):
                 # n_balls=5, box_size=5., loc_std=1., vel_norm=0.5,
                 # interaction_strength=1., noise_var=0.):
        self.args = args

        self.n_objects = args.n_objects
        self.box_size = 5.
        self.loc_std = 1.
        self.vel_norm = 0.5
        self.interaction_strength = args.interaction_strength
        self.noise_var = args.noise_var
        self.timesteps = 49

        self.sample_freq = args.sample_freq # default: 100
        self.sequence_length = args.sequence_length # default: 1000

        self._charge_types = np.array([-1., 0., 1.])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

        # Generate off-diagonal interaction graph
        off_diag = np.ones([args.n_objects, args.n_objects]) - np.eye(args.n_objects)

        self.rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        self.rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        # self.rel_rec = torch.FloatTensor(rel_rec)
        # self.rel_send = torch.FloatTensor(rel_send)

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def _energy(self, loc, vel, edges):

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[
                            i, j] / dist
            return U + K

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def sample_trajectory(self, T=10000, sample_freq=10,
                          charge_prob=[1. / 2, 0, 1. / 2]):
        n = self.n_objects
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        # Sample edges
        charges = np.random.choice(self._charge_types, size=(self.n_objects, 1),
                                   p=charge_prob)
        edges = charges.dot(charges.transpose())
        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        loc_next = np.random.randn(2, n) * self.loc_std
        vel_next = np.random.randn(2, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):
            # half step leapfrog
            l2_dist_power3 = np.power(
                self._l2(loc_next.transpose(), loc_next.transpose()), 3. / 2.)

            # size of forces up to a 1/|r| factor
            # since I later multiply by an unnormalized r vector
            forces_size = self.interaction_strength * edges / l2_dist_power3
            np.fill_diagonal(forces_size,
                             0)  # self forces are zero (fixes division by zero)
            assert (np.abs(forces_size[diag_mask]).min() > 1e-10)
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                l2_dist_power3 = np.power(
                    self._l2(loc_next.transpose(), loc_next.transpose()),
                    3. / 2.)
                forces_size = self.interaction_strength * edges / l2_dist_power3
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n,
                                                                   n)))).sum(
                    axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_objects) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_objects) * self.noise_var

            # Note: Normalize and prepare data
            num_atoms = loc.shape[2]

            loc_max = loc.max()
            loc_min = loc.min()
            vel_max = vel.max()
            vel_min = vel.min()

            # Normalize to [-1, 1]
            loc = (loc - loc_min) * 2 / (loc_max - loc_min) - 1
            vel = (vel - vel_min) * 2 / (vel_max - vel_min) - 1

            # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
            loc = np.transpose(loc, [2, 0, 1])
            vel = np.transpose(vel, [2, 0, 1])
            feat = np.concatenate([loc, vel], axis=-1)
            edges = np.reshape(edges, [-1, num_atoms ** 2])
            edges = np.array((edges + 1) / 2, dtype=np.int64)

            # Exclude self edges
            off_diag_idx = np.ravel_multi_index(
                np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
                [num_atoms, num_atoms])
            edges = edges[:, off_diag_idx]

            return feat, edges

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing
        """

        # args = self.args
        feat, edges = self.sample_trajectory(T=self.sequence_length, sample_freq=self.sample_freq)
        # TODO: Might have to use other restrictions for sampling in test
        return (torch.Tensor(feat), torch.Tensor(edges)), \
               (torch.Tensor(self.rel_rec), torch.Tensor(self.rel_send)), index

    def __len__(self):
        """Return the total number of trajectories in the dataset."""
        return 100000 # Hardcoded

class SpringsParticles(data.Dataset):
    def __init__(self, args, split):
        # n_balls=5, box_size=5., loc_std=1., vel_norm=0.5,
        # interaction_strength=1., noise_var=0.):
        self.args = args
        self.n_objects = args.n_objects
        suffix = '_springs'+str(self.n_objects)
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
        print("Normalized Charged Dataset")
        loc_max = loc.max()
        loc_min = loc.min()
        vel = vel / 10
        # Note: In simulation our increase in T (delta T) is 0.001.
        #  Then we sample 1/10 generated samples.
        #  Therefore the ratio between loc and velocity is vel/(incrLoc) = 10
        vel_max = vel.max()
        vel_min = vel.min()

        # Normalize to [-1, 1]
        loc = (loc - loc_min) * 2 / (loc_max - loc_min) - 1
        # vel = (vel - vel_min) * 2 / (vel_max - vel_min) - 1
        vel = vel * 2 / (loc_max - loc_min)

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

        suffix = '_charged'+str(self.n_objects)
        suffix += '_nobox_05int-strength'
        # suffix += 'inter0.5_nowalls_sf100_len5000'
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
        vel = vel / 10
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


### DATASET LOADER FROM FQA
import pandas as pd
import errno
import pickle

# import sys
# sys.path.append('./')

def get_files(dirpath):
    return [name for name in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, name))]

def get_dirs(dirpath):
    return [name for name in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, name))]

class TrajectoryDataset(data.Dataset):
    def __init__(self, args, dset_name, split,
                 max_length=20,
                 min_length=10,
                 to_tensor=True,
                 random_rotation=False,
                 burn_in_steps=8,
                 skip_frames=0,
                 frame_shift=1,
                 no_filter=False,
                 use_pickled=True):

        data_dir = '/data/Armand/FQA/'
        if dset_name == 'collisions':
            data_dir = os.path.join(data_dir, 'collisions')
        elif dset_name == 'ethucy':
            data_dir = os.path.join(data_dir, 'ethucy')
        else: raise NotImplementedError

        dir = os.path.join(data_dir, split)
        data_files = [os.path.join(dirname, filename) for dirname in get_dirs(dir) for filename in get_files(os.path.join(dir, dirname))]
        data_files = [os.path.join(dir, filepath) for filepath in data_files]

        # Max and min length of consecutive sequence
        self.max_length = max_length
        self.min_length = min_length
        self.burn_in_steps = burn_in_steps
        self.skip_frames = skip_frames
        self.frame_shift = frame_shift
        self.no_filter = no_filter

        # If true return torch.Tensor, or otherwise return np.array
        self.to_tensor = to_tensor

        # If true, apply random rotation (around origin) to data points
        self.random_rotation = random_rotation

        # List containing raw data for each scene
        self.scenes = {}

        # List of dictionary containing:
        #   "scene_id": (int)
        #   "start": (int)
        #   "interval": (int)
        self.frame_seqs = []

        # Load data from files
        self.load_data(data_files, use_pickled)

    def load_data(self, data_files, use_pickled=False):

        # Each data file contains raw data from a single scene
        for scene_file in data_files:
            pickle_path = self.get_pickle_path(scene_file)

            if use_pickled and os.path.exists(pickle_path):
                scene = _load_data(scene_file)
                frame_seqs = _load_pickle(pickle_path)
            else:
                scene, frame_seqs = self.load_from_file(scene_file)
                _dump_pickle(frame_seqs, pickle_path)

            self.frame_seqs += frame_seqs
            self.scenes[scene_file] = scene


    def get_pickle_path(self, scene_file):
        """ Return pickle path for the scene_file """
        dirname = os.path.dirname(scene_file)
        filename = os.path.splitext(os.path.basename(scene_file))[0]

        tokens = [filename, str(self.max_length), str(self.min_length), str(self.skip_frames)]
        if self.burn_in_steps > 0:
            tokens.append(str(self.burn_in_steps))

        if self.frame_shift > 1:
            tokens.append('fs'+str(self.frame_shift))

        pickle_name = '-'.join(tokens) + '.pkl'
        pickle_path = os.path.join(dirname, 'pickles', pickle_name)

        return pickle_path


    def load_from_file(self, scene_file):
        """ Processes and load data from scene_file
        Args:
            scene_file - Path to the file to be loaded
        Returns:
            scene - Pandas DataFrame containing data from scene_file
            frame_seqs - List containing info about frame sequences
        """
        scene = _load_data(scene_file)
        interval = _get_frame_interval(scene)
        frame_ids = _get_frame_ids(scene)
        frame_seqs = []

        if self.skip_frames > 0:
            interval = self.skip_frames

        # For each frame in a scene,
        # check if there are at least min_length frames
        # starting from the current frame.
        for frame in frame_ids[::self.frame_shift]:
            start = int(frame)
            end = start + (self.max_length * interval)

            # get data sequence between the "start" and "end" frames (inclusive)
            data_seq = _get_data_sequence_between(scene, start, end, interval)
            num_frames = len(_get_frame_ids(data_seq))

            # Check if there are at least min_length frames
            if num_frames >= self.min_length + 1: # 1 extra frame checked for to incorporate both data, preds
                # add the info about frame sequence to frame_seqs
                frame_seqs.append({
                    "scene_id": scene_file,
                    "start": start,
                    "interval": interval
                })

        return scene, frame_seqs

    def __getitem__(self, index):
        """
        N : number of agents
        L : max sequence length
        D : dimension of spatial trajectory coordinate
        Return:
            source - tensor of shape (N, L, D)
            target - tensor of shape (N, L, D)
            mask - tensor of shape (N, L)
        """
        # Get frame sequence and scene
        frame_seq = self.frame_seqs[index]
        scene = self.scenes[frame_seq['scene_id']]
        interval = frame_seq['interval']

        start = frame_seq['start']
        end = start + (self.max_length * interval)

        # Get data sequence of length (max_length + 1) from the scene
        data_seq = _get_data_sequence_between(scene, start, end, interval)

        # Convert raw data sequence to arrays
        data, mask = _to_array(data_seq, self.max_length + 1, start, interval)

        # Filtering on agents
        if (not self.no_filter):
            m = mask[:,:,0]
            # Remove agents present for less than half the trajectory
            m[np.sum(m, axis=1) / m.shape[1] < 1/2] = 0

            # Remove agents not present at time == br - 3
            if self.burn_in_steps >= 3:
                m[m[:,self.burn_in_steps-3] == 0] = 0
            mask = np.tile(np.expand_dims(m,2), (1,1,2))

        # Get source and target data and masks
        source = data[:, 0:self.max_length, :]
        target = data[:, 1:self.max_length+1, :]
        source_mask = mask[:, 0:self.max_length, :]
        target_mask = mask[:, 1:self.max_length+1, :]

        if (self.random_rotation):
            theta = 2 * np.pi * np.random.random()
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c,-s), (s, c)))
            source = np.matmul(source, R)
            target = np.matmul(target, R)

        if (self.to_tensor):
            source = torch.Tensor(source)
            target = torch.Tensor(target)
            source_mask = torch.Tensor(source_mask)
            target_mask = torch.Tensor(target_mask)

        return source, target, source_mask, target_mask, index

        # (torch.tensor(self.feat[index]), torch.tensor(self.edges[index])), \
        # (torch.tensor(self.rel_rec), torch.tensor(self.rel_send)), index
        # off_diag = np.ones([args.n_objects, args.n_objects]) - np.eye(args.n_objects)
        # self.rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        # self.rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)

        ### LOADER ###
        # elif FLAGS.dataset == 'collisions' or FLAGS.dataset == 'ethucy':
        # dataset = TrajectoryDataset(FLAGS, FLAGS.dataset, 'train') # Note: May add other arguments
        # test_dataset = TrajectoryDataset(FLAGS, FLAGS.dataset, 'test')
    def __len__(self):
        return len(self.frame_seqs)

# Helper functions
def _load_data(data_file):
    """ Read data from a file and returns a pd.DataFrame """
    df = pd.read_csv(data_file, sep="\s+", header=None)
    df.columns = ["frame_id", "agent_id", "x", "y"]
    return df

def _load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def _dump_pickle(data, path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except:
            if exc.errno != errno.EEXIST:
                raise
    with open(path, 'wb') as f:
        pickle.dump(data, f)

#def _get_data_sequence_between(df, start, end):
#    """ Returns all data with frame_id between start and end (inclusive) """
#    return df.loc[df.frame_id.between(start, end)]

def _get_data_sequence_between(df, start, end, interval):
    """ Returns all data with frame_id between start and end (inclusive) """
    data_seq = df.loc[df.frame_id.between(start, end)]
    data_seq = data_seq[(data_seq.frame_id - start) % interval == 0]
    return data_seq

def _get_frame_interval(df):
    """ Calculate frame interval of the DataFrame df.
        Assumes that the first two frame_ids are consecutive
    """
    frame_ids = sorted(list(_get_frame_ids(df)))
    return int(frame_ids[1] - frame_ids[0])

def _get_frame_ids(df):
    """ Returns unique frame_ids in the DataFrame df """
    return df.frame_id.unique()

def _get_agent_ids(df):
    """ Returns unique agent_ids in df """
    return df.agent_id.unique()

def _to_array(df, max_length, start, interval, dim=2):
    """ Convert input DataFrame df to 3-dimensional Numpy array"""
    agents = sorted(list(_get_agent_ids(df)))
    num_agents = len(agents)

    # First create an array of shape (N, L, D) filled with inf.
    # We will replace infs with 0 after we compute mask
    array = np.full((num_agents, max_length, dim), np.inf)

    # Compute indexs to fill out
    agent_idxs = (df.agent_id.apply(agents.index)).astype(int)
    frame_idxs = ((df.frame_id - start) / interval).astype(int)
    coords = df[['x', 'y']]
    assert len(coords) == len(frame_idxs) and len(coords) == len(agent_idxs)

    # Fill out arrays with coordinates
    array[agent_idxs, frame_idxs] = coords

    # Compute mask where mask[i, j] == 0 for array[i, j] == inf
    mask = (array != np.inf).astype(int)

    # Finally replace inf with 0
    array[array == np.inf] = 0

    return array, mask

if __name__ == "__main__":
    exit()
    # for data in loader:
    #     print("here")
