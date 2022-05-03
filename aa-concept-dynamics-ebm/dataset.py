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
        suffix += 'inter0.1_nowalls_sf100_len5000'
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

        # ini_id = np.random.randint(0, feat.shape[2]-self.timesteps, (feat.shape[0],))[:, None].repeat(self.timesteps, 1)
        # batch_id = np.arange(0, feat.shape[0])[:, None].repeat(self.timesteps, 1)
        # ini_id += np.arange(0,self.timesteps)[None].repeat(feat.shape[0], 0)
        # self.feat, self.edges = np.transpose(feat[batch_id, :, ini_id],(0,2,1,3)), edges
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
        # suffix += '_nobox_05int-strength'
        suffix += 'inter0.5_nowalls_sf100_len5000'
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

        # ini_id = np.random.randint(0, feat.shape[2]-self.timesteps, (feat.shape[0],))[:, None].repeat(self.timesteps, 1)
        # batch_id = np.arange(0, feat.shape[0])[:, None].repeat(self.timesteps, 1)
        # ini_id += np.arange(0,self.timesteps)[None].repeat(feat.shape[0], 0)
        # self.feat, self.edges = np.transpose(feat[batch_id, :, ini_id],(0,2,1,3)), edges
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

class ChargedSpringsParticles(data.Dataset):
    def __init__(self, args, split):
        # n_balls=5, box_size=5., loc_std=1., vel_norm=0.5,
        # interaction_strength=1., noise_var=0.):
        self.args = args
        self.n_objects = args.n_objects

        suffix = '_charged-springs'+str(self.n_objects)
        # suffix += '_nobox_05int-strength'
        suffix += 'inter0.5_nowalls_sf100_len5000_test-mixed'

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
