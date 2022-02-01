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

        feat, edges, stats = self._load_data(suffix='_springs'+str(self.n_objects), split=split)
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
        print("Unnormalized Spring Dataset")
        # loc_max = loc.max()
        # loc_min = loc.min()
        # vel_max = vel.max()
        # vel_min = vel.min()
        #b
        # # Normalize to [-1, 1]
        # loc = (loc - loc_min) * 2 / (loc_max - loc_min) - 1
        # vel = (vel - vel_min) * 2 / (vel_max - vel_min) - 1
        loc_max = None
        loc_min = None
        vel_max = None
        vel_min = None

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

        feat, edges, stats = self._load_data(suffix='_charged'+str(self.n_objects), split=split)
        # TODO: loc_max, loc_min, vel_max, vel_min
        assert self.n_objects == feat.shape[1]
        self.length = feat.shape[0]
        self.timesteps = 49 #feat.shape[2]

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

        loc_max = loc.max()
        loc_min = loc.min()
        vel_max = vel.max()
        vel_min = vel.min()

        # Normalize to [-1, 1]
        loc = (loc - loc_min) * 2 / (loc_max - loc_min) - 1
        vel = (vel - vel_min) * 2 / (vel_max - vel_min) - 1

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

### Note: Image datasets.
class IntPhysDataset(data.Dataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, args):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        train_path = "/private/home/yilundu/dataset/intphys"
        # train_path = "/data/vision/billf/scratch/jerrymei/newIntPhys/render/output/train_v7"
        # random.seed(rank_idx)

        p = train_path

        dirs = os.listdir(p)
        files = []
        depth_files = []

        for d in dirs:
            base_path = osp.join(p, d, 'imgs')
            ims = os.listdir(base_path)
            ims = sorted(ims)
            ims = ims

            im_paths = [osp.join(base_path, im) for im in ims]
            files.append(im_paths)

        self.args = args
        self.A_paths = files
        self.D_paths = depth_files
        self.frames = 2
        self.im_size = args.im_size
        self.temporal = args.temporal


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        args = self.args

        if args.single:
            index = 0

        index = index % len(self.A_paths)

        A_path = self.A_paths[index]
        ix = random.randint(0, len(A_path) - 20)

        ix_next = ix + random.randint(0, 19)

        im = imread(A_path[ix])[:, :, :3]
        im_next = imread(A_path[ix_next])[:, :, :3]

        im = imresize(im, (64, 64))[:, :, :3]
        im_next = imresize(im_next, (64, 64))[:, :, :3]

        im = torch.Tensor(im).permute(2, 0, 1)
        im_next = torch.Tensor(im_next).permute(2, 0, 1)

        if self.temporal:
            im = torch.stack([im, im_next], dim=0)

        return im, index


    def __len__(self):
        """Return the total number of images in the dataset."""
        return 1000000

class ToyDataset(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.components = 3

        # self.nsample = 10000
        self.samples = []



    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        FLAGS = self.opt
        A = np.zeros((3, 64, 64))
        dot_size = 5

        intersects = []

        for i in range(self.components):
            while True:
                x, y = random.randint(dot_size, 64 - dot_size), random.randint(dot_size, 64 - dot_size)

                valid = True
                for xi, yi in intersects:
                    if (abs(x - xi) < 2 * dot_size)  and (abs(y - yi) < 2 * dot_size):
                        valid = False
                        break

                if valid:
                    A[i, x-dot_size:x+dot_size, y-dot_size:y+dot_size] = 0.8
                    intersects.append((x, y))
                    break

        A = torch.Tensor(A)

        return A, index

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return 1000000

# class Blender(data.Dataset):
#     def __init__(self, stage=0):
#         self.path = "/private/home/yilundu/sandbox/data/blender/continual/test_rotation"
# 
#     def __len__(self):
#         return 10000
# 
#     def __getitem__(self, index):
#         im = imread(osp.join(self.path, "r_{}.png".format(index)))
#         im = imresize(im, (64, 64))[:, :, :3]
#         im = im / 255.
# 
#         im = torch.Tensor(im).permute(2, 0, 1)
# 
#         return im, index

# class Blender(data.Dataset):
#     def __init__(self, stage=0):
#         self.path = "/private/home/yilundu/sandbox/data/CLEVR_v1.0/images/train"
#         self.images = glob.glob(self.path + "/*.png")
# 
#     def __len__(self):
#         return len(self.images)
# 
#     def __getitem__(self, index):
#         im_path = self.images[index]
#         im = imread(im_path)
#         im = imresize(im, (64, 64))[:, :, :3]
#         im = im / 255.
# 
#         im = torch.Tensor(im).permute(2, 0, 1)
# 
#         return im, index

class Blender(data.Dataset):
    def __init__(self, stage=0):
        self.path = "/private/home/yilundu/dataset/shop_vrb/images/train"
        self.images = glob.glob(self.path + "/*.png")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im_path = self.images[index]
        im = imread(im_path)
        im = imresize(im, (64, 64))[:, :, :3]
        im = im / 255.

        im = torch.Tensor(im).permute(2, 0, 1)

        return im, index

class Cub(data.Dataset):
    def __init__(self, stage=0):
        self.path = "/private/home/yilundu/sandbox/data/CUB/images/*/*.jpg"
        self.images = glob.glob(self.path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im_path = self.images[index]
        im = imread(im_path)
        im = imresize(im, (64, 64))[:, :, :3]
        im = im / 255.

        im = torch.Tensor(im).permute(2, 0, 1)

        return im, index

class Nvidia(data.Dataset):
    def __init__(self, stage=0, filter_light=False):
        self.path = "/data/vision/billf/scratch/yilundu/dataset/disentanglement/Falcor3D_down128/images/{:06}.png"
        self.labels = np.load("/data/vision/billf/scratch/yilundu/dataset/disentanglement/Falcor3D_down128/train-rec.labels")
        label_mask = (self.labels[:, 0] > 0) & (self.labels[:, 0] < 1)
        idxs = np.arange(self.labels.shape[0])

        self.filter_light = filter_light

        # if self.filter_light:
        #     self.idxs = idxs[label_mask]
        # else:
        self.idxs = idxs

    def __len__(self):
        return self.idxs.shape[0]

    def __getitem__(self, index):
        index = self.idxs[index]
        im_path = self.path.format(index)
        # im_path = self.images[index]
        im = imread(im_path)
        im = imresize(im, (128, 128))[:, :, :3][:, :, ::-1].copy()
        im = im

        im = torch.Tensor(im).permute(2, 0, 1)

        return im, index

class NvidiaDisentangle(data.Dataset):
    def __init__(self, stage=0, filter_light=True):
        self.path = "/data/vision/billf/scratch/yilundu/dataset/disentanglement/Falcor3D_down128/images/{:06}.png"
        self.labels = np.load("/data/vision/billf/scratch/yilundu/dataset/disentanglement/Falcor3D_down128/train-rec.labels")
        idxs = np.arange(self.labels.shape[0])
        self.idxs = idxs

    def __len__(self):
        return self.idxs.shape[0]

    def __getitem__(self, index):
        index = self.idxs[index]
        im_path = self.path.format(index)
        # im_path = self.images[index]
        im = imread(im_path)
        im = imresize(im, (128, 128))[:, :, :3]
        im = im

        # label = int(self.labels[index, 0] * 5)
        label = self.labels[index, 1:]
        im = torch.Tensor(im).permute(2, 0, 1)

        return im, label

class Clevr(data.Dataset):
    def __init__(self, stage=0):
        self.path = "/data/Armand/clevr/images_clevr/*.png"
        self.images = sorted(glob(self.path))
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im_path = self.images[index]
        im = imread(im_path)
        im = imresize(im, (64, 64))[:, :, :3]

        im = torch.Tensor(im).permute(2, 0, 1)

        return im, index

class ClevrLighting(data.Dataset):
    def __init__(self, stage=0):
        self.path = "/data/vision/billf/scratch/yilundu/dataset/clevr_lighting/images_large_lighting/*.png"
        self.images = sorted(glob(self.path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im_path = self.images[index]
        im = imread(im_path)
        im = imresize(im, (64, 64))[:, :, :3]

        im = torch.Tensor(im).permute(2, 0, 1)

        return im, index

class Exercise(data.Dataset):
    def __init__(self, args):
        self.temporal = args.temporal
        self.path = "/private/home/yilundu/sandbox/data/release_data_set/images/*_im1.png"
        self.images = glob(self.path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        try:
            im_path = self.images[index]
            im_path_next = im_path.replace("im1", "im2")
            im = imread(im_path)
            im_next = imread(im_path_next)
            im = imresize(im, (64, 64))[:, :, :3]
            im_next = imresize(im_next, (64, 64))[:, :, :3]

            im = torch.Tensor(im).permute(2, 0, 1)
            im_next = torch.Tensor(im_next).permute(2, 0, 1)

            if self.temporal:
                im = torch.stack([im, im_next], dim=0)

            return im, index
        except:
            return self.__getitem__((index + 1) % len(self.images))

class CelebaHQ(data.Dataset):
    def __init__(self, resolution=64):
        self.name = 'celebahq'
        self.channels = 3
        self.paths = glob("/data/vision/billf/scratch/yilundu/dataset/celebahq/data128x128/*.jpg")
        self.resolution = resolution

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        frame = imread(path)
        frame = imresize(frame, (self.resolution, self.resolution))[:, :, :3]

        im = torch.Tensor(frame).permute(2, 0, 1)

        return im, index

class Airplane(data.Dataset):
    def __init__(self, stage=0):
        # self.path = "/private/home/yilundu/sandbox/video_ebm/dataset/images/*.png"
        self.name = 'celebahq'
        self.channels = 3
        # self.path = "/datasets01/celebAHQ/081318/imgHQ{:05}.npy"
        self.paths = glob("/data/vision/billf/scratch/yilundu/nerf-pytorch/large_render/*.png")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        frame = imread(path)
        frame = imresize(frame, (64, 64))[:, :, :3]

        im = torch.Tensor(frame).permute(2, 0, 1)

        return im, index

class Anime(data.Dataset):
    def __init__(self, stage=0):
        # self.path = "/private/home/yilundu/sandbox/video_ebm/dataset/images/*.png"
        self.name = 'celebahq'
        self.channels = 3
        # self.path = "/datasets01/celebAHQ/081318/imgHQ{:05}.npy"
        anime_paths = sorted(glob("/data/vision/billf/scratch/yilundu/dataset/anime/cropped/*.jpg"))[:30000]
        self.paths = anime_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        try:
            path = self.paths[index]
            frame = imread(path)
            frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)

            im = torch.Tensor(frame).permute(2, 0, 1) / 255.

            return im, index
        except:
            ix = random.randint(0, len(self.paths) - 1)
            return self.__getitem__(ix)

class Faces(data.Dataset):
    def __init__(self, stage=0):
        # self.path = "/private/home/yilundu/sandbox/video_ebm/dataset/images/*.png"
        self.name = 'celebahq'
        self.channels = 3
        # self.path = "/datasets01/celebAHQ/081318/imgHQ{:05}.npy"
        paths = sorted(glob("/data/vision/billf/scratch/yilundu/dataset/celebahq/data128x128/*.jpg"))
        anime_paths = sorted(glob("/data/vision/billf/scratch/yilundu/dataset/anime/cropped/*.jpg"))[:30000]
        paths = list(paths) + anime_paths
        random.shuffle(paths)
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        try:
            path = self.paths[index]
            frame = imread(path)
            frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)

            im = torch.Tensor(frame).permute(2, 0, 1) / 255.

            return im, index
        except:
            ix = random.randint(0, len(self.paths) - 1)
            return self.__getitem__(ix)

class DSprites(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.components = opt.components
        self.data = np.load("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")['imgs']
        self.n = self.data.shape[0]

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        A = np.zeros((3, 64, 64))
        ix = random.randint(0, self.n-1)
        im = self.data[ix]
        for i in range(3):
            A[i] = im

        A = torch.Tensor(A)

        return A, index

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return 100000

class MultiDspritesLoader():

    def __init__(self, batchsize):
        tf_records_path = 'dataset/multi_dsprites_colored_on_colored.tfrecords'
        batch_size = batchsize

        dataset = multi_dsprites.dataset(tf_records_path, 'colored_on_colored')
        batched_dataset = dataset.batch(batch_size)  # optional batching
        iterator = batched_dataset.make_one_shot_iterator()
        self.data = iterator.get_next()
        self.sess = tf.InteractiveSession()

    def __iter__(self):
        return self

    def __next__(self):
        d = self.sess.run(self.data)
        img = d['image']
        img = img.transpose((0, 3, 1, 2))
        img = img / 255.
        img = torch.Tensor(img)

        return img, torch.ones(1)

    def __len__(self):
        return 1e6


class TetrominoesLoader():

    def __init__(self, batchsize):
        # tf_records_path = '/home/yilundu/my_repos/dataset/tetrominoes_train.tfrecords'
        tf_records_path = '/home/gridsan/yilundu/my_files/ebm_video/dataset/tetrominoes_train.tfrecords'
        batch_size = batchsize

        dataset = tetrominoes.dataset(tf_records_path)
        batched_dataset = dataset.batch(batch_size)  # optional batching
        iterator = batched_dataset.make_one_shot_iterator()
        self.data = iterator.get_next()
        config = tf.ConfigProto(
                device_count = {'GPU': 0}
            )
        self.sess = tf.InteractiveSession(config=config)

    def __iter__(self):
        return self

    def __next__(self):
        d = self.sess.run(self.data)
        img = d['image']
        img = img.transpose((0, 3, 1, 2))
        img = img / 255.
        img = torch.Tensor(img).contiguous()

        return img, torch.ones(1)

    def __len__(self):
        return 1e6

class TFImagenetLoader(data.Dataset):

    def __init__(self, split, batchsize, idx, num_workers, return_label=False):
        IMAGENET_NUM_TRAIN_IMAGES = 1281167
        IMAGENET_NUM_VAL_IMAGES = 50000
        self.return_label = return_label

        if split == "train":
            im_length = IMAGENET_NUM_TRAIN_IMAGES
        else:
            im_length = IMAGENET_NUM_VAL_IMAGES

        self.curr_sample = 0

        index_path = osp.join('/data/vision/billf/scratch/yilundu/imagenet', 'index.json')
        with open(index_path) as f:
            metadata = json.load(f)
            counts = metadata['record_counts']

        if split == 'train':
            files = list(sorted([x for x in counts.keys() if x.startswith('train')]))
        else:
            files = list(sorted([x for x in counts.keys() if x.startswith('validation')]))

        files = [osp.join('/data/vision/billf/scratch/yilundu/imagenet', x) for x in files]
        preprocess_function = ImagenetPreprocessor(224, dtype=tf.float32, train=False).parse_and_preprocess

        ds = tf.data.TFRecordDataset.from_generator(lambda: files, output_types=tf.string)
        ds = ds.apply(tf.data.TFRecordDataset)
        ds = ds.take(im_length)
        # ds = ds.prefetch(buffer_size=4)
        ds = ds.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000))
        ds = ds.apply(batching.map_and_batch(map_func=preprocess_function, batch_size=batchsize, num_parallel_batches=4))
        ds = ds.prefetch(buffer_size=2)

        ds_iterator = ds.make_initializable_iterator()
        labels, images = ds_iterator.get_next()
        self.images = tf.clip_by_value(images / 256 + tf.random_uniform(tf.shape(images), 0, 1. / 256), 0.0, 1.0)
        self.labels = labels

        config = tf.ConfigProto(device_count = {'GPU': 0})
        sess = tf.Session(config=config)
        sess.run(ds_iterator.initializer)

        # self.im_length = im_length // batchsize
        self.im_length = im_length

        self.sess = sess

    def __next__(self):
        self.curr_sample += 1

        sess = self.sess

        label, im = sess.run([self.labels, self.images])
        label = label.squeeze() - 1
        im = torch.from_numpy(im).permute((0, 3, 1, 2))
        label = torch.LongTensor(label)

        if self.return_label:
            return im, label
        else:
            return im[:, None, :]

    def __iter__(self):
        return self

    def __len__(self):
        return self.im_length

class TFTaskAdaptation(data.Dataset):

    def __init__(self, split, batchsize):
        data_params = {
            # "dataset": "data." + "clevr(task='count_all')",
            "dataset": "data." + "svhn()",
            "dataset_train_split_name": "trainval",
            "dataset_eval_split_name": "test",
            "shuffle_buffer_size": 10000,
            "prefetch": True,
            "train_examples": None,
            "batch_size": batchsize,
            "batch_size_eval": batchsize,
            "data_for_eval": split == "test",
            "data_dir": "/private/home/yilundu/tensorflow_datasets",
            "input_range": [0.0, 1.0]
        }
        ds = build_data_pipeline(data_params, split)
        ds = ds({'batch_size': batchsize})
        # ds = ds.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000))
        # ds = ds.apply(batching.map_and_batch(map_func=preprocess_function, batch_size=FLAGS.batch_size, num_parallel_batches=4))
        # ds = ds.prefetch(buffer_size=2)

        ds_iterator = tf.data.make_initializable_iterator(ds)
        outputs = ds_iterator.get_next()
        image, label = outputs['image'], outputs['label']
        self.images = image
        self.labels = label
        self.split = split

        config = tf.ConfigProto(device_count = {'GPU': 0})
        sess = tf.Session(config=config)
        sess.run(ds_iterator.initializer)

        self.im_length = 1000
        self.curr_sample = 0

        self.sess = sess

    def __next__(self):
        self.curr_sample += 1

        sess = self.sess
        label, im = sess.run([self.labels, self.images])

        if self.split == "train":
            im = im[:, 0].transpose((0, 1, 4, 2, 3))
        else:
            im = im.transpose((0, 3, 1, 2))

        if self.curr_sample == 1000:
            self.curr_sample = 0
            raise StopIteration

        return [torch.Tensor(im[:]), torch.Tensor(label).long()]

    def __iter__(self):
        return self

    def __len__(self):
        return self.im_length

class CubesColor(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, opt, return_label=False, train=True):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.data = np.load("/private/home/yilundu/dataset/cubes_varied_position_812.npz")
        self.ims = np.array(self.data['ims'])
        self.labels = np.array(self.data['labels'])
        self.return_label = return_label
        self.opt = opt

        n = self.ims.shape[0]
        split_idx = int(0.9 * n)

        if train:
            self.ims = self.ims[:split_idx]
            self.labels = self.labels[:split_idx]
        else:
            self.ims = self.ims[split_idx:]
            self.labels = self.labels[split_idx:]


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        opt = self.opt

        im = np.array(self.ims[index])
        label = torch.FloatTensor(np.array(self.labels[index]))
        im = imresize(im, (opt.im_size, opt.im_size))
        s = im.shape
        im = im.transpose((2, 0, 1)) / 256 + np.random.uniform(0, 1, (s[2], s[0], s[1])) / 256
        im = torch.FloatTensor(im[None, :])

        if self.return_label:
            return im, label
        else:
            return im

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return self.ims.shape[0]

class CubesColorPair(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, opt, return_label=False, train=True):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.data = np.load("/private/home/yilundu/dataset/cubes_varied_multi_311.npz")
        self.ims = np.array(self.data['ims'])
        self.labels = np.array(self.data['labels'])
        self.return_label = return_label
        self.opt = opt

        n = self.ims.shape[0]
        split_idx = int(0.9 * n)

        if train:
            self.ims = self.ims[:split_idx]
            self.labels = self.labels[:split_idx]
        else:
            self.ims = self.ims[split_idx:]
            self.labels = self.labels[split_idx:]


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        opt = self.opt

        im = self.ims[index]
        label = np.array(self.labels[index])
        im = imresize(im, (opt.im_size, opt.im_size))
        s = im.shape
        im = im.transpose((2, 0, 1)) / 256 + np.random.uniform(0, 1, (s[2], s[0], s[1])) / 256
        im = torch.Tensor(im[None, :])

        if self.return_label:
            return im, label
        else:
            return im

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return self.ims.shape[0]

class Kitti(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, opt, return_label=False, train=True):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        ims = sorted(glob("/data/vision/billf/scratch/yilundu/dataset/kitti/training/image_02/*/*.png"))
        virtual_ims = sorted(glob("/data/vision/billf/scratch/yilundu/dataset/kitti/virtual_kitti/*/*/frames/rgb/Camera_0/*.jpg"))

        ims = ims * 3 + virtual_ims
        self.ims = ims
        self.opt = opt

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        # 433 - 808
        im = self.ims[index]
        im = imread(im)
        im = im[:, 433:808, :]
        im = imresize(im, (64, 64))[:, :, :3]

        im = torch.Tensor(im).permute(2, 0, 1)

        return im, index

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return len(self.ims)

class VirtualKitti(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self,return_label=False, train=True):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        ims = sorted(glob("/data/vision/billf/scratch/yilundu/dataset/kitti/training/image_02/*/*.png"))
        virtual_ims = sorted(glob("/data/vision/billf/scratch/yilundu/dataset/kitti/virtual_kitti/*/*/frames/rgb/Camera_0/*.jpg"))

        self.ims = virtual_ims

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        # 433 - 808
        im = self.ims[index]
        im = imread(im)
        im = im[:, 433:808, :]
        im = imresize(im, (64, 64))[:, :, :3]

        im = torch.Tensor(im).permute(2, 0, 1)

        return im, index

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return len(self.ims)

class KittiLabel(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self,return_label=False, train=True):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        virtual_ims = sorted(glob("/data/vision/billf/scratch/yilundu/dataset/kitti/virtual_kitti/*/*/frames/rgb/Camera_0/*.jpg"))
        self.labels = ['fog', 'morning', 'overcast', 'rain', 'sunset']
        self.ims = virtual_ims

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        # 433 - 808
        try:
            im_path = self.ims[index]
            label_id = im_path.split("/")[-5]
            label_id = self.labels.index(label_id)

            im = imread(im_path)

            im = im[:, 433:808, :]
            im = imresize(im, (64, 64))[:, :, :3]

            im = torch.Tensor(im).permute(2, 0, 1)

            return im, label_id
        except:
            return self.__getitem__(random.randint(0, len(self.ims) - 1))

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return len(self.ims)

class RealKittiLabel(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self,return_label=False, train=True):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        ims = sorted(glob("/data/vision/billf/scratch/yilundu/dataset/kitti/training/image_02/*/*.png"))

        self.ims = ims

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        # 433 - 808
        im_path = self.ims[index]
        im = imread(im_path)
        im = im[:, 433:808, :]
        im = imresize(im, (64, 64))[:, :, :3]

        im = torch.Tensor(im).permute(2, 0, 1)

        return im, 0

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return len(self.ims)

class RealKitti(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self,return_label=False, train=True):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        ims = sorted(glob("/data/vision/billf/scratch/yilundu/dataset/kitti/training/image_02/*/*.png"))

        self.ims = ims

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        # 433 - 808
        im = self.ims[index]
        im = imread(im)
        im = im[:, 433:808, :]
        im = imresize(im, (64, 64))[:, :, :3]

        im = torch.Tensor(im).permute(2, 0, 1)

        return im, index

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return len(self.ims)

if __name__ == "__main__":
    loader = Kitti(None)
    # for data in loader:
    #     print("here")
