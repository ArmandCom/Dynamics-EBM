import numpy as np
import math
import random
import matplotlib.pyplot as plt
import datetime
import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

n = 2

if torch.cuda.is_available():
    train_loc_file = 'loc_train_springs%d.npy' %n
    train_vel_file = 'vel_train_springs%d.npy' %n
    train_edges_file = 'edges_train_springs%d.npy' %n
    test_loc_file = 'loc_test_springs%d.npy' %n
    test_vel_file = 'vel_test_springs%d.npy' %n
    test_edges_file = 'edges_test_springs%d.npy' %n
else:
    train_loc_file = '../../../Desktop/loc_train_springs%d.npy' %n
    train_vel_file = '../../../Desktop/vel_train_springs%d.npy' %n
    train_edges_file = '../../../Desktop/edges_train_springs%d.npy' %n
    test_loc_file = '../../../Desktop/loc_test_springs%d.npy' %n
    test_vel_file = '../../../Desktop/vel_test_springs%d.npy' %n
    test_edges_file = '../../../Desktop/edges_test_springs%d.npy' %n

prefix = '../../../Desktop/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def to_tensor(array):
    return torch.from_numpy(np.array(array)).float().to(device)

def to_numpy(tensor):
    return tensor.cpu().numpy()

def transform(array, n):
    return np.transpose(array, (0, 3, 2, 1)).reshape(len(array), 2*n, -1)

def swish(x):
    return x * torch.sigmoid(x)

class Trajectory_Model(nn.Module):
    def __init__(self, length=10, units=128, activation='relu', convolute=True):
        super(Trajectory_Model, self).__init__()
        self.length = length
        self.channels = 4
        self.units = units
        self.convolute = convolute

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'swish':
            self.activation = swish
            
        if self.convolute:
            self.conv = nn.Conv1d(4, 16, 3)
            self.fc1 = nn.Linear(16 * (length-2), self.units)
        else:
            self.fc1 = nn.Linear(self.channels * self.length, self.units)
        self.fc2 = nn.Linear(self.units, self.units)
        self.fc3 = nn.Linear(self.units, 1)

    def forward(self, x):
        if self.convolute:
            x = self.conv(x)
            x = self.activation(x.reshape(x.size(0), -1))
        else:
            x = x.reshape((x.size(0), -1))
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

class Interaction_Model(nn.Module):
    def __init__(self, length=10, units=128, activation='relu', convolute=True):
        super(Interaction_Model, self).__init__()
        self.length = length
        self.channels = 8
        self.units = units
        self.convolute = convolute

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'swish':
            self.activation = swish
        
        if self.convolute:
            self.conv = nn.Conv1d(8, 32, 3)
            self.fc1 = nn.Linear(32 * (length-2), self.units)
        else:
            self.fc1 = nn.Linear(self.channels * self.length, self.units)
        self.fc2 = nn.Linear(self.units, self.units)
        self.fc3 = nn.Linear(self.units, 1)

    def forward(self, x):
        if self.convolute:
            x = self.conv(x)
            x = self.activation(x.reshape(x.size(0), -1))
        else:
            x = x.reshape((x.size(0), -1))
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

class Trajectory_Data(Dataset):
    def __init__(self, loc, vel, n=5, length=10, dist=1):
        self.n = n
        self.length = length
        self.dist = dist
        self.scale = n * (101 - length * dist)
        self.loc = transform(loc, n)
        self.vel = transform(vel, n)

    def __len__(self):
        return len(self.loc) * self.scale

    def __getitem__(self, index):
        rem = index % self.scale
        l = to_tensor(self.loc[int(index/self.scale),2*(rem%self.n):2*(rem%self.n)+2,int(rem/self.n):int(rem/self.n)+(self.length*self.dist):self.dist]).to(device)
        v = to_tensor(self.vel[int(index/self.scale),2*(rem%self.n):2*(rem%self.n)+2,int(rem/self.n):int(rem/self.n)+(self.length*self.dist):self.dist]).to(device)
        return torch.cat((l, v), 0)

class Interaction_Data(Dataset):
    def __init__(self, loc, vel, edges, n=5, length=10, dist=1, interaction=1):
        self.n = n
        self.length = length
        self.dist = dist
        self.interaction = interaction
        self.loc = transform(loc, n)
        self.vel = transform(vel, n)
        self.edges = edges
        self.pairs = np.array([np.concatenate((self.loc[ind,2*i:2*i+2,l:l+(length*dist):dist], self.vel[ind,2*i:2*i+2,l:l+(length*dist):dist], \
                                               self.loc[ind,2*j:2*j+2,l:l+(length*dist):dist], self.vel[ind,2*j:2*j+2,l:l+(length*dist):dist]), axis=0) \
                               for ind in range(len(self.loc)) for i in range(n) for j in range(i+1, n) for l in range(101-length*dist) if edges[ind][i][j] == interaction])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return to_tensor(self.pairs[index]).to(device)

class ReplayBuffer(object):
    def __init__(self, size=10000):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, ims):
        batch_size = ims.shape[0]
        if self._next_idx >= len(self._storage):
            self._storage.extend(list(ims))
        else:
            if batch_size + self._next_idx < self._maxsize:
                self._storage[self._next_idx:self._next_idx +
                              batch_size] = list(ims)
            else:
                split_idx = self._maxsize - self._next_idx
                self._storage[self._next_idx:] = list(ims)[:split_idx]
                self._storage[:batch_size - split_idx] = list(ims)[split_idx:]
        self._next_idx = (self._next_idx + ims.shape[0]) % self._maxsize

    def _encode_sample(self, idxes):
        ims = []
        for i in idxes:
            ims.append(self._storage[i])
        return np.array(ims)

    def sample(self, batch_size=243):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        """
        idxes = [random.randint(0, len(self._storage) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes)

def train(model, dataloader, initial_loc, initial_vel, initial_edges, sampling='MPPI', antidataloader=None, writer=None, name='Trajectory', epoch_num=0, \
          steps=200, std_dev=0.1, step_size=1, temperature=10, lr=10, inertia=False, dist=1, buffer=None, buffer_prop=0.95, regularizer=True, backprop=True):
    if lr == '10':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif lr == '5':
        optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.5, 0.99))
    elif lr == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    count = 0
    for epoch in range(100):
        #print(epoch)
        for trajectory in dataloader:
            data_output = model(trajectory)
            data_energy = data_output.mean()

            if antidataloader:
                num_samples = 128
            else:
                num_samples = 256

            samples = sample_train(model, initial_loc, initial_vel, initial_edges, sampling, num_samples, writer=writer, name=name, epoch=epoch_num, steps=steps, \
                                   std_dev=std_dev, step_size=step_size, temperature=temperature, inertia=inertia, dist=dist, buffer=buffer, buffer_prop=buffer_prop)

            if antidataloader:
                for antitrajectory in antidataloader:
                    break
                samples = torch.cat((samples, antitrajectory), 0)

            sample_output = model(samples.detach())
            sample_energy = sample_output.mean()

            loss = data_output.mean() - sample_output.mean()

            if backprop:
                model.requires_grad_(False)
                copy_output = model.forward(samples)
                model.requires_grad_(True)

                loss = loss + copy_output.mean()

            if lr == 'BCE':
                data_labels = torch.zeros(256, 1)
                sample_labels = torch.ones(256, 1)
                loss = criterion(data_output, data_labels) + criterion(sample_output, sample_labels)
            elif regularizer:
                loss = loss + torch.mean(data_output * data_output) + torch.mean(sample_output * sample_output)

            if writer:
                writer.add_scalar(name + '_Positive_Energy', data_energy, (epoch_num-1)* 100 + count)
                writer.add_scalar(name + '_Negative_Energy', sample_energy, (epoch_num-1) * 100 + count)
                writer.add_scalar(name + '_Energy_Difference', sample_energy - data_energy, (epoch_num-1) * 100 + count)
                writer.add_scalar(name + '_Loss', loss, (epoch_num-1) * 100 + count)
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            count = count + 1
            if count == 100:
                break
        if count == 100:
            break

def initialize(initial_loc, initial_vel, initial_edges, model_length, length, num_channels, num_samples, inertia=False, dist=1, random=False, position=None, ret_ind=False):
    n = int(num_channels/4)
    initial_loc = to_tensor(transform(initial_loc, n))
    initial_vel = to_tensor(transform(initial_vel, n))
    traj = torch.mul(torch.add(torch.rand(num_samples, num_channels, length), -0.5), 2)
    edges = torch.zeros(num_samples, n, n)
    rand = torch.randint(0, len(initial_loc), (num_samples,))
    placement = torch.randint(0, len(initial_loc[0][0])-(model_length*dist)+1, (num_samples,))
    for i in range(num_samples):
        if not random:
            rand[i] = i
        if position != None:
            placement[i] = position
        traj[i,:,:model_length] = torch.cat((initial_loc[rand[i],:int(num_channels/2),placement[i]:placement[i]+(model_length*dist):dist].reshape(-1, 2*model_length), \
                                             initial_vel[rand[i],:int(num_channels/2),placement[i]:placement[i]+(model_length*dist):dist].reshape(-1, 2*model_length)), 1).reshape(num_channels, model_length)
        
        if initial_edges is not None:
            edges[i] = to_tensor(initial_edges[rand[i]])

        if inertia:
            vel = torch.cat((torch.mul(initial_vel[rand[i],:2*n,placement[i]+(model_length*dist)-1:placement[i]+(model_length*dist)].reshape(-1, 2), dist/10), \
                             torch.zeros(n, 2)), 1).reshape(num_channels, 1)
            traj[i,:,model_length:] = traj[i,:,model_length-1:model_length] + vel * torch.arange(1, length - model_length + 1).repeat(num_channels, 1)

    if ret_ind:
        return traj, edges, rand
    return traj, edges

def sample_train(model, initial_loc, initial_vel, initial_edges, sampling='MPPI', num_samples=256, length=None, writer=None, name='Trajectory', epoch=0, \
                 steps=200, std_dev=0.1, step_size=1, temperature=10, inertia=False, dist=1, buffer=None, buffer_prop=0.95, display=False):
    if sampling.lower() == 'mppi':
        return sample_train_mppi(model, initial_loc, initial_vel, initial_edges, num_samples=num_samples, length=length, writer=writer, name=name, epoch=epoch, \
                                 steps=steps, std_dev=std_dev, temperature=temperature, inertia=inertia, dist=dist, buffer=buffer, buffer_prop=buffer_prop, display=display)
    if sampling.lower() == 'langevin':
        return sample_train_langevin(model, initial_loc, initial_vel, initial_edges, num_samples=num_samples, length=length, writer=writer, name=name, epoch=epoch, \
                                     steps=steps, std_dev=std_dev, step_size=step_size, inertia=inertia, buffer=buffer, buffer_prop=buffer_prop, display=display)

def sample_test(trajectory_model, interaction_model, initial_loc, initial_vel, initial_edges, n, sampling='MPPI', num_samples=256, length=None, writer=None, name='Test', epoch=0, \
                steps=200, std_dev=0.1, step_size=1, temperature=10, random=False, inertia=False, dist=1, position=0, display=False, show_graph=False):
    if sampling.lower() == 'mppi':
        return sample_test_mppi(trajectory_model, interaction_model, initial_loc, initial_vel, initial_edges, n, num_samples=num_samples, \
                                length=length, writer=writer, name=name, epoch=epoch, steps=steps, std_dev=std_dev, temperature=temperature, \
                                random=random, inertia=inertia, dist=dist, position=position, display=display, show_graph=show_graph)
    if sampling.lower() == 'langevin':
        return sample_test_langevin(trajectory_model, interaction_model, initial_loc, initial_vel, initial_edges, n, num_samples=num_samples, \
                                    length=length, writer=writer, name=name, epoch=epoch, steps=steps, std_dev=std_dev, step_size=step_size, \
                                    random=random, inertia=inertia, dist=dist, position=position, display=display, show_graph=show_graph)

def sample_train_mppi(model, initial_loc, initial_vel, initial_edges, num_samples=256, length=None, writer=None, name='Trajectory', epoch=0, \
                      steps=200, std_dev=0.1, temperature=10, inertia=False, dist=1, buffer=None, buffer_prop=0.95, display=False):
    with torch.no_grad():
        model_length = model.length
        num_channels = model.channels
        num_random = 100
        
        if not length:
            length = model_length

        if buffer == None or len(buffer) == 0:
            if inertia:
                traj = initialize(initial_loc, initial_vel, None, 1, length, num_channels, num_samples, \
                                  inertia=inertia, dist=dist, random=True, position=None)[0]
            else:
                traj = torch.mul(torch.add(torch.rand((num_samples, num_channels, length)), -0.5), 2)
        else:
            num_buffer = int(buffer_prop * num_samples)
            if inertia:
                traj = initialize(initial_loc, initial_vel, None, 1, length, num_channels, num_samples-num_buffer, \
                                  inertia=inertia, dist=dist, random=True, position=True)[0]
            else:
                traj = torch.mul(torch.add(torch.rand((num_samples-num_buffer, num_channels, length)), -0.5), 2)
            traj = torch.cat((traj, to_tensor(buffer.sample(num_buffer))), 0)

        if display:
            print(model(traj[:,:,:model_length]))
            graph(traj.squeeze(0))

        traj = torch.unsqueeze(traj, 1)
        
        en = [[], [], []]
        for i in range(steps - 1):
            noise = torch.normal(0, std_dev, (num_samples, num_random, num_channels, model_length))
            noise[:,0] = torch.zeros(num_channels, model_length)
            samples = traj[:,:,:,:model_length] + noise
            energy = torch.mul(model(samples.view(-1, num_channels, model_length)).view(num_samples, num_random, 1), temperature)
            weights = F.softmin(energy, dim=1)
            traj[:,:,:,:model_length] = torch.unsqueeze((samples.view(num_samples, num_random, num_channels*model_length) * \
                                                         weights).sum(dim=1).view(num_samples, num_channels, model_length), 1)
            if display or writer:
                for j in range(3):
                    en[j].append(model(traj[j,:,:,:model_length]))

    traj_copy = traj.clone()
    traj.requires_grad = True
    
    noise = torch.normal(0, std_dev, (num_samples, num_random, num_channels, model_length))
    noise[:,0] = torch.zeros(num_channels, model_length)
    samples = traj_copy + noise
    energy = torch.mul(model(samples.reshape(-1, num_channels, model_length)).view(num_samples, num_random, 1), temperature)
    weights = F.softmin(energy, dim=1)
    traj = torch.unsqueeze((samples.view(num_samples, num_random, num_channels*model_length) * \
                            weights).sum(dim=1).view(num_samples, num_channels, model_length), 1)
    if display or writer:
        for j in range(3):
            en[j].append(model(traj[j,:,:,:model_length]))

    if writer:
        for j in range(3):
            writer.add_figure('%s_Energy_During_MPPI_%d' %(name, j+1), graph_time(en[j]), epoch)
    if display:
        for j in range(3):
            print(en[j][-1])
        graph_time(en[0])

    traj = torch.squeeze(traj, 1)

    if buffer != None:
        buffer.add(to_numpy(traj.detach()))

    return traj

def sample_test_mppi(trajectory_model, interaction_model, initial_loc, initial_vel, initial_edges, n, num_samples=256, length=None, writer=None, name='Test', epoch=0, \
                     steps=200, std_dev=0.1, temperature=10, random=False, inertia=False, dist=1, position=0, display=False, show_graph=False):
    with torch.no_grad():
        model_length = trajectory_model.length
        num_channels = 4 * n
        num_random = 100
        
        if not length:
            length = model_length

        if random:
            traj, edges, init_ind = initialize(initial_loc, initial_vel, initial_edges, model_length, length, num_channels, num_samples, \
                                               inertia=inertia, dist=dist, random=True, position=position, ret_ind=True)
        else:
            traj, edges = initialize(initial_loc, initial_vel, initial_edges, model_length, length, num_channels, num_samples, \
                                     inertia=inertia, dist=dist, position=position)
            
        traj = torch.unsqueeze(traj, 1)

        c1 = torch.arange(n).repeat_interleave(n)
        c2 = torch.arange(n).repeat(n)

        if inertia:
            string = 'Inertia'
        else:
            string = 'Random'
        
        for ind in range(model_length, length):
            en = []
            for i in range(steps):
                noise = torch.cat((torch.zeros(num_samples, num_random, num_channels, model_length-1), \
                                   torch.normal(0, std_dev, (num_samples, num_random, num_channels, 1))), dim=3)
                noise[:,0] = torch.zeros(num_channels, model_length)
                samples = traj[:,:,:,ind-model_length+1:ind+1] + noise
                energy = torch.sum(trajectory_model(samples.reshape(-1, 4, model_length)).view(num_samples, num_random, n, 1), 2) + \
                         torch.sum(edges.unsqueeze(1).expand(-1, num_random, -1, -1) * \
                                   interaction_model(torch.cat([torch.cat((samples[:,:,4*c1[j]:4*c1[j]+4,:], samples[:,:,4*c2[j]:4*c2[j]+4,:]), 2) \
                                   for j in range(n*n)], 2).view(-1, 8, model_length)).view(num_samples, num_random, n, n), (2, 3)).unsqueeze(2)
            
                energy = torch.mul(energy, temperature)
                weights = F.softmin(energy, dim=1)
                traj[:,:,:,ind-model_length+1:ind+1] = torch.unsqueeze((samples.view(num_samples, num_random, num_channels*model_length) * \
                                                                        weights).sum(dim=1).view(num_samples, num_channels, model_length), 1)
                if display or (writer and length != 15):
                    e = torch.sum(trajectory_model(traj[:1,:,:,ind-model_length+1:ind+1].reshape(-1, 4, model_length))) + \
                        torch.sum(edges[0] * interaction_model(torch.cat([torch.cat((traj[:1,:,4*c1[j]:4*c1[j]+4,ind-model_length+1:ind+1], \
                                  traj[:1,:,4*c2[j]:4*c2[j]+4,ind-model_length+1:ind+1]), 2) for j in range(n*n)], 2).view(-1, 8, model_length)).view(n, n))
                    
                    en.append(e)
                    if writer and ind == model_length:
                        writer.add_scalar('%s_%s_Energy_During_MPPI' %(name, string), e, steps * (epoch-1) + i)
                        
            if display:
                print(en[-1])
                if show_graph:
                    graph_time(en)
            if writer and length != 15:
                writer.add_scalar('%s_%s_Energy_Sum' %(name, string), en[-1], 100 * (epoch-1) + ind-model_length)
                
                for j in range(n):
                    writer.add_scalar(('%s_%s_Trajectory_Energy_%d' %(name, string, j+1)), trajectory_model(traj[:1,:,4*j:4*j+4,ind-model_length+1:ind+1].squeeze(0)), 100 * (epoch-1) + ind-model_length)
                for j in range(n):
                    for k in range(j+1, n):
                        nrg = interaction_model(torch.cat((traj[:1,:,4*j:4*j+4,ind-model_length+1:ind+1], traj[:1,:,4*k:4*k+4,ind-model_length+1:ind+1]), 2).squeeze(0))
                        writer.add_scalar(('%s_%s_Interaction_Energy_%d_%d' %(name, string, j+1, k+1)), nrg, 100 * (epoch-1) + ind-model_length)
                
        traj = torch.squeeze(traj, 1)

        if writer:
            for i in range(min(4, len(traj))):
                t_energy = sum([torch.mean(trajectory_model(traj[i:i+1,:,ind-model_length+1:ind+1].reshape(-1, 4, model_length))) \
                                for ind in range(model_length, length)]) / (length - model_length)
                i_energy = sum([torch.sum(edges[i] * interaction_model(torch.cat([torch.cat((traj[i:i+1,4*c1[j]:4*c1[j]+4,ind-model_length+1:ind+1], \
                                traj[i:i+1,4*c2[j]:4*c2[j]+4,ind-model_length+1:ind+1]), 1) for j in range(n*n)], 1).view(-1, 8, model_length)).view(n, n)) \
                                for ind in range(model_length, length)]) / (torch.sum(edges[i]) * (length - model_length))
                if length == 15:
                    tag = '%s_Graph%d_%d' %(string, i+1, position)
                    if dist < 4:
                        border = 3
                    else:
                        border = 5
                else:
                    tag = '%s_Sample_%d' %(string, i+1)
                    border = 5
                writer.add_figure(tag, graph(traj[i].cpu(), border=border, t_energy=t_energy, i_energy=i_energy, training=True), epoch)

            if length != 15:
                criterion = nn.MSELoss(reduction = 'none')
                t = to_tensor(transform(initial_loc, n))[:num_samples,:,:length]
                v = to_tensor(transform(initial_vel, n))[:num_samples,:,:length]

                position_loss = criterion(traj[:num_samples].reshape(num_samples, 2*n, -1)[:,::2].reshape(num_samples, 2*n, -1), t)
                loss = criterion(traj[:num_samples], torch.cat((t.reshape(num_samples, n, -1), v.reshape(num_samples, n, -1)), 2).reshape(num_samples, 4*n, -1))

                for i in range(num_samples):
                    p = torch.mean(position_loss[i], 0)
                    l = torch.mean(loss[i], 0)
                
                    writer.add_figure('%s_Sample_Position_Error_%d' %(string, i+1), graph_time(p.cpu(), title='Position MSE Loss', x='Step', y='Error', training=True), epoch)
                    writer.add_figure('%s_Sample_Error_%d' %(string, i+1), graph_time(l.cpu(), title='MSE Loss', x='Step', y='Error', training=True), epoch)

        if random:
            return traj, init_ind
        return traj

def subsample_test_mppi(trajectory_model, interaction_model, dist_trajectory_model, dist_interaction_model, initial_loc, initial_vel, initial_edges, n, \
                   num_samples=256, length=None, writer=None, epoch=0, steps=200, std_dev=0.1, temperature=10, \
                   random=False, inertia=False, dist=5, display=False, position=0, show_graph=False):
    with torch.no_grad():
        model_length = trajectory_model.length
        num_channels = 4 * n
        num_random = 100
        
        if not length:
            length = model_length

        traj, edges = initialize(initial_loc, initial_vel, initial_edges, model_length, length, num_channels, num_samples, \
                                 inertia=inertia, dist=dist, random=random, position=position)
        traj = torch.unsqueeze(traj, 1)

        c1 = torch.arange(n).repeat_interleave(n)
        c2 = torch.arange(n).repeat(n)

        if inertia:
            string = 'Inertia'
        else:
            string = 'Random'
        
        for ind in range(model_length, length):
            en = []
            for i in range(steps):
                noise = torch.cat((torch.zeros(num_samples, num_random, num_channels, model_length-1), \
                                   torch.normal(0, std_dev, (num_samples, num_random, num_channels, 1))), dim=3)
                noise[:,0] = torch.zeros(num_channels, model_length)
                samples = traj[:,:,:,ind-model_length+1:ind+1] + noise
                energy = torch.sum(trajectory_model(samples.reshape(-1, 4, model_length)).view(num_samples, num_random, n, 1), 2) + \
                         torch.sum(edges.unsqueeze(1).expand(-1, num_random, -1, -1) * \
                                   interaction_model(torch.cat([torch.cat((samples[:,:,4*c1[j]:4*c1[j]+4,:], samples[:,:,4*c2[j]:4*c2[j]+4,:]), 2) \
                                   for j in range(n*n)], 2).view(-1, 8, model_length)).view(num_samples, num_random, n, n), (2, 3)).unsqueeze(2)
            
                energy = torch.mul(energy, temperature)
                weights = F.softmin(energy, dim=1)
                traj[:,:,:,ind-model_length+1:ind+1] = torch.unsqueeze((samples.view(num_samples, num_random, num_channels*model_length) * \
                                                                        weights).sum(dim=1).view(num_samples, num_channels, model_length), 1)
                if display or (writer and length != 20):
                    e = torch.sum(trajectory_model(traj[:1,:,:,ind-model_length+1:ind+1].reshape(-1, 4, model_length))) + \
                        torch.sum(edges[0] * interaction_model(torch.cat([torch.cat((traj[:1,:,4*c1[j]:4*c1[j]+4,ind-model_length+1:ind+1], \
                                  traj[:1,:,4*c2[j]:4*c2[j]+4,ind-model_length+1:ind+1]), 2) for j in range(n*n)], 2).view(-1, 8, model_length)).view(n, n))
                    
                    en.append(e)
                        
            if display:
                print(en[-1])
                if show_graph:
                    graph_time(en)

        for ind in range(model_length*dist-1, length):
            en = []
            for i in range(steps):
                noise = torch.cat((torch.zeros(num_samples, num_random, num_channels, model_length-8), \
                                   torch.normal(0, std_dev, (num_samples, num_random, num_channels, 8))), dim=3)
                noise[:,0] = torch.zeros(num_channels, model_length)
                samples = traj[:,:,:,ind-model_length*dist+1:ind+1:dist] + noise
                energy = torch.sum(dist_trajectory_model(samples.reshape(-1, 4, model_length)).view(num_samples, num_random, n, 1), 2) + \
                         torch.sum(edges.unsqueeze(1).expand(-1, num_random, -1, -1) * \
                                   dist_interaction_model(torch.cat([torch.cat((samples[:,:,4*c1[j]:4*c1[j]+4,:], samples[:,:,4*c2[j]:4*c2[j]+4,:]), 2) \
                                   for j in range(n*n)], 2).view(-1, 8, model_length)).view(num_samples, num_random, n, n), (2, 3)).unsqueeze(2)
            
                energy = torch.mul(energy, temperature)
                weights = F.softmin(energy, dim=1)
                traj[:,:,:,ind-model_length*dist+1:ind+1:dist] = torch.unsqueeze((samples.view(num_samples, num_random, num_channels*model_length) * \
                                                                                  weights).sum(dim=1).view(num_samples, num_channels, model_length), 1)

        traj = torch.squeeze(traj, 1)

        if writer:
            for i in range(4):
                t_energy = sum([torch.mean(trajectory_model(traj[i:i+1,:,ind-model_length+1:ind+1].reshape(-1, 4, model_length))) \
                                for ind in range(model_length, length)]) / (length - model_length)
                i_energy = sum([torch.sum(edges[i] * interaction_model(torch.cat([torch.cat((traj[i:i+1,4*c1[j]:4*c1[j]+4,ind-model_length+1:ind+1], \
                                traj[i:i+1,4*c2[j]:4*c2[j]+4,ind-model_length+1:ind+1]), 1) for j in range(n*n)], 1).view(-1, 8, model_length)).view(n, n)) \
                                for ind in range(model_length, length)]) / (torch.sum(edges[i]) * (length - model_length))
                
                tag = 'Sample_%d_%s' %(i+1, string)
                border = 5
                writer.add_figure(tag, graph(traj[i].cpu(), border=border, t_energy=t_energy, i_energy=i_energy, training=True), epoch)

        return traj

def sample_train_langevin(model, initial_loc, initial_vel, initial_edges, num_samples=256, length=None, writer=None, name='Trajectory', epoch=0, \
                          steps=200, std_dev=0.01, step_size=1, inertia=False, buffer=None, buffer_prop=0.95, display=False):
    model_length = model.length
    num_channels = model.channels
        
    if not length:
        length = model_length

    if buffer == None or len(buffer) == 0:
        if inertia:
            traj = initialize(initial_loc, initial_vel, None, 1, length, num_channels, num_samples, inertia=inertia, random=True, position=None)[0]
        else:
            traj = torch.mul(torch.add(torch.rand((num_samples, num_channels, length)), -0.5), 2)
    else:
        num_buffer = int(buffer_prop * num_samples)
        if inertia:
            traj = initialize(initial_loc, initial_vel, None, 1, length, num_channels, num_samples-num_buffer, inertia=inertia, random=True, position=True)[0]
        else:
            traj = torch.mul(torch.add(torch.rand((num_samples-num_buffer, num_channels, length)), -0.5), 2)
        traj = torch.cat((traj, to_tensor(buffer.sample(num_buffer))), 0)

    traj.requires_grad_(True)

    en = [[], [], []]
    for i in range(steps):
        if std_dev:
            traj =  traj + torch.normal(0, std_dev, (num_samples, num_channels, model_length))

        energy = model(traj[:,:,:model_length])
        if i == steps-1:
            traj_grad = torch.autograd.grad([energy.sum()], [traj], create_graph=True)[0]
        else:
            traj_grad = torch.autograd.grad([energy.sum()], [traj])[0]
        traj = torch.clamp(traj - step_size * traj_grad, -5, 5)
        
        if display or writer:
            for j in range(3):
                en[j].append(model(traj[j:j+1,:,:model_length]))
    if writer:
        for j in range(3):
            writer.add_figure('%s_Energy_During_Langevin_%d' %(name, j+1), graph_time(en[j]), epoch)
    if display:
        for j in range(3):
            print(en[j][-1])
            graph_time(en[j])

    if buffer != None:
        buffer.add(to_numpy(traj.detach()))
            
    return traj

def sample_test_langevin(trajectory_model, interaction_model, initial_loc, initial_vel, initial_edges, n, num_samples=256, length=None, writer=None, \
                         name='Test', epoch=0, steps=200, std_dev=0.01, step_size=1, random=False, inertia=False, dist=1, position=0, display=False, show_graph=False):
    model_length = trajectory_model.length
    num_channels = 4 * n

    if not length:
        length = model_length

    if random:
        traj, edges, init_ind = initialize(initial_loc, initial_vel, initial_edges, model_length, length, num_channels, num_samples, \
                                 inertia=inertia, dist=dist, random=True, position=position, ret_ind=True)
    else:
        traj, edges = initialize(initial_loc, initial_vel, initial_edges, model_length, length, num_channels, num_samples, \
                                 inertia=inertia, dist=dist, position=position)

    traj.requires_grad_(True)

    c1 = torch.arange(n).repeat_interleave(n)
    c2 = torch.arange(n).repeat(n)

    if inertia:
        string = 'Inertia'
    else:
        string = 'Random'

    for ind in range(model_length, length):
        en = []
        for i in range(steps):
            if std_dev:
                traj[:,:,ind-model_length+1:ind+1] = traj[:,:,ind-model_length+1:ind+1] + torch.normal(0, std_dev, (num_samples, num_channels, model_length))
           
            energy = torch.sum(trajectory_model(traj[:,:,ind-model_length+1:ind+1].reshape(-1, 4, model_length)).view(num_samples, n, 1), 1) + \
                     torch.sum(edges * interaction_model(torch.cat([torch.cat((traj[:,4*c1[j]:4*c1[j]+4,ind-model_length+1:ind+1], traj[:,4*c2[j]:4*c2[j]+4,ind-model_length+1:ind+1]), 1) \
                                                                    for j in range(n*n)], 1).view(-1, 8, model_length)).view(num_samples, n, n), (1, 2)).unsqueeze(1)
            traj_grad = torch.autograd.grad([energy.sum()], [traj])[0]
            traj[:,:,ind] = torch.clamp(traj[:,:,ind] - step_size * traj_grad[:,:,ind], -5, 5)

            if display or (writer and length != 15):
                e = torch.sum(trajectory_model(traj[:1,:,ind-model_length+1:ind+1].reshape(-1, 4, model_length)).view(1, n, 1), 1) + \
                    torch.sum(edges[:1] * interaction_model(torch.cat([torch.cat((traj[:1,4*c1[j]:4*c1[j]+4,ind-model_length+1:ind+1], traj[:1,4*c2[j]:4*c2[j]+4,ind-model_length+1:ind+1]), 1) \
                              for j in range(n*n)], 1).view(-1, 8, model_length)).view(1, n, n), (1, 2)).unsqueeze(1)

                en.append(e)
                if writer and ind == model_length:
                    writer.add_scalar('%s_%s_Energy_During_Langevin' %(name, string), e, steps * (epoch-1) + i)
                        
            if display:
                print(en[-1])
                if show_graph:
                    graph_time(en)

    if writer:
        for i in range(min(4, len(traj))):
            t_energy = sum([torch.mean(trajectory_model(traj[i:i+1,:,ind-model_length+1:ind+1].reshape(-1, 4, model_length))) \
                            for ind in range(model_length, length)]) / (length - model_length)
            i_energy = sum([torch.sum(edges[i] * interaction_model(torch.cat([torch.cat((traj[i:i+1,4*c1[j]:4*c1[j]+4,ind-model_length+1:ind+1], \
                            traj[i:i+1,4*c2[j]:4*c2[j]+4,ind-model_length+1:ind+1]), 1) for j in range(n*n)], 1).view(-1, 8, model_length)).view(n, n)) \
                            for ind in range(model_length, length)]) / (torch.sum(edges[i]) * (length - model_length))
            if length == 15:
                tag = '%s_Graph%d_%d' %(string, i+1, position)
                if dist < 4:
                    border = 3
                else:
                    border = 5
            else:
                tag = '%s_Sample_%d' %(string, i+1)
                border = 5
            writer.add_figure(tag, graph(traj[i].detach().cpu(), border=border, t_energy=t_energy, i_energy=i_energy, training=True), epoch)

        if length != 15:
            criterion = nn.MSELoss(reduction = 'none')
            t = to_tensor(transform(initial_loc, n))[:num_samples,:,:length]
            v = to_tensor(transform(initial_vel, n))[:num_samples,:,:length]

            position_loss = criterion(traj[:num_samples].reshape(num_samples, 2*n, -1)[:,::2].reshape(num_samples, 2*n, -1), t)
            loss = criterion(traj[:num_samples], torch.cat((t.reshape(num_samples, n, -1), v.reshape(num_samples, n, -1)), 2).reshape(num_samples, 4*n, -1))

            for i in range(num_samples):
                p = torch.mean(position_loss[i], 0)
                l = torch.mean(loss[i], 0)

                writer.add_figure('%s_Sample_Position_Error_%d' %(string, i+1), graph_time(p.detach().cpu(), title='Position MSE Loss', x='Step', y='Error', training=True), epoch)
                writer.add_figure('%s_Sample_Error_%d' %(string, i+1), graph_time(l.detach().cpu(), title='MSE Loss', x='Step', y='Error', training=True), epoch)

    if random:
        return traj, init_ind
    return traj

def inference(interaction_model, traj, n, display=False, cutoff='mean'):
    with torch.no_grad():
        model_length = interaction_model.length
        num_traj = len(traj)
        length = len(traj[0][0])
        
        edges = torch.zeros(num_traj, n, n)
        
        c1 = torch.arange(n).repeat_interleave(n)
        c2 = torch.arange(n).repeat(n)

        for ind in range(model_length, length):
            edges = edges - interaction_model(torch.cat([torch.cat((traj[:,4*c1[i]:4*c1[i]+4,ind-model_length+1:ind+1], \
                            traj[:,4*c2[i]:4*c2[i]+4,ind-model_length+1:ind+1]), 1) for i in range(n*n)], 1).view(-1, 8, model_length)).view(num_traj, n, n)

        edges = torch.div(edges, length-model_length)
        
        if display:
            print(edges)
            print('Mean:', torch.mean(edges).item())
            print('Median:', torch.median(edges).item())
        
        if type(cutoff) in [int, float]:
            boundary = -cutoff
        elif cutoff.lower() == 'zero':
            boundary = 0
        elif cutoff.lower() == 'mean':
            boundary = torch.mean(edges)
        elif cutoff.lower() == 'median':
            boundary = torch.median(edges)
        edges = torch.round(torch.clamp(edges - boundary + 0.5, 0, 1))
        for i in range(n):
            edges[:,i,i] = 0
        return edges

def test_prediction(trajectory_model, interaction_model, test_loc, test_vel, test_edges, n, sampling='MPPI', writer=None, name='Test', epoch=0, \
                    num_samples=30, length=15, steps=100, std_dev=0.1, step_size=1, temperature=10, dist=1, display=False):
    criterion = nn.MSELoss(reduction = 'none')

    for inertia in [True, False]:
        for position in [0, 20, 40, 50, 60, 70, 80]:
            traj, ind = sample_test(trajectory_model, interaction_model, test_loc, test_vel, test_edges, n, sampling=sampling, writer=writer, name=name, \
                                    epoch=epoch, num_samples=num_samples, length=length, steps=steps, std_dev=std_dev, step_size=step_size, \
                                    temperature=temperature, random=True, inertia=inertia, dist=dist, position=position)

            t = to_tensor(transform(test_loc, n))
            v = to_tensor(transform(test_vel, n))
            t = torch.index_select(t, 0, ind)[:,:,position:position+length*dist:dist]
            v = torch.index_select(v, 0, ind)[:,:,position:position+length*dist:dist]

            position_loss = criterion(traj.reshape(num_samples, 2*n, -1)[:,::2].reshape(num_samples, 2*n, -1), t)
            position_loss = torch.mean(position_loss, (0, 1))[10:]

            loss = criterion(traj, torch.cat((t.reshape(num_samples, n, -1), v.reshape(num_samples, n, -1)), 2).reshape(num_samples, 4*n, -1))
            loss = torch.mean(loss, (0, 1))[10:]

            if inertia:
                string = 'Inertia'
            else:
                string = 'Random'

            if not writer:
                print(loss)
            else:
                for i in range(5):
                    writer.add_scalar('%s_%s_Position_Error_Step_%d' %(name, string, position+11+i), position_loss[i], epoch)
                    writer.add_scalar('%s_%s_Error_Step_%d' %(name, string, position+11+i), loss[i], epoch)

    if display and epoch == 0:
        initialization = initialize(test_loc, test_vel, test_edges, trajectory_model.length, length, 20, num_samples, inertia=True, dist=dist)[0]

        position_loss_init = criterion(initialization.reshape(num_samples, 2*n, -1)[:,::2].reshape(num_samples, 2*n, -1), t)
        position_loss_init = torch.mean(position_loss_init, (0, 1))

        loss_init = criterion(initialization, torch.cat((t.reshape(num_samples, n, -1), v.reshape(num_samples, n, -1)), 2).reshape(num_samples, 4*n, -1))
        loss_init = torch.mean(loss_init, (0, 1))
        
        for i in range(10, length):
            print('Position Error At Step %d: ' %(i+1), position_loss_init[i].item())

        for i in range(10, length):
            print('Error At Step %d: ' %(i+1), loss_init[i].item())

def test_inference(interaction_model, test_loc, test_vel, test_edges, n, num_to_test=1000, cutoff='Mean', writer=None, name='Test', epoch=0, training=False):  
    test_loc = to_tensor(transform(test_loc, n)[:num_to_test])
    test_vel = to_tensor(transform(test_vel, n)[:num_to_test])
    test_edges = to_tensor(test_edges[:num_to_test,:,:])

    traj = torch.cat((test_loc.view(num_to_test, n, -1), test_vel.view(num_to_test, n, -1)), 2).reshape(num_to_test, 4*n, -1)
    guess_edges = inference(interaction_model, traj, n, cutoff=cutoff)

    actual = torch.sum(test_edges).item() / num_to_test
    guess = torch.sum(guess_edges).item() / num_to_test
    ones = torch.sum(torch.clamp(guess_edges - test_edges, 0, 1)).item() / num_to_test
    zeros = torch.sum(torch.clamp(test_edges - guess_edges, 0, 1)).item() / num_to_test

    if not training or not writer:
        print('%s Num Actual Ones: %.3f' %(name, actual))
        print('%s Num Guessed Ones: %.3f' %(name, guess))
        print('%s Diff: %.3f' %(name, guess - actual))
        print('%s Extra Ones: %.3f' %(name, ones))
        print('%s Extra Zeros: %.3f' %(name, zeros))
        print('%s Total incorrect: %.3f' %(name, ones + zeros))
    else:
        if type(cutoff) == str:
            writer.add_scalar('Num_Guessed Interactions_%s_%s' %(cutoff, name), guess, epoch)
            writer.add_scalar('Extra_Ones_Guessed_%s_%s' %(cutoff, name), ones, epoch)
            writer.add_scalar('Extra_Zeros_Guessed_%s_%s' %(cutoff, name), zeros, epoch)
            writer.add_scalar('Num_Incorrect_%s_%s' %(cutoff, name), ones + zeros, epoch)
        elif type(cutoff) == int:
            writer.add_scalar('Num_Guessed Interactions_%d_%s' %(cutoff, name), guess, epoch)
            writer.add_scalar('Extra_Ones_Guessed_%d_%s' %(cutoff, name), ones, epoch)
            writer.add_scalar('Extra_Zeros_Guessed_%d_%s' %(cutoff, name), zeros, epoch)
            writer.add_scalar('Num_Incorrect_%d_%s' %(cutoff, name), ones + zeros, epoch)
        else:
            writer.add_scalar('Num_Guessed Interactions_%.1f_%s' %(cutoff, name), guess, epoch)
            writer.add_scalar('Extra_Ones_Guessed_%.1f_%s' %(cutoff, name), ones, epoch)
            writer.add_scalar('Extra_Zeros_Guessed_%.1f_%s' %(cutoff, name), zeros, epoch)
            writer.add_scalar('Num_Incorrect_%.1f_%s' %(cutoff, name), ones + zeros, epoch)
        if epoch == 0:
            print('Num Actual Ones:', actual)

def graph_line(traj, ax, length, color=None):
    start_alpha = 0.1
    end_alpha = 1
    segments = 10

    for i in range(segments):
        alpha = start_alpha + (end_alpha - start_alpha) * i / (segments - 1)
        start = int(length*i/segments)
        end = (int(length*(i+1)/segments) if i == segments-1 else int(length*(i+1)/segments)+1)
        ax.plot(traj[0][start:end], traj[1][start:end], '.-', color=color, alpha=alpha)

def graph(traj, n=5, length=None, border=5, t_energy=None, i_energy=None, graph_vel=False, includes_vel=True, transformed=True, training=False, ground_truth=False, save=False):
    fig, ax = plt.subplots()

    if type(traj) == np.ndarray:
        traj = to_tensor(traj)

    red = '#d62728'
    pink = '#ff00ff' #'#e377c2'
    yellow = '#ffff00' #'#bcbd22'
    green = '#00ff00' #'#2ca02c'
    cyan = '#00ffff' #'#17becf'
    blue = '#1f77b4'
    purple = '#b041ff' #'#9467bd'
    colors = {0: pink, 1: yellow, 2: green, 3: cyan, 4: purple}
    if includes_vel:
        spacing = 4
    else:
        spacing = 2

    if graph_vel:
        title = 'Velocities'
        shift = 2
    else:
        title = 'Trajectories'
        shift = 0
        
    if not transformed:
        traj = transform(traj.unsqueeze(0), n).squeeze(0)
        title = 'Ground Truth ' + title
    elif ground_truth:
        title = 'Ground Truth ' + title

    if t_energy or i_energy:
        title = title + ' ('
        if t_energy:
            title = title + 'T: %.4f' %t_energy + ('; ' if i_energy else '')
        if i_energy:
            title = title + 'I: %.4f' %i_energy
        title = title + ')'

    if not length:
        length = len(traj[0])

    for i in range(0, len(traj), spacing):
        graph_line(traj[i+shift:i+shift+2], ax, length, colors[(i/spacing)%5])
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set(xlim = (-border, border), ylim = (-border, border))
    
    if save:
        time = datetime.datetime.now()
        graph_file = 'trajectories %02d-%02d %02d-%02d-%02d.png' %(time.month, time.day, time.hour, time.minute, time.second)
        plt.savefig(graph_file)

    if not training:
        plt.show()

    return fig

def graph_overlay(trajectories, test_loc, n=5, num_graphs=5, length=None, border=5, includes_vel=True, training=False, save=False):
    red = '#d62728'
    pink = '#ff00ff' #'#e377c2'
    yellow = '#ffff00' #'#bcbd22'
    green = '#00ff00' #'#2ca02c'
    cyan = '#00ffff' #'#17becf'
    blue = '#1f77b4'
    purple = '#b041ff' #'#9467bd'
    colors_light = {0: pink, 1: yellow, 2: green, 3: cyan, 4: purple}
    colors_dark = {0: '#e377c2', 1: '#bcbd22', 2: '#2ca02c', 3: '#17becf', 4: '#9467bd'}

    title = 'Trajectories'
    shift = 0
    
    if not length:
        length = 100

    figures = []
    test_loc = to_tensor(transform(test_loc, n)).cpu()
    
    for num in range(num_graphs):
        fig, ax = plt.subplots()

        spacing = 2
        traj = test_loc[num]
        for i in range(0, len(traj), spacing):
            graph_line(traj[i+shift:i+shift+2], ax, length, colors_dark[(i/spacing)%5])

        if includes_vel:
            spacing = 4
        traj = trajectories[num]
        for i in range(0, len(traj), spacing):
            graph_line(traj[i+shift:i+shift+2], ax, length, colors_light[(i/spacing)%5])
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set(xlim = (-border, border), ylim = (-border, border))

        if save:
            time = datetime.datetime.now()
            graph_file = 'overlaid %d %02d-%02d %02d-%02d-%02d.png' %(num, time.month, time.day, time.hour, time.minute, time.second)
            plt.savefig(graph_file)

        if training:
            figures.append(fig)
        else:
            plt.show()

    return figures

def graph_time(traj, title='Energy Over Time', x='Time', y='Energy', training=False):
    fig, ax = plt.subplots()
    
    t = range(len(traj))

    ax.plot(t, traj)
    ax.plot(t, traj, '.')

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    
    if not training:
        plt.show()

    return fig

def graph_ground_truth(test_loc, test_vel, test_edges, n, writer, num_graphs=4, model_length=10, length=100):
    for i in range(num_graphs):
        t = to_tensor(transform(test_loc, n))[i]
        v = to_tensor(transform(test_vel, n))[i]
        edges = to_tensor(test_edges[i])
        traj = torch.cat((t.view(n, -1), v.view(n, -1)), 1).reshape(1, 4*n, -1)

        fig = graph(traj.squeeze(0).cpu(), training=True, ground_truth=True)

        for string in ['Inertia', 'Random']:
            writer.add_figure('%s_Sample_%d' %(string, i+1), fig)
        
def main():
    length = 10
    
    train_loc = np.load(train_loc_file, allow_pickle=True)
    train_vel = np.load(train_vel_file, allow_pickle=True)
    train_edges = np.load(train_edges_file, allow_pickle=True)
    test_loc = np.load(test_loc_file, allow_pickle=True)
    test_vel = np.load(test_vel_file, allow_pickle=True)
    test_edges = np.load(test_edges_file, allow_pickle=True)

    train_trajectory_dataset = Trajectory_Data(train_loc, train_vel, n, length)
    train_trajectory_dataloader = DataLoader(train_trajectory_dataset, batch_size=256, shuffle=True, drop_last=True)
    train_interaction_dataset = Interaction_Data(train_loc, train_vel, train_edges, n, length)
    train_interaction_dataloader = DataLoader(train_interaction_dataset, batch_size=256, shuffle=True, drop_last=True)

    trajectory_model = Trajectory_Model(length, convolute=False)
    interaction_model = Interaction_Model(length, convolute=False)
    
if __name__ == '__main__':
    main()
