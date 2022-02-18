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
    PATH1 = 'springs_trajectory_net.pth'
    PATH2 = 'springs_interaction_net.pth'
    PATH1_OLD = 'springs_trajectory_net_saved.pth'
    PATH2_OLD = 'springs_interaction_net_saved.pth'
else:
    train_loc_file = '../../../Desktop/loc_train_springs%d.npy' %n
    train_vel_file = '../../../Desktop/vel_train_springs%d.npy' %n
    train_edges_file = '../../../Desktop/edges_train_springs%d.npy' %n
    test_loc_file = '../../../Desktop/loc_test_springs%d.npy' %n
    test_vel_file = '../../../Desktop/vel_test_springs%d.npy' %n
    test_edges_file = '../../../Desktop/edges_test_springs%d.npy' %n

    input_file = '../../../Desktop/springs_trajectory.npy'
    PATH1 = '../../../Desktop/springs_trajectory_net.pth'
    PATH2 = '../../../Desktop/springs_interaction_net.pth'


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

class Model(nn.Module):
    def __init__(self, n=5, length=10, units=128, activation='relu', convolute=False):
        super(Model, self).__init__()
        self.n = n
        self.length = length
        self.channels = 4*n
        self.units = units
        self.convolute = convolute
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'swish':
            self.activation = swish

        if self.convolute:
            self.conv1 = nn.Conv1d(4, 16, 3)
            self.traj1 = nn.Linear(16 * (length-2), self.units)
            
            self.conv2 = nn.Conv1d(8, 16, 3)
            self.int1 = nn.Linear(32 * (length-2), self.units)
        else:
            self.traj1 = nn.Linear(4 * self.length, self.units)
            self.int1 = nn.Linear(8 * self.length, self.units)
        
        self.traj2 = nn.Linear(self.units, self.units)
        self.traj3 = nn.Linear(self.units, 1)
        
        self.int2 = nn.Linear(self.units, self.units)
        self.int3 = nn.Linear(self.units, 1)

    def traj_energy(self, x):
        if self.convolute:
            x = self.conv1(x)
            x = self.activation(x.reshape(x.size(0), -1))
        else:
            x = x.reshape((x.size(0), -1))
        x = self.activation(self.traj1(x))
        x = self.activation(self.traj2(x))
        x = self.traj3(x)
        return x

    def int_energy(self, x):
        if self.convolute:
            x = self.conv2(x)
            x = self.activation(x.reshape(x.size(0), -1))
        else:
            x = x.reshape((x.size(0), -1))
        x = self.activation(self.int1(x))
        x = self.activation(self.int2(x))
        x = self.int3(x)
        return x

    def forward(self, x, edges):
        c1 = torch.arange(self.n).repeat_interleave(self.n)
        c2 = torch.arange(self.n).repeat(self.n)
        
        energy = torch.sum(self.traj_energy(x.reshape(-1, 4, self.length)).view(x.size(0), self.n, 1), 1) + \
                 torch.sum(edges * self.int_energy(torch.cat([torch.cat((x[:,4*c1[j]:4*c1[j]+4,:], x[:,4*c2[j]:4*c2[j]+4,:]), 1) \
                           for j in range(self.n*self.n)], 1).view(-1, 8, self.length)).view(x.size(0), self.n, self.n), (1, 2)).view(-1, 1)

        return energy

class Data(Dataset):
    def __init__(self, loc, vel, edges, n=5, length=10, dist=1):
        self.n = n
        self.length = length
        self.dist = dist
        self.scale = 101 - length * dist
        self.loc = to_tensor(transform(loc, n)).to(device)
        self.vel = to_tensor(transform(vel, n)).to(device)
        self.edges = edges

    def __len__(self):
        return len(self.loc) * self.scale

    def __getitem__(self, index):
        rem = index % self.scale
        l = self.loc[int(index/self.scale),:,index%self.scale:index%self.scale+(self.length*self.dist):self.dist]
        v = self.vel[int(index/self.scale),:,index%self.scale:index%self.scale+(self.length*self.dist):self.dist]

        return torch.cat((l.reshape(self.n, -1), v.reshape(self.n, -1)), 1).reshape(4*self.n, -1), self.edges[int(index/self.scale)]

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
        self._edges = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, ims, edges):
        batch_size = ims.shape[0]
        if self._next_idx >= len(self._storage):
            self._storage.extend(list(ims))
            self._edges.extend(list(edges))
        else:
            if batch_size + self._next_idx < self._maxsize:
                self._storage[self._next_idx:self._next_idx + batch_size] = list(ims)
                self._edges[self._next_idx:self._next_idx + batch_size] = list(edges)
            else:
                split_idx = self._maxsize - self._next_idx
                self._storage[self._next_idx:] = list(ims)[:split_idx]
                self._storage[:batch_size - split_idx] = list(ims)[split_idx:]
                self._edges[self._next_idx:] = list(edges)[:split_idx]
                self._edges[:batch_size - split_idx] = list(edges)[split_idx:]
        self._next_idx = (self._next_idx + ims.shape[0]) % self._maxsize

    def _encode_sample(self, idxes):
        ims = []
        edges = []
        for i in idxes:
            ims.append(self._storage[i])
            edges.append(self._edges[i])
        return np.array(ims), np.array(edges)

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

def train(model, dataloader, initial_loc, initial_vel, initial_edges, n, antidataloader=None, writer=None, epoch_num=0, \
          mppi_steps=200, std_dev=0.1, temperature=10, lr=10, inertia=False, dist=1, buffer=None, regularizer=True):
    if lr == '10':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif lr == '5':
        optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.5, 0.99))
    elif lr == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(1):
        #print(epoch)
        count = 0
        for trajectory, edges in dataloader:
            data_output = model(trajectory, edges)
            data_energy = data_output.mean()

            if antidataloader:
                num_samples = 128
            else:
                num_samples = 256
                
            samples, sample_edges = sample(model, initial_loc, initial_vel, initial_edges, n, num_samples, writer=writer, epoch=epoch_num, \
                                    mppi_steps=mppi_steps, std_dev=std_dev, temperature=temperature, training=True, inertia=inertia, dist=dist, buffer=buffer)                

            if antidataloader:
                for antitrajectory, anti_edges in antidataloader:
                    break
                anti_edges = torch.round(torch.rand(128, n, n))
                for i in range(n):
                    anti_edges[:,i,i] = 0
                samples = torch.cat((samples, antitrajectory[:128]), 0)
                sample_edges = torch.cat((sample_edges, anti_edges), 0)

            sample_output = model(samples.detach(), sample_edges)
            sample_energy = sample_output.mean()

            copy_output = copy.deepcopy(model)(samples, sample_edges)
            
            loss = data_output.mean() - sample_output.mean() + copy_output.mean()
            if lr == 'BCE':
                data_labels = torch.zeros(256, 1)
                sample_labels = torch.ones(256, 1)
                loss = criterion(data_output, data_labels) + criterion(sample_output, sample_labels)
            elif regularizer:
                loss = loss + torch.mean(data_output * data_output) + torch.mean(sample_output * sample_output)

            if writer:
                writer.add_scalar('Positive_Energy', data_energy, (epoch_num-1)* 100 + count)
                writer.add_scalar('Negative_Energy', sample_energy, (epoch_num-1) * 100 + count)
                writer.add_scalar('Energy_Difference', sample_energy - data_energy, (epoch_num-1) * 100 + count)
                writer.add_scalar('Loss', loss, (epoch_num-1) * 100 + count)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            count = count + 1
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
            vel = torch.cat((torch.mul(initial_vel[rand[i],:int(num_channels/2),placement[i]+(model_length*dist)-1:placement[i]+(model_length*dist)].reshape(-1, 2), dist/10), \
                             torch.zeros(n, 2)), 1).reshape(num_channels, 1)
            traj[i,:,model_length:] = traj[i,:,model_length-1:model_length] + vel * torch.arange(1, length - model_length + 1).repeat(num_channels, 1)

    if ret_ind:
        return traj, edges, rand
    return traj, edges

def sample(model, initial_loc, initial_vel, initial_edges, n, num_samples=256, length=None, writer=None, epoch=0, mppi_steps=200, std_dev=0.1, \
           temperature=10, training=False, random=False, inertia=False, dist=1, position=0, buffer=None, display=False, show_graph=False):
    with torch.no_grad():
        model_length = model.length
        num_channels = model.channels
        num_random = 100
        
        if not length:
            length = model_length

        if training:
            if buffer == None or len(buffer) == 0:
                if inertia:
                    traj, edges = initialize(initial_loc, initial_vel, initial_edges, 1, length, num_channels, num_samples, \
                                             inertia=inertia, dist=dist, random=True, position=None)
                else:
                    traj = torch.mul(torch.add(torch.rand(num_samples, num_channels, length), -0.5), 2)
                    edges = torch.round(torch.rand(num_samples, n, n))
            else:
                num_buffer = int(0.95 * num_samples)
                if inertia:
                    traj, edges = initialize(initial_loc, initial_vel, initial_edges, 1, length, num_channels, num_samples-num_buffer, \
                                             inertia=inertia, dist=dist, random=True, position=None)
                else:
                    traj = torch.mul(torch.add(torch.rand(num_samples-num_buffer, num_channels, length), -0.5), 2)
                    edges = torch.round(torch.rand(num_samples-num_buffer, n, n))
                buf_traj, buf_edges = buffer.sample(num_buffer)
                traj = torch.cat((traj, to_tensor(buf_traj)), 0)
                edges = torch.cat((edges, to_tensor(buf_edges)), 0)
            
        else:
            if random:
                traj, edges, init_ind = initialize(initial_loc, initial_vel, initial_edges, model_length, length, num_channels, num_samples, \
                                                   inertia=inertia, dist=dist, random=True, position=position, ret_ind=True)
            else:
                traj, edges = initialize(initial_loc, initial_vel, initial_edges, model_length, length, num_channels, num_samples, \
                                         inertia=inertia, dist=dist, position=position)
            if inertia:
                string = 'Inertia'
            else:
                string = 'Random'

        if display and name:
            print(model(traj[:,:,:model_length], edges))
            graph(traj.squeeze(0))

        traj = torch.unsqueeze(traj, 1)
        
        for ind in range(0 if training else 1, length - model_length + 1):
            en = [[], [], []]
            for i in range((mppi_steps-1 if training else mppi_steps)):
                noise = torch.normal(0, std_dev, (num_samples, num_random, num_channels, model_length)) if training else \
                        torch.cat((torch.zeros(num_samples, num_random, num_channels, model_length-1), \
                             torch.normal(0, std_dev, (num_samples, num_random, num_channels, 1))), dim=3)
                noise[:,0] = torch.zeros(num_channels, model_length)
                samples = traj[:,:,:,ind:ind+model_length] + noise
                energy = torch.mul(model(samples.reshape(-1, num_channels, model_length), edges.unsqueeze(1).expand(-1, num_random, -1, -1).reshape(-1, n, n)). \
                                   view(num_samples, num_random, 1), temperature)
                weights = F.softmin(energy, dim=1)
                traj[:,:,:,ind:ind+model_length] = torch.unsqueeze((samples.view(num_samples, num_random, num_channels*model_length) * \
                                                                    weights).sum(dim=1).view(num_samples, num_channels, model_length), 1)
                if display or writer:
                    en[0].append(model(traj[0,:,:,ind:ind+model_length], edges[0]))
                    en[1].append(model(traj[1,:,:,ind:ind+model_length], edges[1]))
                    en[2].append(model(traj[2,:,:,ind:ind+model_length], edges[2]))
            if writer and not training:
                writer.add_figure('Test_Energy_During_MPPI_%s' %string, graph_time(en[0]), epoch)
            if display and not training:
                for j in range(3):
                    print(en[j][-1])
                if show_graph:
                    graph_time(en[0])

    if training:
        traj_copy = traj.clone()
        traj.requires_grad = True
        
        noise = torch.normal(0, std_dev, (num_samples, num_random, num_channels, model_length))
        noise[:,0] = torch.zeros(num_channels, model_length)
        samples = traj_copy + noise
        energy = torch.mul(model(samples.reshape(-1, num_channels, model_length), edges.unsqueeze(1).expand(-1, num_random, -1, -1).reshape(-1, n, n)). \
                           view(num_samples, num_random, 1), temperature)
        weights = F.softmin(energy, dim=1)
        traj = torch.unsqueeze((samples.view(num_samples, num_random, num_channels*model_length) * \
                                weights).sum(dim=1).view(num_samples, num_channels, model_length), 1)
        if display or writer:
            en[0].append(model(traj[0,:,:,ind:ind+model_length], edges[0]))
            en[1].append(model(traj[1,:,:,ind:ind+model_length], edges[1]))
            en[2].append(model(traj[2,:,:,ind:ind+model_length], edges[2]))
            
        if writer:
            for j in range(3):
                writer.add_figure('Energy_During_MPPI_%d' %(j+1), graph_time(en[j]), epoch)
        if display:
            for j in range(3):
                print(en[j][-1])
            if show_graph:
                graph_time(en[0])

    traj = torch.squeeze(traj, 1)

    if buffer != None:
        buffer.add(to_numpy(traj.detach()), to_numpy(edges))

    if writer and not training:
        for i in range(4):
            mean_energy = sum([torch.mean(model(traj[i:i+1,:,ind-model_length+1:ind+1].detach(), edges[i:i+1])) \
                               for ind in range(model_length, length)]) / (length - model_length)
            if length == 15:
                tag = 'Graph_%d_%s' %(i+1, string)
                if dist < 4:
                    border = 3
                else:
                    border = 5
            else:
                tag = 'Sample_%d_%s' %(i+1, string)
                border = 5
            if position == 0:
                writer.add_figure(tag, graph(traj[i].detach().cpu(), border=border, energy=mean_energy, training=True), epoch)

    if training:
        return traj, edges
    if random:
        return traj, init_ind
    return traj

def sample_langevin(trajectory_model, interaction_model, initial_loc, initial_vel, initial_edges, \
                    num_samples=256, length=None, random=False, display=False, steps=300, lam=0.0000001):
    model_length = trajectory_model.length
    num_channels = 20
    if not length:
        length = model_length

    traj, edges = initialize(initial_loc, initial_vel, initial_edges, model_length, length, num_channels, num_samples, random)

    traj.requires_grad_(True)

    lam = 0.00001

    for ind in range(model_length, length):
        en = []
        #print(ind)
        for i in range(steps):
            noise = torch.normal(0, lam, (num_samples, num_channels, model_length))
            energy = torch.zeros(1)
            for j in range(5):
                energy = energy + torch.mul(trajectory_model(traj[:,4*j:4*j+4,ind-model_length+1:ind+1]), lam/2)
            for j in range(5):
                for k in range(j+1, 5):
                    energy = energy + edges[:,j:j+1,k:k+1] * torch.mul(interaction_model(torch.cat((traj[:,4*j:4*j+4,ind-model_length+1:ind+1], \
                                    traj[:,4*k:4*k+4,ind-model_length+1:ind+1]), 1)), lam/2)

            energy.backward()
            traj.requires_grad_(False)
            if torch.isnan(traj.grad).any():
                print(i)
                break
            traj_grad = torch.autograd.grad([energy.sum()], [trajectory])[0]
            #traj[:,:,ind-model_length+1:ind+1] = traj[:,:,ind-model_length+1:ind+1] - traj.grad[:,:,ind-model_length+1:ind+1] + noise
            traj.requires_grad_(True)
            traj.retain_grad()
            
            if display:
                en.append(energy)
        if display:
            print(en[-1])
            graph_time(en)
            
    return traj

def inference(model, traj, n, display=False, cutoff='mean'):
    with torch.no_grad():
        model_length = model.length
        num_traj = len(traj)
        length = len(traj[0][0])
        
        edges = torch.zeros(num_traj, n, n)
        
        c1 = torch.arange(n).repeat_interleave(n)
        c2 = torch.arange(n).repeat(n)

        for ind in range(model_length, length):
            edges = edges - model.int_energy(torch.cat([torch.cat((traj[:,4*c1[i]:4*c1[i]+4,ind-model_length+1:ind+1], \
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

def test_prediction(model, test_loc, test_vel, test_edges, n, writer=None, epoch=0, \
                    num_samples=30, length=15, mppi_steps=100, std_dev=0.1, temperature=10, dist=1, display=False):
    criterion = nn.MSELoss(reduction = 'none')

    for inertia in [True, False]:
        for position in [0, 20, 40, 50, 60, 70, 80]:
            traj, ind = sample(model, test_loc, test_vel, test_edges, n, writer=writer, epoch=epoch, num_samples=num_samples, length=length, \
                               mppi_steps=mppi_steps, std_dev=std_dev, temperature=temperature, random=True, inertia=inertia, dist=dist, position=position)

            
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
                    writer.add_scalar('%s_Position_Error_Step_%d' %(string, position+11+i), position_loss[i], epoch)
                    writer.add_scalar('%s_Error_Step_%d' %(string, position+11+i), loss[i], epoch)

    if display and epoch == 0:
        initialization = initialize(test_loc, test_vel, test_edges, model.length, length, 20, num_samples, inertia=True, dist=dist)[0]

        position_loss_init = criterion(initialization.reshape(num_samples, 2*n, -1)[:,::2].reshape(num_samples, 2*n, -1), t)
        position_loss_init = torch.mean(position_loss_init, (0, 1))

        loss_init = criterion(initialization, torch.cat((t.reshape(num_samples, n, -1), v.reshape(num_samples, n, -1)), 2).reshape(num_samples, 4*n, -1))
        loss_init = torch.mean(loss_init, (0, 1))
        
        for i in range(10, length):
            print('Position Error At Step %d: ' %(i+1), position_loss_init[i].item())

        for i in range(10, length):
            print('Error At Step %d: ' %(i+1), loss_init[i].item())

def test_inference(model, test_loc, test_vel, test_edges, n, num_to_test=1000, cutoff='Mean', writer=None, name='Test', epoch=0, training=False):
    test_loc = to_tensor(transform(test_loc, n)[:num_to_test])
    test_vel = to_tensor(transform(test_vel, n)[:num_to_test])
    test_edges = to_tensor(test_edges[:num_to_test,:,:])

    traj = torch.cat((test_loc.view(num_to_test, n, -1), test_vel.view(num_to_test, n, -1)), 2).reshape(num_to_test, 4*n, -1)
    guess_edges = inference(model, traj, n, cutoff=cutoff)

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

def graph(traj, n=5, length=None, border=5, energy=None, graph_vel=False, includes_vel=True, transformed=True, training=False, ground_truth=False, save=False):
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

    if energy:
        title = title + ' (E: %.4f)' %energy

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

def graph_time(traj, title='Energy Over Time', training=False):
    fig, ax = plt.subplots()
    
    t = range(len(traj))

    ax.plot(t, traj)
    ax.plot(t, traj, '.')

    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
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

        #mean_energy = sum([torch.mean(model(traj[:,:,ind-model_length+1:ind+1], edges)) \
         #                  for ind in range(model_length, length)]) / (length - model_length)

        fig = graph(traj.squeeze(0).cpu(), training=True, ground_truth=True)

        for string in ['Inertia', 'Random']:
            writer.add_figure('Sample_%d_%s' %(i+1, string), fig)
        
def main():
    length = 10
    
    train_loc = np.load(train_loc_file, allow_pickle=True)
    train_vel = np.load(train_vel_file, allow_pickle=True)
    train_edges = np.load(train_edges_file, allow_pickle=True)
    test_loc = np.load(test_loc_file, allow_pickle=True)
    test_vel = np.load(test_vel_file, allow_pickle=True)
    test_edges = np.load(test_edges_file, allow_pickle=True)

    train_trajectory_dataset = Trajectory_Data(train_loc, train_vel)
    train_trajectory_dataloader = DataLoader(train_trajectory_dataset, batch_size=256, shuffle=True, drop_last=True)
    train_interaction_dataset = Interaction_Data(train_loc, train_vel, train_edges)
    train_interaction_dataloader = DataLoader(train_interaction_dataset, batch_size=256, shuffle=True, drop_last=True)

    test_trajectory_dataset = Trajectory_Data(test_loc, test_vel, length)
    test_trajectory_dataloader = DataLoader(test_trajectory_dataset, batch_size=256, shuffle=True, drop_last=True)
    test_interaction_dataset = Interaction_Data(test_loc, test_vel, train_edges, length)
    test_interaction_dataloader = DataLoader(test_interaction_dataset, batch_size=256, shuffle=True, drop_last=True)

    trajectory_model = Trajectory_Model(length)
    interaction_model = Interaction_Model(length)
    trajectory_model_old = Trajectory_Model()
    interaction_model_old = Interaction_Model()

    trajectory_model.load_state_dict(torch.load(PATH1, map_location=torch.device('cpu')))
    interaction_model.load_state_dict(torch.load(PATH2, map_location=torch.device('cpu')))
    trajectory_model_old.load_state_dict(torch.load(PATH1_OLD, map_location=torch.device('cpu')))
    interaction_model_old.load_state_dict(torch.load(PATH2_OLD, map_location=torch.device('cpu')))
    #train(trajectory_model, train_trajectory_dataloader, test_loc, test_vel, test_edges)
    #torch.save(trajectory_model.state_dict(), PATH1)
    #train(interaction_model, train_interaction_dataloader, test_loc, test_vel, test_edges)
    #torch.save(interaction_model.state_dict(), PATH2)
    
    for traj in np.load(input_file, allow_pickle=True):
       graph(traj[:,:100])
    
if __name__ == '__main__':
    main()
