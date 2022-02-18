import numpy as np
import math
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from linear_trajectories import Net, Trajectory_Data, linear, circular, helix, random, to_tensor, graph_time

PATH_L = '../../../Desktop/linear_trajectory_net.pth'
PATH_C = '../../../Desktop/circular_trajectory_net.pth'
data_file = '../../../Desktop/parallel_trajectory_data.npy'
input_file = '../../../Desktop/parallel_trajectory.npy'
PATH_COM = '../../../Desktop/fixed_composed_trajectory_net.pth'
PATH_PAR_L = '../../../Desktop/parallel_l_trajectory_net.pth'
PATH_PAR_C = '../../../Desktop/parallel_c_trajectory_net.pth'

def composed(length=6, tensor=True, flat=False, unsqueeze=True, denom=6, rand=None, constrain_length=None): # rand<0.5 = linear first; rand>0.5 = circular first
    if rand == None:
        rand = torch.rand((1,))
    if constrain_length == None:
        constrain_length = length
        
    start = torch.mul(torch.add(torch.rand((2,)), -0.5), 2)
    end = start[:]
    while min(abs(torch.div(end - start, constrain_length-1))) < 0.003:
        velocity = torch.mul(torch.add(torch.rand((2,)), -0.5), 0.04) # -0.02 to 0.02
        end = torch.clamp(start + torch.mul(velocity, constrain_length-1), -1, 1)
    
    traj = torch.zeros(4, length)
    traj[0:1,:] = torch.linspace(start[0], end[0], constrain_length)[:length]
    traj[1:2,:] = torch.linspace(start[1], end[1], constrain_length)[:length]
    traj[2:3,:] = torch.linspace(start[0], end[0], constrain_length)[:length]
    traj[3:4,:] = torch.linspace(start[1], end[1], constrain_length)[:length]
    
    radius = torch.zeros(1)
    while radius < 0.003:
        radius = torch.mul(torch.rand((1,)), 0.05)
    start_angle = torch.mul(torch.rand((1,)), 2 * math.pi) # 0 to 2pi
    ang_velocity = torch.zeros(1)
    while abs(ang_velocity) < math.pi / 6:
        ang_velocity = torch.mul(torch.add(torch.rand((1,)), -0.5), 4 * math.pi / denom) # -2pi/denom to 2pi/denom

    if rand < 0.5:
        for i in range(length):
            traj[2][i] = traj[2][i] + radius * torch.cos(start_angle + torch.mul(ang_velocity, i))
            traj[3][i] = traj[3][i] + radius * torch.sin(start_angle + torch.mul(ang_velocity, i))
    else:
        for i in range(length):
            traj[0][i] = traj[0][i] + radius * torch.cos(start_angle + torch.mul(ang_velocity, i))
            traj[1][i] = traj[1][i] + radius * torch.sin(start_angle + torch.mul(ang_velocity, i))
        
    if flat:
        traj = torch.transpose(traj, 0, 1)
        traj = traj.flatten()
    if unsqueeze:
        traj = torch.unsqueeze(traj, 0)
    if tensor:
        return traj
    return traj.numpy().tolist()

def parallel_l(length=6, tensor=True, flat=False, unsqueeze=True, constrain_length=None):
    if constrain_length == None:
        constrain_length = length
        
    start = torch.mul(torch.add(torch.rand((2,)), -0.5), 2)
    end = start[:]
    while min(abs(torch.div(end - start, constrain_length-1))) < 0.003:
        velocity = torch.mul(torch.add(torch.rand((2,)), -0.5), 0.04) # -0.02 to 0.02
        end = torch.clamp(start + torch.mul(velocity, constrain_length-1), -1, 1)
    
    traj = torch.zeros(4, length)
    traj[0:1,:] = torch.linspace(start[0], end[0], constrain_length)[:length]
    traj[1:2,:] = torch.linspace(start[1], end[1], constrain_length)[:length]

    translation = torch.mul(torch.add(torch.rand((2,)), -0.5), 0.5) # -0.25 to 0.25
    while not torch.equal(torch.clamp(start + translation, -1, 1), start + translation) or not torch.equal(torch.clamp(end + translation, -1, 1), end + translation) or min(abs(translation)) < 0.01:
        translation = torch.mul(torch.add(torch.rand((2,)), -0.5), 0.5)
        
    for i in range(length):
        traj[2:,i:i+1] = traj[0:2,i:i+1] + torch.transpose(translation.unsqueeze(0), 0, 1)
        
    if flat:
        traj = torch.transpose(traj, 0, 1)
        traj = traj.flatten()
    if unsqueeze:
        traj = torch.unsqueeze(traj, 0)
    if tensor:
        return traj
    return traj.numpy().tolist()

def parallel_c(length=6, tensor=True, flat=False, unsqueeze=True, denom=6, constrain_length=None):
    if constrain_length == None:
        constrain_length = length
        
    start = torch.mul(torch.add(torch.rand((2,)), -0.5), 2)
    end = start[:]
    while min(abs(torch.div(end - start, constrain_length-1))) < 0.003:
        velocity = torch.mul(torch.add(torch.rand((2,)), -0.5), 0.04) # -0.02 to 0.02
        end = torch.clamp(start + torch.mul(velocity, constrain_length-1), -1, 1)
    
    traj = torch.zeros(4, length)
    traj[0:1,:] = torch.linspace(start[0], end[0], constrain_length)[:length]
    traj[1:2,:] = torch.linspace(start[1], end[1], constrain_length)[:length]

    translation = torch.mul(torch.add(torch.rand((2,)), -0.5), 0.5) # -0.25 to 0.25
    while not torch.equal(torch.clamp(start + translation, -1, 1), start + translation) or not torch.equal(torch.clamp(end + translation, -1, 1), end + translation) or min(abs(translation)) < 0.01:
        translation = torch.mul(torch.add(torch.rand((2,)), -0.5), 0.5)
    
    radius = torch.zeros(1)
    while radius < 0.003:
        radius = torch.mul(torch.rand((1,)), 0.05)
    start_angle = torch.mul(torch.rand((1,)), 2 * math.pi) # 0 to 2pi
    ang_velocity = torch.zeros(1)
    while abs(ang_velocity) < math.pi / 6:
        ang_velocity = torch.mul(torch.add(torch.rand((1,)), -0.5), 4 * math.pi / denom) # -2pi/denom to 2pi/denom
        
    for i in range(length):
        traj[0][i] = traj[0][i] + radius * torch.cos(start_angle + torch.mul(ang_velocity, i))
        traj[1][i] = traj[1][i] + radius * torch.sin(start_angle + torch.mul(ang_velocity, i))
        traj[2:,i:i+1] = traj[0:2,i:i+1] + torch.transpose(translation.unsqueeze(0), 0, 1)
        
    if flat:
        traj = torch.transpose(traj, 0, 1)
        traj = traj.flatten()
    if unsqueeze:
        traj = torch.unsqueeze(traj, 0)
    if tensor:
        return traj
    return traj.numpy().tolist()

def triple(length=6, tensor=True, flat=False, unsqueeze=True, denom=6, constrain_length=None):
    if constrain_length == None:
        constrain_length = length
        
    start = torch.mul(torch.add(torch.rand((2,)), -0.5), 2)
    end = start[:]
    while min(abs(torch.div(end - start, constrain_length-1))) < 0.003:
        velocity = torch.mul(torch.add(torch.rand((2,)), -0.5), 0.04) # -0.02 to 0.02
        end = torch.clamp(start + torch.mul(velocity, constrain_length-1), -1, 1)
    
    traj = torch.zeros(6, length)
    traj[0:1,:] = torch.linspace(start[0], end[0], constrain_length)[:length]
    traj[1:2,:] = torch.linspace(start[1], end[1], constrain_length)[:length]

    translation = torch.mul(torch.add(torch.rand((2,)), -0.5), 0.5) # -0.25 to 0.25
    while not torch.equal(torch.clamp(start + translation, -1, 1), start + translation) or not torch.equal(torch.clamp(end + translation, -1, 1), end + translation) or min(abs(translation)) < 0.01:
        translation = torch.mul(torch.add(torch.rand((2,)), -0.5), 0.5)
    
    radius = torch.zeros(1)
    while radius < 0.003:
        radius = torch.mul(torch.rand((1,)), 0.05)
    start_angle = torch.mul(torch.rand((1,)), 2 * math.pi) # 0 to 2pi
    ang_velocity = torch.zeros(1)
    while abs(ang_velocity) < math.pi / 6:
        ang_velocity = torch.mul(torch.add(torch.rand((1,)), -0.5), 4 * math.pi / denom) # -2pi/denom to 2pi/denom
        
    for i in range(length):
        traj[2][i] = traj[0][i] + radius * torch.cos(start_angle + torch.mul(ang_velocity, i))
        traj[3][i] = traj[1][i] + radius * torch.sin(start_angle + torch.mul(ang_velocity, i))
        traj[4:,i:i+1] = traj[2:4,i:i+1] + torch.transpose(translation.unsqueeze(0), 0, 1)
        
    if flat:
        traj = torch.transpose(traj, 0, 1)
        traj = traj.flatten()
    if unsqueeze:
        traj = torch.unsqueeze(traj, 0)
    if tensor:
        return traj
    return traj.numpy().tolist()

def quint(length=6, tensor=True, flat=False, unsqueeze=True, denom=6, constrain_length=None):
    if constrain_length == None:
        constrain_length = length
        
    start = torch.mul(torch.add(torch.rand((2,)), -0.5), 2)
    end = start[:]
    while min(abs(torch.div(end - start, constrain_length-1))) < 0.003:
        velocity = torch.mul(torch.add(torch.rand((2,)), -0.5), 0.04) # -0.02 to 0.02
        end = torch.clamp(start + torch.mul(velocity, constrain_length-1), -1, 1)
    
    traj = torch.zeros(10, length)
    traj[0:1,:] = torch.linspace(start[0], end[0], constrain_length)[:length]
    traj[1:2,:] = torch.linspace(start[1], end[1], constrain_length)[:length]

    translation1 = torch.mul(torch.add(torch.rand((2,)), -0.5), 0.5) # -0.25 to 0.25
    while not torch.equal(torch.clamp(start + translation1, -1, 1), start + translation1) or not torch.equal(torch.clamp(end + translation1, -1, 1), end + translation1) or min(abs(translation1)) < 0.01:
        translation1 = torch.mul(torch.add(torch.rand((2,)), -0.5), 0.5)

    translation2 = torch.mul(torch.add(torch.rand((2,)), -0.5), 0.5) # -0.25 to 0.25
    while not torch.equal(torch.clamp(start + translation2, -1, 1), start + translation2) or not torch.equal(torch.clamp(end + translation2, -1, 1), end + translation2) or min(abs(translation2)) < 0.01:
        translation2 = torch.mul(torch.add(torch.rand((2,)), -0.5), 0.5)
    
    radius = torch.zeros(1)
    while radius < 0.003:
        radius = torch.mul(torch.rand((1,)), 0.05)
    start_angle = torch.mul(torch.rand((1,)), 2 * math.pi) # 0 to 2pi
    ang_velocity = torch.zeros(1)
    while abs(ang_velocity) < math.pi / 6:
        ang_velocity = torch.mul(torch.add(torch.rand((1,)), -0.5), 4 * math.pi / denom) # -2pi/denom to 2pi/denom
        
    for i in range(length):
        traj[2:4,i:i+1] = traj[:2,i:i+1] + torch.transpose(translation1.unsqueeze(0), 0, 1)
        traj[4:6,i:i+1] = traj[:2,i:i+1] + torch.transpose(translation2.unsqueeze(0), 0, 1)
        traj[6][i] = traj[0][i] + radius * torch.cos(start_angle + torch.mul(ang_velocity, i))
        traj[7][i] = traj[1][i] + radius * torch.sin(start_angle + torch.mul(ang_velocity, i))
        traj[8][i] = traj[2][i] + radius * torch.cos(start_angle + torch.mul(ang_velocity, i))
        traj[9][i] = traj[3][i] + radius * torch.sin(start_angle + torch.mul(ang_velocity, i))
        
    if flat:
        traj = torch.transpose(traj, 0, 1)
        traj = traj.flatten()
    if unsqueeze:
        traj = torch.unsqueeze(traj, 0)
    if tensor:
        return traj
    return traj.numpy().tolist()

def generate_data(num_samples=1000, length=6, save=False, flat=False, file=data_file):
    comp = [composed(length, tensor=False, flat=flat, unsqueeze=False) for i in range(num_samples)]
    rand = [random(length, composed=True, tensor=False, flat=flat, unsqueeze=False) for i in range(num_samples)]
    trajs = [None for i in range(2 * num_samples)]
    trajs[::2] = comp
    trajs[1::2] = rand
    labels = [(0 if i % 2 == 0 else 1) for i in range(2 * num_samples)]
    data = [trajs, labels]
    if save:
        np.save(file, data)
    return data

def generate_parallel(shape, num_samples=1000, length=6, save=False, flat=False, file=data_file):
    if shape == 'l':
        par = [parallel_l(length, tensor=False, flat=flat, unsqueeze=False) for i in range(num_samples)]
    elif shape == 'c':
        par = [parallel_c(length, tensor=False, flat=flat, unsqueeze=False) for i in range(num_samples)]
    rand = [random(length, composed=True, tensor=False, flat=flat, unsqueeze=False) for i in range(num_samples)]
    trajs = [None for i in range(2 * num_samples)]
    trajs[::2] = par
    trajs[1::2] = rand
    labels = [(0 if i % 2 == 0 else 1) for i in range(2 * num_samples)]
    data = [trajs, labels]
    if save:
        np.save(file, data)
    return data

class Composed_Net(nn.Module):
    def __init__(self, length=6):
        super(Composed_Net, self).__init__()
        self.length = length
        self.conv = nn.Conv1d(4, 32, 3)
        self.fc1 = nn.Linear(32 * (length-2), 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(net, net_l, net_c, dataloader):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(1):
        #print(epoch)
        for data, data_labels in dataloader:
            data_output = net(data)

            samples = sample(net, net_l, net_c, 256)
            sample_labels = torch.ones(256, 1)
            sample_output = net(samples)
            
            loss = criterion(data_output, data_labels) + criterion(sample_output, sample_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def sample(net, net_l, net_c, num_samples=256, length=None):
    with torch.no_grad():
        net_length = net.length
        if length == None:
            length = net_length
        traj = torch.mul(torch.add(torch.rand((num_samples, 4, length)), -0.5), 2)
        traj = torch.unsqueeze(traj, 1)

        num_random = 100
        #rand = torch.round(torch.rand((num_samples, 1)))
        #rand = torch.mul(rand, torch.ones(num_random)).view(-1, 1)
        
        for ind in range(0, length - net_length + 1, 2):
            for i in range(200):
                noise = torch.normal(0, 0.01, (num_samples, num_random, 4, net_length))
                noise[:,0] = torch.zeros(4, net_length)
                samples = traj[:,:,:,ind:ind+net_length] + noise
                #energy = net(samples.view(-1, 4, net_length)) + rand * (net_c(samples[:,:,:2,:].reshape(-1, 2, net_length)) + net_l(samples[:,:,2:4,:].reshape(-1, 2, net_length))) + \
                 #   (1-rand) * (net_l(samples[:,:,:2,:].reshape(-1, 2, net_length)) + net_c(samples[:,:,2:4,:].reshape(-1, 2, net_length)))
                energy = net(samples.view(-1, 4, net_length)) + net_l(samples[:,:,:2,:].reshape(-1, 2, net_length)) + net_c(samples[:,:,2:4,:].reshape(-1, 2, net_length))
                weights = F.softmin(energy.view(num_samples, num_random, 1), dim=1)
                traj[:,:,:,ind:ind+net_length] = torch.unsqueeze((samples.view(num_samples, num_random, 4*net_length) * weights).sum(dim=1).view(num_samples, 4, net_length), 1)
                
        return torch.squeeze(traj, 1).detach()

##    traj = torch.zeros(0)
##    with torch.no_grad():
##        generated = sample_without_batch(net, net_l, net_c, 3, 100)
##        for i in range(3):
##            for j in range(4, 388, 4):
##                traj = torch.cat((traj, torch.unsqueeze(generated[i][j:j+16], 0)))
##    return traj

def sample_without_batch(shape, net, net_l, net_c, num_samples=256, length=None):
    with torch.no_grad():
        net_length = net.length
        if length == None:
            length = net_length
        traj = torch.mul(torch.add(torch.rand((num_samples, 4, length)), -0.5), 2)
        #rand = torch.round(torch.rand((num_samples, 1)))
        for i in range(num_samples):
            if shape == 'l':
                traj[i][:,:net_length] = parallel_l(net_length, constrain_length=length)
            elif shape == 'c':
                traj[i][:,:net_length] = parallel_c(net_length, constrain_length=length)            
        traj = torch.unsqueeze(traj, 1)

        num_random = 100
        #rand = torch.mul(rand, torch.ones(num_random)).view(-1, 1)
        
        for ind in range(net_length, length):
            #print(ind)
            #en = []
            for i in range(200):
                noise = torch.cat((torch.zeros(num_samples, num_random, 4, net_length-1), torch.normal(0, 0.01, (num_samples, num_random, 4, 1))), dim=3)
                noise[:,0] = torch.zeros(4, net_length)
                samples = traj[:,:,:,ind-net_length+1:ind+1] + noise
                #energy = net(samples.reshape(-1, 4, net_length)) + rand * (net_c(samples[:,:,:2,:].reshape(-1, 2, net_length)) + net_l(samples[:,:,2:4,:].reshape(-1, 2, net_length))) + \
                 #   (1-rand) * (net_l(samples[:,:,:2,:].reshape(-1, 2, net_length)) + net_c(samples[:,:,2:4,:].reshape(-1, 2, net_length)))
                energy = net(samples.reshape(-1, 4, net_length)) + net_l(samples[:,:,:2,:].reshape(-1, 2, net_length)) + net_c(samples[:,:,2:4,:].reshape(-1, 2, net_length))
                weights = F.softmin(energy.view(num_samples, num_random, 1), dim=1)
                traj[:,:,:,ind-net_length+1:ind+1] = torch.unsqueeze((samples.view(num_samples, num_random, 4*net_length) * weights).sum(dim=1).view(num_samples, 4, net_length), 1)
##                s = net(traj[:,:,:,ind-net_length+1:ind+1].squeeze(0)) + net_l(traj[:,:,:2,ind-net_length+1:ind+1].squeeze(0)) + net_c(traj[:,:,2:4,ind-net_length+1:ind+1].squeeze(0))
##                en.append(s)
##            graph_time(en)
            
        return torch.squeeze(traj, 1).detach()

def sample_triple(net_l, net_c, net_com, net_par, num_samples=256, length=None):
    with torch.no_grad():
        net_length = net_l.length
        if length == None:
            length = net_length
        traj = torch.mul(torch.add(torch.rand((num_samples, 6, length)), -0.5), 2)
        for i in range(num_samples):
            traj[i][:,:net_length] = triple(net_length, constrain_length=length)
        traj = torch.unsqueeze(traj, 1)

        num_random = 100
        
        for ind in range(net_length, length):
            #print(ind)
            en = []
            for i in range(200):
                noise = torch.cat((torch.zeros(num_samples, num_random, 6, net_length-1), torch.normal(0, 0.01, (num_samples, num_random, 6, 1))), dim=3)
                noise[:,0] = torch.zeros(6, net_length)
                samples = traj[:,:,:,ind-net_length+1:ind+1] + noise
                #energy = net_com(samples[:,:,:4,:].reshape(-1, 4, net_length)) + net_par(samples[:,:,2:,:].reshape(-1, 4, net_length)) + \
                 #        net_l(samples[:,:,:2,:].reshape(-1, 2, net_length)) + net_c(samples[:,:,2:4,:].reshape(-1, 2, net_length)) + \
                  #       net_c(samples[:,:,4:6,:].reshape(-1, 2, net_length))
                energy = torch.cat((net_com(samples[:,:,:4,:].reshape(-1, 4, net_length)), net_par(samples[:,:,2:,:].reshape(-1, 4, net_length))), dim=1)
                energy = (energy * F.softmax(energy, dim=1)).sum(dim=1).view(-1, 1)
                energy = energy + net_l(samples[:,:,:2,:].reshape(-1, 2, net_length)) + net_c(samples[:,:,2:4,:].reshape(-1, 2, net_length)) + \
                         net_c(samples[:,:,4:6,:].reshape(-1, 2, net_length))
                weights = F.softmin(energy.view(num_samples, num_random, 1), dim=1)
                traj[:,:,:,ind-net_length+1:ind+1] = torch.unsqueeze((samples.view(num_samples, num_random, 6*net_length) * weights).sum(dim=1).view(num_samples, 6, net_length), 1)
##                s = net_com(traj[:,:,:4,ind-net_length+1:ind+1].squeeze(0)) + net_par(traj[:,:,2:,ind-net_length+1:ind+1].squeeze(0)) + \
##                    net_l(traj[:,:,:2,ind-net_length+1:ind+1].squeeze(0)) + net_c(traj[:,:,2:4,ind-net_length+1:ind+1].squeeze(0)) + \
##                    net_c(traj[:,:,4:6,ind-net_length+1:ind+1].squeeze(0))
##                if i > 100:
##                    en.append(s)
##            graph_time(en)
##            print(net_com(traj[:,:,:4,ind-net_length+1:ind+1].squeeze(0)))
##            print(net_par(traj[:,:,2:,ind-net_length+1:ind+1].squeeze(0)))
##            print(s)
##            print()

        return torch.squeeze(traj, 1).detach()

def sample_quint(net_l, net_c, net_com, net_par_l, net_par_c, num_samples=256, length=None):
    with torch.no_grad():
        net_length = net_l.length
        if length == None:
            length = net_length
        traj = torch.mul(torch.add(torch.rand((num_samples, 10, length)), -0.5), 2)
        for i in range(num_samples):
            traj[i][:,:net_length] = quint(net_length, constrain_length=length)
        traj = torch.unsqueeze(traj, 1)

        num_random = 100
        
        for ind in range(net_length, length):
            #print(ind)
            for i in range(1000):
                noise = torch.cat((torch.zeros(num_samples, num_random, 10, net_length-1), torch.normal(0, 0.01, (num_samples, num_random, 10, 1))), dim=3)
                noise[:,0] = torch.zeros(10, net_length)
                samples = traj[:,:,:,ind-net_length+1:ind+1] + noise
                energy = torch.cat((net_par_l(samples[:,:,:4,:].reshape(-1, 4, net_length)), net_par_l(samples[:,:,2:6,:].reshape(-1, 4, net_length)), \
                                    net_par_c(samples[:,:,6:,:].reshape(-1, 4, net_length)), \
                                    net_com(torch.cat((samples[:,:,:2,:], samples[:,:,6:8,:]), dim=2).reshape(-1, 4, net_length)), \
                                    net_com(torch.cat((samples[:,:,2:4,:], samples[:,:,8:,:]), dim=2).reshape(-1, 4, net_length))), dim=1)
                energy = (energy * F.softmax(energy, dim=1)).sum(dim=1).view(-1, 1)
                energy = energy + net_l(samples[:,:,:2,:].reshape(-1, 2, net_length)) + net_l(samples[:,:,2:4,:].reshape(-1, 2, net_length)) + \
                         net_l(samples[:,:,4:6,:].reshape(-1, 2, net_length)) + net_c(samples[:,:,6:8,:].reshape(-1, 2, net_length)) + \
                         net_c(samples[:,:,8:,:].reshape(-1, 2, net_length))
                weights = F.softmin(energy.view(num_samples, num_random, 1), dim=1)
                traj[:,:,:,ind-net_length+1:ind+1] = torch.unsqueeze((samples.view(num_samples, num_random, 10*net_length) * weights).sum(dim=1).view(num_samples, 10, net_length), 1)

        return torch.squeeze(traj, 1).detach()

def sample_separate(net_l, net_c, num_samples=256, length=4):
    traj = torch.mul(torch.add(torch.rand((num_samples, length, 4)), -0.5), 2)
    for i in range(num_samples):
        start = torch.mul(torch.add(torch.rand((2,)), -0.5), 2)
        end = torch.mul(torch.add(torch.rand((2,)), -0.5), 2)
        traj[i][:4,0:1] = torch.linspace(start[0], end[0], length)[:4].view(4, 1)
        traj[i][:4,1:2] = torch.linspace(start[1], end[1], length)[:4].view(4, 1)
        c = circular(denom = min(length * 0.5, 20))
        traj[i][:4,2:3] = c[::2].view(4, 1)
        traj[i][:4,3:4] = c[1::2].view(4, 1)
    traj = torch.unsqueeze(traj, 1)
    
    for ind in range(4, length):
        for i in range(20):
            num_random = 100
            noise = torch.cat((torch.zeros(num_samples, num_random, 6), torch.normal(0, 0.1, (num_samples, num_random, 2))), dim=2).view(num_samples, num_random, 4, 2)
            samples = torch.flatten(traj[:,:,ind-3:ind+1,0:2] + noise, start_dim=2)
            energy = net_l(samples)
            weights = F.softmin(energy, dim=1)
            traj[:,:,ind-3:ind+1,0:2] = torch.unsqueeze((samples * weights).view(num_samples, num_random, 4, 2).sum(dim=1), 1)

            noise = torch.cat((torch.zeros(num_samples, num_random, 6), torch.normal(0, 0.1, (num_samples, num_random, 2))), dim=2).view(num_samples, num_random, 4, 2)
            samples = torch.flatten(traj[:,:,ind-3:ind+1,2:4] + noise, start_dim=2)
            energy = net_c(samples)
            weights = F.softmin(energy, dim=1)
            traj[:,:,ind-3:ind+1,2:4] = torch.unsqueeze((samples * weights).view(num_samples, num_random, 4, 2).sum(dim=1), 1)
            
    return torch.flatten(torch.squeeze(traj, 1), start_dim=1).detach()

def graph(traj, flat=False, save=False):
    fig, ax = plt.subplots()

    if flat:
        x1 = traj[::4]
        y1 = traj[1::4]
        x2 = traj[2::4]
        y2 = traj[3::4]
    else:
        x1 = traj[0]
        y1 = traj[1]
        x2 = traj[2]
        y2 = traj[3]

    ax.plot(x1, y1)
    ax.plot(x1, y1, '.')
    ax.plot(x1[:6], y1[:6], '.')

    ax.plot(x2, y2)
    ax.plot(x2, y2, '.')
    ax.plot(x2[:6], y2[:6], '.')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Composed Trajectory')

    y_range = max(max(y1), max(y2)) - min(min(y1), min(y2)) + 0.2
    x_range = y_range * fig.get_figwidth() / fig.get_figheight()

    y_mid = (min(min(y1), min(y2)) + max(max(y1), max(y2))) / 2
    x_mid = (min(min(x1), min(x2)) + max(max(x1), max(x2))) / 2

    plt.xticks(np.linspace(x_mid - x_range / 2, x_mid + x_range / 2, 11))
    plt.yticks(np.linspace(y_mid - y_range / 2, y_mid + y_range / 2, 11))

    if save:
        time = datetime.datetime.now()
        graph_file = 'composed %02d-%02d %02d-%02d-%02d.png' %(time.month, time.day, time.hour, time.minute, time.second)
        plt.savefig(graph_file)
    
    plt.show()

def graph_triple(traj, save=False):
    fig, ax = plt.subplots()

    x1 = traj[0]
    y1 = traj[1]
    x2 = traj[2]
    y2 = traj[3]
    x3 = traj[4]
    y3 = traj[5]

    ax.plot(x1, y1)
    ax.plot(x1, y1, '.')
    ax.plot(x1[:6], y1[:6], '.')

    ax.plot(x2, y2)
    ax.plot(x2, y2, '.')
    ax.plot(x2[:6], y2[:6], '.')

    ax.plot(x3, y3)
    ax.plot(x3, y3, '.')
    ax.plot(x3[:6], y3[:6], '.')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Composed Trajectory')

    y_range = max(max(y1), max(y2), max(y3)) - min(min(y1), min(y2), min(y3)) + 0.2
    x_range = y_range * fig.get_figwidth() / fig.get_figheight()

    y_mid = (min(min(y1), min(y2), min(y3)) + max(max(y1), max(y2), max(y3))) / 2
    x_mid = (min(min(x1), min(x2), min(x3)) + max(max(x1), max(x2), max(x3))) / 2

    plt.xticks(np.linspace(x_mid - x_range / 2, x_mid + x_range / 2, 11))
    plt.yticks(np.linspace(y_mid - y_range / 2, y_mid + y_range / 2, 11))

    if save:
        time = datetime.datetime.now()
        graph_file = 'composed %02d-%02d %02d-%02d-%02d.png' %(time.month, time.day, time.hour, time.minute, time.second)
        plt.savefig(graph_file)
    
    plt.show()

def graph_quint(traj, save=False):
    fig, ax = plt.subplots()

    x1 = traj[0]
    y1 = traj[1]
    x2 = traj[2]
    y2 = traj[3]
    x3 = traj[4]
    y3 = traj[5]
    x4 = traj[6]
    y4 = traj[7]
    x5 = traj[8]
    y5 = traj[9]

    ax.plot(x1, y1)
    ax.plot(x1, y1, '.')
    ax.plot(x1[:6], y1[:6], '.')

    ax.plot(x2, y2)
    ax.plot(x2, y2, '.')
    ax.plot(x2[:6], y2[:6], '.')

    ax.plot(x3, y3)
    ax.plot(x3, y3, '.')
    ax.plot(x3[:6], y3[:6], '.')

    ax.plot(x4, y4)
    ax.plot(x4, y4, '.')
    ax.plot(x4[:6], y4[:6], '.')

    ax.plot(x5, y5)
    ax.plot(x5, y5, '.')
    ax.plot(x5[:6], y5[:6], '.')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Composed Trajectory')

    y_range = max(max(y1), max(y2), max(y3), max(y4), max(y5)) - min(min(y1), min(y2), min(y3), min(y4), min(y5)) + 0.2
    x_range = y_range * fig.get_figwidth() / fig.get_figheight()

    y_mid = (min(min(y1), min(y2), min(y3), min(y3), min(y4), min(y5)) + max(max(y1), max(y2), max(y3), max(y4), max(y5))) / 2
    x_mid = (min(min(x1), min(x2), min(x3), min(x4), min(x5)) + max(max(x1), max(x2), max(x3), max(x4), max(x5))) / 2

    plt.xticks(np.linspace(x_mid - x_range / 2, x_mid + x_range / 2, 11))
    plt.yticks(np.linspace(y_mid - y_range / 2, y_mid + y_range / 2, 11))

    if save:
        time = datetime.datetime.now()
        graph_file = 'composed %02d-%02d %02d-%02d-%02d.png' %(time.month, time.day, time.hour, time.minute, time.second)
        plt.savefig(graph_file)
    
    plt.show()

def main():
    #data = generate_data(10240, save=True)
    #data = np.load(data_file, allow_pickle=True)
    #dataset = Trajectory_Data(data)
    #dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)
    net_com = Composed_Net()
    net_par_l = Composed_Net()
    net_par_c = Composed_Net()
    net_l = Net()
    net_c = Net()
    net_com.load_state_dict(torch.load(PATH_COM, map_location=torch.device('cpu')))
    net_par_l.load_state_dict(torch.load(PATH_PAR_L, map_location=torch.device('cpu')))
    net_par_c.load_state_dict(torch.load(PATH_PAR_C, map_location=torch.device('cpu')))
    net_l.load_state_dict(torch.load(PATH_L, map_location=torch.device('cpu')))
    net_c.load_state_dict(torch.load(PATH_C, map_location=torch.device('cpu')))
    #train(net, net_l, net_c, dataloader)
    #torch.save(net.state_dict(), PATH)
if __name__ == '__main__':
    main()
