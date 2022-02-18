import numpy as np
import math
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

data_file = '../../../Desktop/linear_trajectory_data.npy'
input_file = '../../../Desktop/linear_trajectory.npy'
PATH = '../../../Desktop/linear_trajectory_net.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def linear(length=6, tensor=True, flat=False, unsqueeze=True, constrain_length=None):
    if constrain_length == None:
        constrain_length = length
        
    start = torch.mul(torch.add(torch.rand((2,)), -0.5), 2)
    end = start[:]
    while min(abs(torch.div(end - start, constrain_length-1))) < 0.003:
        velocity = torch.mul(torch.add(torch.rand((2,)), -0.5), 0.04) # -0.02 to 0.02
        end = torch.clamp(start + torch.mul(velocity, constrain_length-1), -1, 1)
    traj = torch.zeros(2, length)
    traj[0:1,:] = torch.linspace(start[0], end[0], constrain_length)[:length]
    traj[1:2,:] = torch.linspace(start[1], end[1], constrain_length)[:length]
    if flat:
        traj = torch.transpose(traj, 0, 1)
        traj = traj.flatten()
    if unsqueeze:
        traj = torch.unsqueeze(traj, 0)
    if tensor:
        return traj
    return traj.numpy().tolist()

def circular(length=6, tensor=True, flat=False, unsqueeze=True, denom=6):
    center = torch.mul(torch.add(torch.rand((2,)), -0.5), 1.5)
    radius = torch.zeros(1)
    while radius < 0.01:
        radius = torch.mul(torch.rand((1,)), min(min(center + 1), min(1 - center)))
    start = torch.mul(torch.rand((1,)), 2 * math.pi) # 0 to 2pi
    velocity = torch.zeros(1)
    while abs(velocity) < math.pi / 20:
        velocity = torch.mul(torch.add(torch.rand((1,)), -0.5), 4 * math.pi / denom) # -2pi/denom to 2pi/denom
    traj = torch.zeros(2, length)
    for i in range(length):
        traj[0][i] = center[0] + radius * torch.cos(start + torch.mul(velocity, i))
        traj[1][i] = center[1] + radius * torch.sin(start + torch.mul(velocity, i))
    if flat:
        traj = torch.transpose(traj, 0, 1)
        traj = traj.flatten()
    if unsqueeze:
        traj = torch.unsqueeze(traj, 0)
    if tensor:
        return traj
    return traj.numpy().tolist()

def helix(length=6, tensor=True, flat=False, unsqueeze=True, denom=6, constrain_length=None):
    if constrain_length == None:
        constrain_length = length
        
    start = torch.mul(torch.add(torch.rand((2,)), -0.5), 2)
    end = start[:]
    while min(abs(torch.div(end - start, constrain_length-1))) < 0.003:
        velocity = torch.mul(torch.add(torch.rand((2,)), -0.5), 0.04) # -0.02 to 0.02
        end = torch.clamp(start + torch.mul(velocity, constrain_length-1), -1, 1)
    
    traj = torch.zeros(2, length)
    traj[0:1,:] = torch.linspace(start[0], end[0], constrain_length)[:length]
    traj[1:2,:] = torch.linspace(start[1], end[1], constrain_length)[:length]
    
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
        
    if flat:
        traj = torch.transpose(traj, 0, 1)
        traj = traj.flatten()
    if unsqueeze:
        traj = torch.unsqueeze(traj, 0)
    if tensor:
        return traj
    return traj.numpy().tolist()

def random(length=6, composed=False, tensor=True, flat=False, unsqueeze=True):
    if composed:
        traj = torch.mul(torch.add(torch.rand((4, length)), -0.5), 2)
    else:
        traj = torch.mul(torch.add(torch.rand((2, length)), -0.5), 2)
    if flat:
        traj = torch.transpose(traj, 0, 1)
        traj = traj.flatten()
    if unsqueeze:
        traj = torch.unsqueeze(traj, 0)
    if tensor:
        return traj
    return traj.numpy().tolist()

def generate_data(num_samples=1000, length=6, save=False, flat=False, file=data_file):
    lin = [linear(length, tensor=False, flat=flat, unsqueeze=False) for i in range(num_samples)]
    rand = [random(length, tensor=False, flat=flat, unsqueeze=False) for i in range(num_samples)]
    trajs = [None for i in range(2 * num_samples)]
    trajs[::2] = lin
    trajs[1::2] = rand
    labels = [(0 if i % 2 == 0 else 1) for i in range(2 * num_samples)]
    data = [trajs, labels]
    if save:
        np.save(file, data)
    return data

def to_tensor(array):
    return torch.from_numpy(np.array(array)).float()


class Net(nn.Module):
    def __init__(self, length=6):
        super(Net, self).__init__()
        self.length = length
        self.conv = nn.Conv1d(2, 16, 3)
        self.fc1 = nn.Linear(16 * (length-2), 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Trajectory_Data(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        return to_tensor(self.data[0][index]).to(device), torch.unsqueeze(to_tensor(self.data[1][index]), 0).to(device) # trajectory, label

def train(net, dataloader, net_small=None):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(10):
        print(epoch)
        for data, data_labels in dataloader:
            data_output = net(data)

            samples = sample(net, 256, net_small=net_small)
            sample_labels = torch.ones(256, 1)
            sample_output = net(samples)
            
            loss = criterion(data_output, data_labels) + criterion(sample_output, sample_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def sample(net, num_samples=256, length=None, net_small=None):
    with torch.no_grad():
        net_length = net.length
        num_random = 100
        if length == None:
            length = net_length
            
        if net_small == None:
            traj = torch.mul(torch.add(torch.rand((num_samples, 2, length)), -0.5), 2)
        else:
            traj = sample_without_batch(net_small, num_samples, length)

        traj = torch.unsqueeze(traj, 1)
        
        for ind in range(0, length - net_length + 1, 2):
            #en= []
            for i in range(5000):
                noise = torch.normal(0, 0.001, (num_samples, num_random, 2, net_length))
                noise[:,0] = torch.zeros(2, net_length)
                samples = traj[:,:,:,ind:ind+net_length] + noise
                energy = net(samples.view(-1, 2, net_length)).view(num_samples, num_random, 1)
                if net_small != None:
                    energy = energy + total_energy(net_small, samples.reshape(-1, 2, net_length)).view(num_samples, num_random, 1)
                weights = F.softmin(energy, dim=1)
                traj[:,:,:,ind:ind+net_length] = torch.unsqueeze((samples.view(num_samples, num_random, 2*net_length) * weights).sum(dim=1).view(num_samples, 2, net_length), 1)
                #en.append(net(traj[:,:,:,ind:ind+net_length].squeeze(0)))
            #graph_time(en)
                
        return torch.squeeze(traj, 1).detach()

##    traj = torch.zeros(0)
##    with torch.no_grad():
##        generated = sample_without_batch(net, 3, 100)
##        for i in range(3):
##            for j in range(2, 194, 2):
##                traj = torch.cat((traj, torch.unsqueeze(generated[i][j:j+8], 0)))
##    return traj

def sample_without_batch(net, num_samples=256, length=None, net_small=None):
    with torch.no_grad():
        net_length = net.length
        num_random = 100
        if length == None:
            length = net_length

        if net_small == None:
            traj = torch.mul(torch.add(torch.rand((num_samples, 2, length)), -0.5), 2)
            for i in range(num_samples):
                start = torch.mul(torch.add(torch.rand((2,)), -0.5), 2)
                end = torch.mul(torch.add(torch.rand((2,)), -0.5), 2)
                traj[i][0][:net_length] = torch.linspace(start[0], end[0], length)[:net_length]
                traj[i][1][:net_length] = torch.linspace(start[1], end[1], length)[:net_length]
        else:
            traj = sample_without_batch(net_small, num_samples, length)
        traj = torch.unsqueeze(traj, 1)

        start = (net_length if net_small == None else net_length-1)
        for ind in range(start, length):
            #en = []
            for i in range(100):
                noise = torch.cat((torch.zeros(num_samples, num_random, 2, net_length-1), torch.normal(0, 0.01, (num_samples, num_random, 2, 1))), dim=3)
                noise[:,0] = torch.zeros(2, net_length)
                samples = traj[:,:,:,ind-net_length+1:ind+1] + noise
                energy = net(samples.reshape(-1, 2, net_length)).view(num_samples, num_random, 1)
                if net_small != None:
                    energy = energy + total_energy(net_small, samples.reshape(-1, 2, net_length)).view(num_samples, num_random, 1)
                weights = F.softmin(energy, dim=1)
                traj[:,:,:,ind-net_length+1:ind+1] = torch.unsqueeze((samples.view(num_samples, num_random, 2*net_length) * weights).sum(dim=1).view(num_samples, 2, net_length), 1)
                #en.append(net(traj[:,:,:,ind:ind+net_length].squeeze(0)))
            #graph_time(en)
                
        return torch.squeeze(traj, 1).detach()

def sample_with_batch(net, num_samples=256, length=4, batch_size=10):
    traj = torch.unsqueeze(sample_without_batch(net, num_samples, length), 1)

    count = 0
    for ind in (list(range(2 * batch_size, 2 * length, 2)) + list(range(2 * batch_size, 2 * length, 2)[::-1])) * 10:
        for i in range(20):
            a = min(6, ind-2*batch_size)
            b = min(6, 2*length-2-ind)
            num_random = 10000
            noise = torch.cat((torch.zeros(num_samples, num_random, a), torch.normal(0, 0.5, (num_samples, num_random, 2 * batch_size)), torch.zeros(num_samples, num_random, b)), dim=2)
            samples = traj[:,:,ind-2*batch_size-a:ind+b] + noise
            energy = total_energy(net, samples)
            weights = F.softmin(energy, dim=1)
            traj[:,:,ind-2*batch_size-a:ind+b] = torch.unsqueeze((samples * weights).sum(dim=1), 1)
            
    return torch.squeeze(traj, 1).detach()

def total_energy(net, traj):
    net_length = net.length
    traj_length = len(traj[0][0])
    energy = torch.zeros(len(traj), 1)
    for ind in range(net_length+1, traj_length+1, net_length):
        energy = energy + net(traj[:,:,ind-net_length:ind])
    return energy
    
##    net_length = net.length
##    traj_length = len(traj[0][0])
##    energy = torch.zeros(len(traj), 1)
##    for ind in range(net_length, traj_length+1):
##        energy = energy + net(traj[:,:,ind-net_length:ind])
##    return energy

def graph(traj, flat=False, save=False):
    fig, ax = plt.subplots()

    if flat:
        x = traj[::2]
        y = traj[1::2]
    else:
        x = traj[0]
        y = traj[1]

    ax.plot(x, y)
    ax.plot(x, y, '.')
    ax.plot(x[:6], y[:6], '.')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Linear Trajectory')
    
    if save:
        time = datetime.datetime.now()
        graph_file = 'linear %02d-%02d %02d-%02d-%02d.png' %(time.month, time.day, time.hour, time.minute, time.second)
        plt.savefig(graph_file)

    plt.show()

def graph_time(traj, title='Energy Over Time'):
    fig, ax = plt.subplots()
    
    t = range(len(traj))

    ax.plot(t, traj)
    ax.plot(t, traj, '.')

    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title(title)
    
    plt.show()

def main():
    #data = generate_data(10240, save=True)
    data = np.load(data_file, allow_pickle=True)
    dataset = Trajectory_Data(data)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)
    net = Net()
    net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    train(net, dataloader)
    torch.save(net.state_dict(), PATH)

if __name__ == '__main__':
    main()
