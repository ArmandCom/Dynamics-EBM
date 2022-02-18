import numpy as np
import math
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from linear_trajectories import Net, Trajectory_Data, linear, circular, helix, random, to_tensor, total_energy, sample_with_batch, graph_time

data_file = '../../../Desktop/circular_trajectory_data.npy'
input_file = '../../../Desktop/circular_trajectory.npy'
PATH = '../../../Desktop/circular_trajectory_net.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def generate_data(num_samples=1000, length=6, save=False, flat=False, file=data_file):
    hel = [helix(length, tensor=False, flat=flat, unsqueeze=False) for i in range(num_samples)]
    rand = [random(length, tensor=False, flat=flat, unsqueeze=False) for i in range(num_samples)]
    trajs = [None for i in range(2 * num_samples)]
    trajs[::2] = hel
    trajs[1::2] = rand
    labels = [(0 if i % 2 == 0 else 1) for i in range(2 * num_samples)]
    data = [trajs, labels]
    if save:
        np.save(file, data)
    return data

def train(net, dataloader, net_small=None):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5):
        #print(epoch)
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
            #traj = (torch.mul(torch.add(torch.rand((num_samples, 1)), -0.5), 2) * torch.ones(1, 2 * length)).view(num_samples, 2, length)
        else:
            traj = sample_without_batch(net_small, num_samples, length)

        #graph(traj[0])
        traj = torch.unsqueeze(traj, 1)
        
        for ind in range(0, length - net_length + 1, 2):
            #en= []
            for i in range(10000):
                noise = torch.normal(0, 0.01, (num_samples, num_random, 2, net_length))
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

##    with torch.no_grad():
##        traj = torch.zeros(0)
##        generated = sample_without_batch(net, 3, 100)
##        for i in range(3):
##            for j in range(1, 95):
##                traj = torch.cat((traj, torch.unsqueeze(generated[i][:,j:j+6], 0)))
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
                traj[i][:,:net_length] = helix(net_length, constrain_length=length)
        else:
            traj = sample_without_batch(net_small, num_samples, length)
        traj = torch.unsqueeze(traj, 1)
        
        for ind in range(net_length, length):
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
                #en.append(net(traj[:,:,:,ind-net_length+1:ind+1].squeeze(0)))
            #graph_time(en)
        return torch.squeeze(traj, 1).detach()

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
    ax.set_title('Circular Trajectory')

    y_range = max(y) - min(y) + 0.2
    x_range = y_range * fig.get_figwidth() / fig.get_figheight()

    y_mid = (min(y) + max(y)) / 2
    x_mid = (min(x) + max(x)) / 2

    plt.xticks(np.linspace(x_mid - x_range / 2, x_mid + x_range / 2, 11))
    plt.yticks(np.linspace(y_mid - y_range / 2, y_mid + y_range / 2, 11))

    if save:
        time = datetime.datetime.now()
        graph_file = 'helix %02d-%02d %02d-%02d-%02d.png' %(time.month, time.day, time.hour, time.minute, time.second)
        plt.savefig(graph_file)
    
    plt.show()

def convex(traj):
    p1 = (traj[0], traj[1])
    p2 = (traj[2], traj[3])
    p3 = (traj[4], traj[5])
    p4 = (traj[6], traj[7])
    return (same_side(p1, p2, p3, p4) and same_side(p2, p3, p1, p4) and same_side(p3, p4, p1, p2)).item()

def same_side(p1, p2, p3, p4): # returns True if p3 and p4 are on the same side of the line connecting p1 and p2
    if p1[0] == p2[0]:
        return np.sign(p3[0] - p1[0]) == np.sign(p4[0] - p1[0])
    m = (p1[1] - p2[1]) / (p1[0] - p2[0])
    b = p1[1] - m * p1[0]
    return np.sign(m * p3[0] - p3[1] + b) == np.sign(m * p4[0] - p4[1] + b)

def main():
    #data = generate_data(10240, save=True)
    data = np.load(data_file, allow_pickle=True)
    dataset = Trajectory_Data(data)
    dataloader = DataLoader(dataset, num_workers=2, batch_size=256, shuffle=True, drop_last=True)
    net = Net()
    net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    train(net, dataloader)
    torch.save(net.state_dict(), PATH)

if __name__ == '__main__':
    main()
