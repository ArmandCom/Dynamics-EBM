import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
from os import path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from springs import *

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=5)
parser.add_argument('--num-train', type=int, default=50000)
parser.add_argument('--dist', type=int, default=1)
parser.add_argument('--model-length', type=int, default=10)
parser.add_argument('--units', type=int, default=128)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--convolute', default=False, action='store_true')
parser.add_argument('--no-convolute', dest='convolute', action='store_false')
parser.add_argument('--mppi-steps', type=int, default=200)
parser.add_argument('--std-dev', type=float, default=0.1)
parser.add_argument('--temperature', type=int, default=100)
parser.add_argument('--lr', default='5')
parser.add_argument('--anti', default=False, action='store_true')
parser.add_argument('--no-anti', dest='anti', action='store_false')
parser.add_argument('--inertia', default=False, action='store_true')
parser.add_argument('--no-inertia', dest='inertia', action='store_false')
parser.add_argument('--buffer', default=False, action='store_true')
parser.add_argument('--no-buffer', dest='buffer', action='store_false')
parser.add_argument('--start', type=int, default=1)
parser.add_argument('--end', type=int, default=15)
args = parser.parse_args()

n = args.n
num_train = args.num_train
dist = args.dist
length = args.model_length
units = args.units
activation = args.activation
convolute = args.convolute
mppi_steps = args.mppi_steps
std_dev = args.std_dev
temperature = args.temperature
lr = args.lr
anti = args.anti
inertia = args.inertia
start = args.start
end = args.end

if n == 2 and num_train == 50000:
    folder = ''
else:
    folder = '%d_' %num_train
folder = folder + 'optimized_%d_%d' %(n, dist)
if activation != 'relu':
    folder = folder + '_%s' %activation
if convolute:
    folder = folder + '_conv'
if anti:
    folder = folder + '_anti'
if args.buffer:
    folder = folder + '_persistent'
if inertia:
    folder = folder + '_inertia'
if units != 128:
    folder = folder + '_%d' %units
if lr == 'BCE':
    folder = folder + '_BCE/'
else:
    folder = folder + '_lr%s/' %lr
sub_folder = folder + 'temp%d_%d' %(temperature, mppi_steps)
if std_dev < 0.1:
    sub_folder = sub_folder + '_%.2f' %std_dev
elif std_dev >= 1:
    sub_folder = sub_folder + '_%d' %std_dev
else:
    sub_folder = sub_folder + '_%.1f' %std_dev
sub_folder = sub_folder + '/'
writer = SummaryWriter(sub_folder)
print(sub_folder)

train_loc_file = 'loc_train_springs%d_%d.npy' %(n, num_train)
train_vel_file = 'vel_train_springs%d_%d.npy' %(n, num_train)
train_edges_file = 'edges_train_springs%d_%d.npy' %(n, num_train)
test_loc_file = 'loc_test_springs%d.npy' %n
test_vel_file = 'vel_test_springs%d.npy' %n
test_edges_file = 'edges_test_springs%d.npy' %n

PATH = sub_folder + 'springs_net_temp_%d.pth' %temperature
output_file = sub_folder + 'springs_trajectory_temp_%d.npy' %temperature

def main():
    train_loc = np.load(train_loc_file, allow_pickle=True)
    train_vel = np.load(train_vel_file, allow_pickle=True)
    train_edges = np.load(train_edges_file, allow_pickle=True)
    test_loc = np.load(test_loc_file, allow_pickle=True)
    test_vel = np.load(test_vel_file, allow_pickle=True)
    test_edges = np.load(test_edges_file, allow_pickle=True)
    model = Model(n, length, units=units, activation=activation, convolute=convolute)
    model.cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    dataset = Data(train_loc, train_vel, train_edges, n, length=length)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)
    if anti:
        antidataloader = dataloader
    else:
        antidataloader = None
    if start > 1:
        model.load_state_dict(torch.load(PATH))

    if args.buffer:
        buf = ReplayBuffer()
    else:
        buf = None

    if start == 1:
        print('-')
        if not path.exists(folder + 'ground_truth/'):
            graph_ground_truth(test_loc, test_vel, test_edges, n, SummaryWriter(folder + 'ground_truth/'))
        print('.')
        test_prediction(model, test_loc, test_vel, test_edges, n, writer=writer, mppi_steps=200, std_dev=std_dev, temperature=temperature, epoch=0, dist=dist)
        print('.')
        for c in ['Mean', 'Median', 0]:
            test_inference(model, train_loc, train_vel, train_edges, n, num_to_test=num_train, cutoff=c, writer=writer, name='Train', epoch=0, training=True)
            test_inference(model, test_loc, test_vel, test_edges, n, num_to_test=1000, cutoff=c, writer=writer, name='Test', epoch=0, training=True)

    for i in range(start, end):
        print(i)
        train(model, dataloader, test_loc, test_vel, test_edges, n, antidataloader=antidataloader, writer=writer, epoch_num=i, mppi_steps=mppi_steps, std_dev=std_dev, temperature=temperature, inertia=inertia, dist=dist, buffer=buf, regularizer=True, lr=lr)
        torch.save(model.state_dict(), PATH)

        torch.save(model.state_dict(), sub_folder + 'model_%d.pth' %i)

        print('.')
        test_prediction(model, test_loc, test_vel, test_edges, n, writer=writer, mppi_steps=200, std_dev=std_dev, temperature=temperature, epoch=i, dist=dist)
        print('.')
        for c in ['Mean', 'Median', 0]:
            test_inference(model, train_loc, train_vel, train_edges, n, num_to_test=num_train, cutoff=c, writer=writer, name='Train', epoch=i, training=True)
            test_inference(model, test_loc, test_vel, test_edges, n, num_to_test=1000, cutoff=c, writer=writer, name='Test', epoch=i, training=True)

        print('-')
        for inert in [True, False]:
            sample(model, test_loc, test_vel, test_edges, n, 4, 100, writer=writer, epoch=i, mppi_steps=200, std_dev=std_dev, temperature=temperature, inertia=inert, dist=dist).cpu()

    writer.close()

if __name__ == '__main__':
    main()
