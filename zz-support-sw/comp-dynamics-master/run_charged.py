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
from separate_charged import *

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=5)
parser.add_argument('--num-train', type=int, default=50000)
parser.add_argument('--dist', type=int, default=1)
parser.add_argument('--model-length', type=int, default=10)
parser.add_argument('--units', type=int, default=128)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--convolute', default=False, action='store_true')
parser.add_argument('--no-convolute', dest='convolute', action='store_false')
parser.add_argument('--langevin', default=False, action='store_true')
parser.add_argument('--mppi', dest='langevin', action='store_false')
parser.add_argument('--steps', type=int, default=40)
parser.add_argument('--std-dev', type=float, default=0.1)
parser.add_argument('--noise', dest='std_dev', type=float)
parser.add_argument('--step-size', type=float, default=1)
parser.add_argument('--temperature', type=int, default=100)
parser.add_argument('--lr', default='5')
parser.add_argument('--backprop', default=True, action='store_true')
parser.add_argument('--no-backprop', dest='backprop', action='store_false')
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
steps = args.steps
std_dev = args.std_dev
step_size = args.step_size
temperature = args.temperature
lr = args.lr
anti = args.anti
inertia = args.inertia
start = args.start
end = args.end
if args.langevin:
    sampling = 'Langevin'
else:
    sampling = 'MPPI'

if n == 2 and num_train == 50000:
    folder = ''
else:
    folder = '%d_' %num_train
folder = folder + 'charged'
if sampling != 'MPPI':
    folder = folder + '_%s' %sampling.lower()
folder = folder + '_%d_%d' %(n, dist)
suffix = ''
if activation != 'relu':
    suffix = suffix + '_%s' %activation
if convolute:
    suffix = suffix + '_conv'
if anti:
    suffix = suffix + '_anti'
if args.buffer:
    suffix = suffix + '_persistent'
if inertia:
    suffix = suffix + '_inertia'
if units != 128:
    suffix = suffix + '_%d' %units
if lr == 'BCE':
    suffix = suffix + '_BCE'
else:
    suffix = suffix + '_lr%s' %lr
if not args.backprop:
    suffix = suffix + '_noprop'
suffix = suffix + '/'
folder = folder + suffix
if sampling == 'MPPI':
    inner = 'temp%d_%d' %(temperature, mppi_steps)
    params = [('', std_dev)]
elif sampling == 'Langevin':
    inner = '%d' %steps
    params = [('step', step_size), ('noise', std_dev)]
for tup in params:
    inner = inner + '_%s' %tup[0]
    param = tup[1]
    if param == 0:
        inner = inner + '0'
    elif param < 0.001:
        inner = inner + '%.4f' %param
    elif param < 0.01:
        inner = inner + '%.3f' %param
    elif param < 0.1:
        inner = inner + '%.2f' %param
    elif param >= 1:
        inner = inner + '%d' %param
    else:
        inner = inner + '%.1f' %param
inner = inner + '/'
sub_folder = folder + inner
writer = SummaryWriter(sub_folder)
if dist != 1:
    folder2 = ('separate_optimized_%d_1' %n) + suffix
    sub_folder2 = folder2 + inner
print(sub_folder)

train_loc_file = 'loc_train_charged%d_%d.npy' %(n, num_train)
train_vel_file = 'vel_train_charged%d_%d.npy' %(n, num_train)
train_edges_file = 'edges_train_charged%d_%d.npy' %(n, num_train)
test_loc_file = 'loc_test_charged%d.npy' %n
test_vel_file = 'vel_test_charged%d.npy' %n
test_edges_file = 'edges_test_charged%d.npy' %n

PATH1 = sub_folder + 'charged_trajectory_net.pth'
PATH2 = sub_folder + 'charged_same_net.pth'
PATH3 = sub_folder + 'charged_different_net.pth'
output_file = sub_folder + 'charged_trajectory.npy'

def main():
    train_loc = np.load(train_loc_file, allow_pickle=True)
    train_vel = np.load(train_vel_file, allow_pickle=True)
    train_edges = np.load(train_edges_file, allow_pickle=True)
    test_loc = np.load(test_loc_file, allow_pickle=True)
    test_vel = np.load(test_vel_file, allow_pickle=True)
    test_edges = np.load(test_edges_file, allow_pickle=True)
    trajectory_model = Trajectory_Model(length, units=units, activation=activation, convolute=convolute)
    trajectory_model.cuda()
    same_model = Interaction_Model(length, units=units, activation=activation, convolute=convolute)
    same_model.cuda()
    different_model = Interaction_Model(length, units=units, activation=activation, convolute=convolute)
    different_model.cuda()
    if dist != 1:
        small_trajectory = Trajectory_Model(length, units=units, activation=activation, convolute=convolute)
        small_trajectory.cuda()
        small_interaction = Interaction_Model(length, units=units, activation=activation, convolute=convolute)
        small_interaction.cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    trajectory_dataset = Trajectory_Data(train_loc, train_vel, n=n, length=length, dist=dist)
    trajectory_dataloader = DataLoader(trajectory_dataset, batch_size=256, shuffle=True, drop_last=True)
    same_dataset = Interaction_Data(train_loc, train_vel, train_edges, n=n, length=length, dist=dist)
    same_dataloader = DataLoader(same_dataset, batch_size=256, shuffle=True, drop_last=True)
    different_dataset = Interaction_Data(train_loc, train_vel, train_edges, n=n, length=length, dist=dist, interaction=-1)
    different_dataloader = DataLoader(different_dataset, batch_size=256, shuffle=True, drop_last=True)
    if anti:
        antisame = different_dataloader
        antidifferent = same_dataloader
    else:
        antisame = None
        antidifferent = None
    if start > 1:
        trajectory_model.load_state_dict(torch.load(PATH1))
        same_model.load_state_dict(torch.load(PATH2))
        different_model.load_state_dict(torch.load(PATH3))

    if args.buffer:
        buf1 = ReplayBuffer()
        buf2 = ReplayBuffer()
        buf3 = ReplayBuffer()
    else:
        buf1 = None
        buf2 = None
        buf3 = None

    if start == 1:
        print('-')
        if not path.exists(folder + 'ground_truth/'):
            graph_ground_truth(test_loc, test_vel, test_edges, n, SummaryWriter(folder + 'ground_truth/'))
        print('.')
        test_prediction(trajectory_model, same_model, different_model, test_loc, test_vel, test_edges, n, sampling=sampling, writer=writer, steps=steps, std_dev=std_dev, step_size=step_size, temperature=temperature, epoch=0, dist=dist)
        for c in ['Mean', 'Median', 0]:
            print('.')
            test_inference(same_model, different_model, train_loc, train_vel, train_edges, n, num_to_test=num_train, cutoff=c, writer=writer, name='Train', epoch=0, training=True)
            test_inference(same_model, different_model, test_loc, test_vel, test_edges, n, num_to_test=1000, cutoff=c, writer=writer, name='Test', epoch=0, training=True)

    for i in range(start, end):
        print(i)
        train(trajectory_model, trajectory_dataloader, test_loc, test_vel, test_edges, sampling=sampling, writer=writer, name='Trajectory', epoch_num=i, steps=steps, std_dev=std_dev, step_size=step_size, temperature=temperature, inertia=inertia, dist=dist, buffer=buf1, regularizer=True, lr=lr, backprop=args.backprop)
        torch.save(trajectory_model.state_dict(), PATH1)
        torch.save(trajectory_model.state_dict(), sub_folder + 'trajectory_%d.pth' %i)

        print('-')
        train(same_model, same_dataloader, test_loc, test_vel, test_edges, sampling=sampling, antidataloader=antisame, writer=writer, name='Same_Interaction', epoch_num=i, steps=steps, std_dev=std_dev, step_size=step_size, temperature=temperature, inertia=inertia, dist=dist, buffer=buf2, regularizer=True, lr=lr, backprop=args.backprop)
        torch.save(same_model.state_dict(), PATH2)
        torch.save(same_model.state_dict(), sub_folder + 'same_%d.pth' %i)

        print('-')
        train(different_model, different_dataloader, test_loc, test_vel, test_edges, sampling=sampling, antidataloader=antidifferent, writer=writer, name='Different_Interaction', epoch_num=i, steps=steps, std_dev=std_dev, step_size=step_size, temperature=temperature, inertia=inertia, dist=dist, buffer=buf2, regularizer=True, lr=lr, backprop=args.backprop)
        torch.save(different_model.state_dict(), PATH2)
        torch.save(different_model.state_dict(), sub_folder + 'different_%d.pth' %i)

        print('.')
        test_prediction(trajectory_model, same_model, different_model, test_loc, test_vel, test_edges, n, sampling=sampling, writer=writer, steps=steps, std_dev=std_dev, step_size=step_size, temperature=temperature, epoch=i, dist=dist)
        print('.')
        for c in ['Mean', 'Median', 0]:
            test_inference(same_model, different_model, train_loc, train_vel, train_edges, n, num_to_test=num_train, cutoff=c, writer=writer, name='Train', epoch=i, training=True)
            test_inference(same_model, different_model, test_loc, test_vel, test_edges, n, num_to_test=1000, cutoff=c, writer=writer, name='Test', epoch=i, training=True)

        print('-')
        for inert in [True, False]:
            if dist == 1:
                sample_test(trajectory_model, same_model, different_model, test_loc, test_vel, test_edges, n, sampling, 4, 100, writer=writer, epoch=i, steps=40, std_dev=std_dev, step_size=step_size, temperature=temperature, inertia=inert, dist=dist).cpu()
            else:
                small_trajectory.load_state_dict(torch.load(sub_folder2 + 'trajectory_%d.pth' %i))
                small_interaction.load_state_dict(torch.load(sub_folder2 + 'interaction_%d.pth' %i))
                subsample_test(small_trajectory, small_interaction, trajectory_model, interaction_model, test_loc, test_vel, test_edges, n, 4, 100, writer=writer, epoch=i, mppi_steps=200, std_dev=std_dev, temperature=temperature, inertia=inert, dist=dist).cpu()
 

    writer.close()

if __name__ == '__main__':
    main()
