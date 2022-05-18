import numpy as np
import pandas as pd
import os
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='',
					help='What simulation to generate.')
parser.add_argument('--num-train', type=int, default=50000,
					help='Number of training simulations to generate.')
parser.add_argument('--num-valid', type=int, default=1000,
					help='Number of validation simulations to generate.')
parser.add_argument('--num-test', type=int, default=1000,
					help='Number of test simulations to generate.')
parser.add_argument('--length', type=int, default=100,
					help='Length of trajectory after subsampling.')
parser.add_argument('--length-test', type=int, default=5000,
					help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=2,
					help='How often to sample the trajectory.')
# parser.add_argument('--n-balls', type=int, default=3,
# 					help='Number of balls in the simulation.')
parser.add_argument('--seed', type=int, default=42,
					help='Random seed.')

args = parser.parse_args()

suffix = '_NBA_11'
np.random.seed(args.seed)

print(suffix)

datadir = '/data/Armand/nba-data-large/npy/'
def generate_dataset(num_sims, length, sample_freq):
	loc_all = list()
	vel_all = list()
	edges_all = list()


	num_sims_train, num_sims_val, num_sims_test = num_sims

	splits = []
	count = 0
	num_sims_count = 0
	for file in os.listdir(datadir):
		array = np.load(datadir + file)
		if array.shape[1] < (length * sample_freq):
			continue
		else: res = array.shape[1] % (length * sample_freq)
		num_chunks = array.shape[1] // (length * sample_freq)
		for i in range(num_chunks):
			subarray = array[:, (length * sample_freq)*(i):(length * sample_freq)*(i+1):sample_freq]
			loc_all.append(subarray)
			count += 1
			if num_sims_count > 2: break
			if count == num_sims[num_sims_count]:
				loc_array = np.stack(loc_all)
				splits.append(loc_array)
				num_sims_count += 1
				count = 0
				loc_all = []
				print('Filled split #{}.'.format(num_sims_count))
				break
		if num_sims_count > 2: break
	return splits

print("Generating {},{},{} simulations".format(args.num_train, args.num_valid, args.num_test))
loc_train, loc_valid, loc_test = generate_dataset( [args.num_train, args.num_valid, args.num_test],
													 args.length,
													 args.sample_freq)
#
suffix += '_len{}_sf{}'.format(args.length, args.sample_freq)
# '_{}sims.npy'.format(args.num_train)
print(loc_train.shape)
print(loc_valid.shape)
print(loc_test.shape)
np.save(datadir + 'loc_train' + suffix + '.npy', loc_train)
np.save(datadir + 'loc_valid' + suffix + '.npy', loc_valid)
np.save(datadir + 'loc_test' + suffix + '.npy', loc_test)

print(datadir + 'loc_train' + suffix + '.npy')

# import matplotlib.pyplot as plt
# def visualize_trajectories(list, list_idx=0, len=50, sr=5):
# 	fig = plt.figure()
# 	axes = plt.gca()
# 	lw = 1.5
# 	sz_pt = 30
# 	maps = ['bone', 'magma', 'spring', 'autumn', 'gist_gray', 'afmhot', 'cool', 'Wistia', 'YlGnBu'] #https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
# 	cmap = maps[4]
# 	alpha = 0.6
# 	# if lims is not None:
# 	# 	axes.set_xlim([lims[0], lims[1]])
# 	# 	axes.set_ylim([lims[0], lims[1]])
# 	loc = list[list_idx]
# 	loc = np.transpose(loc, (1,2,0))[:, :2] # L, F, No
# 	colors = ['b', 'r', 'c', 'y', 'k', 'm', 'g']
# 	modes = ['-']*loc.shape[-1]
#
#
# 	for i in range(loc.shape[-1]):
# 		# plt.scatter(loc[:, 0, i], loc[:, 1, i], s=sz_pt, c=np.arange(loc.shape[1]), cmap=cmap, alpha=alpha)
# 		# plt.plot(loc[-len*sr::sr, 0, i], loc[-len*sr::sr, 1, i], modes[i], linewidth=lw)
# 		plt.plot(loc[:len*sr:sr, 0, i], loc[:len*sr:sr, 1, i], modes[i], linewidth=lw)
# 		plt.plot(loc[0, 0, i], loc[0, 1, i], 'o')
# 	plt.show()
#
# if visualize:
# 	visualize_trajectories(final_numpy_list, list_idx=-2, len=20)

