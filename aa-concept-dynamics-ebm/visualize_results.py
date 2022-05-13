import numpy as np
import matplotlib.pyplot as plt


def get_trajectory_figure(state, b_idx, lims=None, plot_type ='loc', highlight_nodes = None, node_thickness = None, args = None):
	fig = plt.figure()
	axes = plt.gca()
	sz_pt = 80
	lw = [1.5]*state.shape[1]
	if node_thickness is not None:
		lw = (node_thickness - node_thickness.min()) / (node_thickness.max() - node_thickness.min())
		lw = 1.5 + lw * 2.5
		labels = [str(node_thickness[i])[:4] for i in range(node_thickness.shape[0])]
	else: labels = [-1]*state.shape[1]
	maps = ['bone', 'magma', 'spring', 'autumn', 'Greys_r', 'gist_gray',
			'afmhot', 'cool', 'Wistia', 'YlGnBu', 'pink', 'copper', 'Reds_r'] #https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
	cmap = [maps[4]]*state.shape[1]
	alpha = 0.99
	if lims is not None:
		axes.set_xlim([lims[0], lims[1]])
		axes.set_ylim([lims[0], lims[1]])
	# state = state[b_idx].permute(1, 2, 0).cpu().detach().numpy()
	state = np.transpose(state[b_idx], (1, 2, 0))
	loc, vel = state[:, :2][None], state[:, 2:][None]

	# vel_norm = np.sqrt((vel ** 2).sum(axis=1))
	colors = ['b', 'r', 'c', 'y', 'k', 'm', 'g', 'aquamarine', 'tab:brown', 'tab:purple', 'tab:pink']
	modes = ['-']*loc.shape[-1]
	if highlight_nodes is not None:
		# modes = ['-' if node == 0 else ':' for node in highlight_nodes]
		# colors = ['b', 'c', 'y', 'k', 'm', 'g', 'aquamarine', 'tab:brown', 'tab:purple', 'tab:pink']
		# colors = ['r' if node == 1 else colors[i] for i, node in enumerate(highlight_nodes)]
		# lw = [1.5 if node == 0 else 2.5 for i, node in enumerate(highlight_nodes)]
		cmap = [maps[-1] if node == 1 else cmap[i] for i, node in enumerate(highlight_nodes)]
		assert len(modes) == loc.shape[-1]

	if plot_type == 'loc' or plot_type == 'both':
		for i in range(loc.shape[-1]):
			plt.scatter(loc[0, :, 0, i], loc[0, :, 1, i], s=sz_pt, c=np.arange(loc.shape[1]), cmap=cmap[i], alpha=alpha)
			plt.plot(loc[0, :, 0, i], loc[0, :, 1, i], modes[i], c=colors[i], linewidth=lw[i], label=labels[i])
			plt.scatter(loc[0, 0, 0, i], loc[0, 0, 1, i], c=colors[i], s=sz_pt)
			# if args.forecast > -1:
			#     plt.plot(loc[0, -args.forecast, 0, i], loc[0, -args.forecast, 1, i], 'x', c=colors[i])
		pass
	if plot_type == 'vel' or plot_type == 'both':
		for i in range(loc.shape[-1]):
			acc_pos = loc[0, 0:1, 0, i], loc[0, 0:1, 1, i]
			vels = vel[0, :, 0, i], vel[0, :, 1, i]
			for t in range(loc.shape[1] - 1):
				acc_pos = np.concatenate([acc_pos[0], acc_pos[0][t:t+1]+vels[0][t:t+1]]), \
						  np.concatenate([acc_pos[1], acc_pos[1][t:t+1]+vels[1][t:t+1]])
			plt.scatter(acc_pos[0], acc_pos[1], s=sz_pt, c=np.arange(loc.shape[1]), cmap=cmap[i], alpha=alpha)
			plt.plot(acc_pos[0], acc_pos[1], modes[i], c=colors[i], linewidth=lw[i], label=labels[i])
			plt.scatter(loc[0, 0, 0, i], loc[0, 0, 1, i], c=colors[i], s=sz_pt)			# if args.forecast > -1:
			#     plt.plot(loc[0, -args.forecast, 0, i], loc[0, -args.forecast, 1, i], 'x', c=colors[i])

	if labels[0] != -1:
		axes.legend(prop={'size': 17})
	n = 0.1
	nn = 0
	nrange = np.arange(-1,1 + n,n)
	nrange = nrange[nrange > np.min(loc[0, :, 0, :]) - nn]
	xminmax = nrange[nrange < np.max(loc[0, :, 0, :]) + nn]
	nrange = np.arange(-1,1+n,n)
	nrange = nrange[nrange > np.min(loc[0, :, 1, :]) - nn]
	yminmax = nrange[nrange < np.max(loc[0, :, 1, :]) + nn]
	plt.xticks(list(xminmax))
	plt.yticks(list(yminmax))

	return plt, fig


data_folder = '/data/Armand/EBM/cachedir/results/'



#### EXPERIMENT 1 ######
# affected_nodes = None
# node_thickness = None
# plot_attr = 'vel'
# limpos = limneg = 1
# lims = [-limneg, limpos]
# lims = None
# n_timesteps = 26
# file_id = 0
# trajnum = 2
#
# iterate_range = 1
# for file_id in range(5):
# 	for trajnum in range(3):
#
# 		# Mixing Experiment
# 		experiment_folder = 'test_recombine/' # 'test_recombine'
# 		filenames = ['Ca+Srk_better', 'Sa+Crc', 'Ca+Srk_decent','Sa+Cbk_maybe', 'Sa+Ccy']
# 		filenames = [file + '_SC_recomb_' for file in filenames]
# 		filename = filenames[file_id]
#
# 		subnames = ['gt_charged', 'gt_springs', 'all_samples']
# 		affected_nodes = np.load(data_folder + experiment_folder + filename + 'affected_nodes' + '.npy')
#
#
# 		if (trajnum == 0 and filename[0] == 'C') or \
# 			(trajnum == 1 and filename[0] == 'S'):
# 			affected_nodes = None
#
# 		trajname = subnames[trajnum]
# 		trajectory = np.load(data_folder + experiment_folder + filename + subnames[trajnum] + '.npy')
# 		if trajname == 'all_samples': trajectory = trajectory[-1]
# 		# iterate_range = all_samples.shape[0]
# 		# affected_nodes = 1 - affected_nodes
#
#
#
# 		for i in range(iterate_range):
# 			if iterate_range>1:
# 				traj = trajectory[i:i+1,:,n_timesteps]
# 			else: traj = trajectory[:,:,:n_timesteps]
#
# 			plt, fig = get_trajectory_figure(traj, lims=lims, b_idx=0,
# 								  plot_type =plot_attr,
# 								  highlight_nodes = affected_nodes,
# 								  node_thickness=node_thickness)
#
# 			plt.show()
# 			save = input('Save with name: (s+(anything): saves it, e: exit, else pass) ')
# 			if save[0] == 's':
# 				plt, fig = get_trajectory_figure(traj, lims=lims, b_idx=0,
# 												 plot_type =plot_attr,
# 												 highlight_nodes = affected_nodes,
# 												 node_thickness=node_thickness)
# 				plt.savefig(data_folder + experiment_folder + filename + trajname + save[1:] + '_figure.png', dpi=300)
# 				print('Saved as {}.'.format(data_folder + experiment_folder + filename + trajname + save[1:] + '_figure.png'))
# 			elif save == 'e':
# 				print('Bye.')
# 				exit()


# Outlier detection
import os
affected_nodes = None
node_thickness = None
plot_attr = 'vel'
limpos = limneg = 1
lims = [-limneg, limpos]
lims = None
n_timesteps = 40

experiment_folder = 'outlier_detection/' # 'test_recombine'
filenames = os.listdir(data_folder + experiment_folder)
filenames = [filename.strip('edges.npy') for filename in filenames if filename.endswith('edges.npy')]
subnames = ['edges', 'energies', 'gt_traj_all', 'gt_traj_pred']

for filename in filenames:
	edges = np.load(data_folder + experiment_folder + filename + subnames[0] + '.npy')
	energies = np.load(data_folder + experiment_folder + filename + subnames[1] + '.npy')
	gt_traj = np.load(data_folder + experiment_folder + filename + subnames[2] + '.npy')
	gt_pred_traj = np.load(data_folder + experiment_folder + filename + subnames[2] + '.npy')
	node_thickness = energies[0]
	highlight_nodes = 1-edges[0]
	traj = gt_traj[:, :, :n_timesteps]

	plt, fig = get_trajectory_figure(traj, lims=lims, b_idx=0,
									 plot_type =plot_attr,
									 highlight_nodes = highlight_nodes,
									 node_thickness=node_thickness)

	plt.show()
	save = input('Save with name: (s+(anything): saves it, e: exit, else pass) ')
	if save[0] == 's':
		plt, fig = get_trajectory_figure(traj, lims=lims, b_idx=0,
										 plot_type =plot_attr,
										 highlight_nodes = highlight_nodes,
										 node_thickness=node_thickness)
		plt.savefig(data_folder + experiment_folder + filename + save[1:] + '_figure.png', dpi=300)
		print('Saved as {}.'.format(data_folder + experiment_folder + filename + save[1:] + '_figure.png'))
	elif save == 'e':
		print('Bye.')
		exit()
