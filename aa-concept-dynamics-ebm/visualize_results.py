import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import ListedColormap

# def get_trajectory_figure(state, b_idx, lims=None, plot_type ='loc', highlight_nodes = None, node_thickness = None, args = None, grads=None, grad_id=None):
# 	fig = plt.figure()
# 	axes = plt.gca()
# 	sz_pt = 80
# 	lw = [1.5]*state.shape[1]
# 	if node_thickness is not None:
# 		lw = (node_thickness - node_thickness.min()) / (node_thickness.max() - node_thickness.min())
# 		lw = 1.5 + lw * 2.5
# 		labels = [str(node_thickness[i])[:4] for i in range(node_thickness.shape[0])]
# 	else: labels = [-1]*state.shape[1]
# 	maps = ['bone', 'magma', 'spring', 'autumn', 'Greys_r', 'gist_gray',
# 			'afmhot', 'cool', 'Wistia', 'YlGnBu', 'pink', 'copper', 'Reds_r'] #https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
# 	cmap = [maps[4]]*state.shape[1]
# 	alpha = 0.99
# 	if lims is not None:
# 		axes.set_xlim([lims[0], lims[1]])
# 		axes.set_ylim([lims[0], lims[1]])
# 	# state = state[b_idx].permute(1, 2, 0).cpu().detach().numpy()
# 	state = np.transpose(state[b_idx], (1, 2, 0))
# 	loc, vel = state[:, :2][None], state[:, 2:][None]
#
# 	# vel_norm = np.sqrt((vel ** 2).sum(axis=1))
#
# 	colors = ['b', 'r', 'c', 'y', 'k', 'm', 'g', 'aquamarine', 'tab:brown', 'tab:purple', 'tab:pink']
# 	if loc.shape[-1] == 11:
# 		colors = ['r',
# 				  (51/255,35/255,1), (108/255,97/255,1), (153/255,145/255,1), (202/255,197/255,1), (220/255,215/255,1),
# 				  (141/255,144/255,21/255),(157/255,160/255,43/255), (180/255,183/255,36/255),  (201/255,204/255,58/255), (219/255,226/255,71/255)]
#
# 	modes = ['-']*loc.shape[-1]
# 	if highlight_nodes is not None:
# 		# modes = ['-' if node == 0 else ':' for node in highlight_nodes]
# 		# colors = ['b', 'c', 'y', 'k', 'm', 'g', 'aquamarine', 'tab:brown', 'tab:purple', 'tab:pink']
# 		# colors = ['r' if node == 1 else colors[i] for i, node in enumerate(highlight_nodes)]
# 		# lw = [1.5 if node == 0 else 2.5 for i, node in enumerate(highlight_nodes)]
# 		cmap = [maps[-1] if node == 1 else cmap[i] for i, node in enumerate(highlight_nodes)]
# 		assert len(modes) == loc.shape[-1]
#
# 	if plot_type == 'loc' or plot_type == 'both':
# 		for i in range(loc.shape[-1]):
# 			plt.scatter(loc[0, :, 0, i], loc[0, :, 1, i], s=sz_pt, c=np.arange(loc.shape[1]), cmap=cmap[i], alpha=alpha)
# 			plt.plot(loc[0, :, 0, i], loc[0, :, 1, i], modes[i], c=colors[i], linewidth=lw[i], label=labels[i])
# 			plt.scatter(loc[0, 0, 0, i], loc[0, 0, 1, i], c=colors[i], s=sz_pt)
# 			# Q = plt.quiver(loc[0, :, 0, i], loc[0, :, 1, i], vel[0, :, 0, i], vel[0, :, 0, i],
# 			# 			   [vel[0, :, 0, i]])
# 			# qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
# 			# 				   coordinates='figure')
# 		# if args.forecast > -1:
# 		#     plt.plot(loc[0, -args.forecast, 0, i], loc[0, -args.forecast, 1, i], 'x', c=colors[i])
# 		pass
# 	if plot_type == 'vel' or plot_type == 'both':
# 		for i in range(loc.shape[-1]):
# 			acc_pos = loc[0, 0:1, 0, i], loc[0, 0:1, 1, i]
# 			vels = vel[0, :, 0, i], vel[0, :, 1, i]
# 			for t in range(loc.shape[1] - 1):
# 				acc_pos = np.concatenate([acc_pos[0], acc_pos[0][t:t+1]+vels[0][t:t+1]]), \
# 						  np.concatenate([acc_pos[1], acc_pos[1][t:t+1]+vels[1][t:t+1]])
# 			plt.scatter(acc_pos[0], acc_pos[1], s=sz_pt, c=np.arange(loc.shape[1]), cmap=cmap[i], alpha=alpha)
# 			plt.plot(acc_pos[0], acc_pos[1], modes[i], c=colors[i], linewidth=lw[i], label=labels[i])
# 			plt.scatter(loc[0, 0, 0, i], loc[0, 0, 1, i], c=colors[i], s=sz_pt)			# if args.forecast > -1:
# 		#     plt.plot(loc[0, -args.forecast, 0, i], loc[0, -args.forecast, 1, i], 'x', c=colors[i])
#
# 	if grads is not None:
# 		grads = np.transpose(grads[b_idx], (1, 2, 0))[None]
# 		for i in range(1, loc.shape[-1]):
# 			if grad_id is not None:
# 				if i != grad_id: continue
# 			num_fixed_timesteps = 5
# 			Q = plt.quiver(loc[0, num_fixed_timesteps:, 0, i], loc[0, num_fixed_timesteps:, 1, i],
# 						   grads[0, num_fixed_timesteps:, 0, i], grads[0, num_fixed_timesteps:, 0, i],
# 						   [(grads[0, num_fixed_timesteps:, 0, i]**2 + grads[0, num_fixed_timesteps:, 1, i]**2)**0.5],
# 						   angles = 'xy', width=0.003*(lw[0]))
#
# 	if labels[0] != -1:
# 		axes.legend(prop={'size': 17})
# 	n = 0.1
# 	nn = 0
# 	nrange = np.arange(-1,1 + n,n)
# 	nrange = nrange[nrange > np.min(loc[0, :, 0, :]) - nn]
# 	xminmax = nrange[nrange < np.max(loc[0, :, 0, :]) + nn]
# 	nrange = np.arange(-1,1+n,n)
# 	nrange = nrange[nrange > np.min(loc[0, :, 1, :]) - nn]
# 	yminmax = nrange[nrange < np.max(loc[0, :, 1, :]) + nn]
# 	plt.xticks(list(xminmax))
# 	plt.yticks(list(yminmax))
#
# 	return plt, fig

from matplotlib.colors import LinearSegmentedColormap
def get_trajectory_figure(state, b_idx, lims=None, plot_type ='loc', highlight_nodes = None, node_thickness = None, args = None, grads=None, grad_id=None, mark_point=None):
	fig = plt.figure()
	axes = plt.gca()
	sz_pt = [80]*state.shape[1]
	lw = [1.5]*state.shape[1]
	if node_thickness is not None:
		sz_pt = (node_thickness - node_thickness.min()) / (node_thickness.max() - node_thickness.min())
		sz_pt = 80 + sz_pt * 170
		labels = [str(node_thickness[i])[:4] for i in range(node_thickness.shape[0])]
	else: labels = [-1]*state.shape[1]

	alpha = 0.7

	try:
		state = np.transpose(state[b_idx], (1, 2, 0))
	except: plt.close(); return None, None
	loc = state[:, :2][None]
	if state.shape[1] == 4:
		vel = state[:, 2:][None]

	if lims is not None and len(lims)==4:
		axes.set_xlim([lims[0], lims[1]])
		axes.set_ylim([lims[2], lims[3]])

	#https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html

	color_arange =  np.arange(loc.shape[1])
	# construct cmap
	palettes = \
		[sns.color_palette("viridis", as_cmap=True),
			sns.color_palette("crest_r", as_cmap=True),
		 	sns.color_palette("rocket", as_cmap=True)]

	if loc.shape[-1] == 11:
		sz_pt[0] = 40
		cmap = [LinearSegmentedColormap.from_list(
			"Custom", [(255/255, 165*1.1/255, 0),(255/255, 165*1.1/255, 0)], N=loc.shape[1])] #['Greys_r']
		cmap.extend([palettes[1]]*5)
		cmap.extend([palettes[2]]*5)
	elif node_thickness is not None:
		cmap = [palettes[2]] # TODO: hardcoded for the two specific examples
		cmap.extend([palettes[1]]*5)
	else:
		colors = sns.color_palette(as_cmap=True)
		cmap = []
		for i in range(loc.shape[-1]):
			palette = sns.color_palette("light:"+colors[i], as_cmap=True)

			cmap.append(palette)
		color_arange = np.flip(color_arange, axis=0)
	if plot_type == 'loc' or plot_type == 'both':
		for i in range(loc.shape[-1]):
			plt.scatter(loc[0, :, 0, i], loc[0, :, 1, i], s=sz_pt[i], c=np.arange(loc.shape[1]), cmap=cmap[i], alpha=alpha, label=labels[i])
			# plt.plot(loc[0, :, 0, i], loc[0, :, 1, i], modes[i], c=colors[i], linewidth=lw[i], label=labels[i])
			# plt.scatter(loc[0, 0, 0, i], loc[0, 0, 1, i], c=colors[i], s=sz_pt)
			if mark_point is not None:
				plt.scatter(loc[0,mark_point,0,i], loc[0,mark_point,1,i],s=sz_pt[0], c='k')

		i = 0
		plt.scatter(loc[0, :, 0, i], loc[0, :, 1, i], s=sz_pt[i], c=np.arange(loc.shape[1]), cmap=cmap[i], alpha=alpha, label=labels[i])

		pass
	plt.grid(color='gray', linestyle='--', linewidth=0.5)
	if plot_type == 'vel' or plot_type == 'both':
		for i in range(loc.shape[-1]):
			acc_pos = loc[0, 0:1, 0, i], loc[0, 0:1, 1, i]
			vels = vel[0, :, 0, i], vel[0, :, 1, i]
			for t in range(loc.shape[1] - 1):
				acc_pos = np.concatenate([acc_pos[0], acc_pos[0][t:t+1]+vels[0][t:t+1]]), \
						  np.concatenate([acc_pos[1], acc_pos[1][t:t+1]+vels[1][t:t+1]])
			plt.scatter(acc_pos[0], acc_pos[1], s=sz_pt[i], c=color_arange, cmap=cmap[i], alpha=alpha, label=labels[i])
			# plt.plot(acc_pos[0], acc_pos[1], modes[i], c=colors[i], linewidth=lw[i], label=labels[i])
			# plt.scatter(loc[0, 0, 0, i], loc[0, 0, 1, i], c=colors[i], s=sz_pt)			# if args.forecast > -1:
			if mark_point is not None:
				plt.scatter(acc_pos[0][mark_point], acc_pos[1][mark_point],s=sz_pt[0], c='k')

	if grads is not None:
		grads = np.transpose(grads[b_idx], (1, 2, 0))[None]
		for i in range(1, loc.shape[-1]):
			if grad_id is not None:
				if i != grad_id: continue
			num_fixed_timesteps = 5
			Q = plt.quiver(loc[0, num_fixed_timesteps:, 0, i], loc[0, num_fixed_timesteps:, 1, i],
						   grads[0, num_fixed_timesteps:, 0, i], grads[0, num_fixed_timesteps:, 0, i],
						   [(grads[0, num_fixed_timesteps:, 0, i]**2 + grads[0, num_fixed_timesteps:, 1, i]**2)**0.5],
						   angles = 'xy', width=0.003*(lw[0]))
	plt.axis('off')
	hfont = {'fontname':'Times New Roman'}
	if labels[0] != -1:
		axes.legend(prop={'size': 18})
	axes.tick_params(axis='both', labelsize=20)

	# n = 0.4
	# nn = 0.4
	# nrange = np.arange(-1,1 + n,n)
	# nrange = nrange[nrange > np.min(loc[0, :, 0, :]) - nn]
	# xminmax = nrange[nrange < np.max(loc[0, :, 0, :]) + nn]
	# nrange = np.arange(-1,1+n,n)
	# nrange = nrange[nrange > np.min(loc[0, :, 1, :]) - nn]
	# yminmax = nrange[nrange < np.max(loc[0, :, 1, :]) + nn]
	# plt.xticks(list(xminmax), **hfont)
	# plt.yticks(list(yminmax), **hfont)

	return plt, fig


data_folder = '/data/Armand/EBM/cachedir/results/'



#### Recombination ######
# affected_nodes = None
# node_thickness = None
# plot_attr = 'vel'
# limpos = limneg = 1
# lims = [-limneg, limpos]
# lims = None
# n_timesteps = 50
# n_timesteps_ini = 2
# file_id = 0
# trajnum = 2
#
# iterate_range = 1
#
# experiment_folder = 'test_recombine/'
# filenames = os.listdir(data_folder + experiment_folder)
# filenames = [filename[:-len('gt_charged.npy')] for filename in filenames if filename.endswith('gt_charged.npy')]
# subnames = ['all_samples', 'gt_charged', 'gt_springs']
# #
# for filename in filenames:
# 	affected_nodes = np.load(data_folder + experiment_folder + filename + 'affected_nodes' + '.npy')
# 	for trajnum in range(3):
#
# 		# if (trajnum == 0 and filename[0] == 'C') or \
# 		# 		(trajnum == 1 and filename[0] == 'S'):
# 		affected_nodes = None
#
# 		trajname = subnames[trajnum]
# 		trajectory = np.load(data_folder + experiment_folder + filename + subnames[trajnum] + '.npy')
# 		if trajname == 'all_samples': trajectory = trajectory[-1]
# 		# iterate_range = all_samples.shape[0]
# 		# affected_nodes = 1 - affected_nodes

# 		for i in range(iterate_range):
# 			if iterate_range>1:
# 				traj = trajectory[i:i+1,:,n_timesteps_ini:n_timesteps]
# 			else: traj = trajectory[:,:,n_timesteps_ini:n_timesteps]
#
# 			plt, fig = get_trajectory_figure(traj, lims=lims, b_idx=0,
# 											 plot_type =plot_attr,
# 											 highlight_nodes = affected_nodes,
# 											 node_thickness=node_thickness)
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


# # Outlier detection
# import os
# affected_nodes = None
# node_thickness = None
# plot_attr = 'loc'
# limpos = limneg = 1
# lims = [-limneg, limpos]
# lims = None
# n_timesteps = 40
#
# experiment_folder = 'outlier_detection/' # 'test_recombine'
# filenames = os.listdir(data_folder + experiment_folder)
# filenames = [filename.strip('edges.npy') for filename in filenames if filename.endswith('edges.npy')]
# subnames = ['edges', 'energies', 'gt_traj_all', 'gt_traj_pred']
#
# for filename in filenames:
# 	edges = np.load(data_folder + experiment_folder + filename + subnames[0] + '.npy')
# 	energies = np.load(data_folder + experiment_folder + filename + subnames[1] + '.npy')
# 	gt_traj = np.load(data_folder + experiment_folder + filename + subnames[2] + '.npy')
# 	gt_pred_traj = np.load(data_folder + experiment_folder + filename + subnames[2] + '.npy')
# 	node_thickness = energies[0]
# 	highlight_nodes = 1-edges[0]
# 	traj = gt_traj[:, :, :n_timesteps]
#
# 	plt, fig = get_trajectory_figure(traj, lims=lims, b_idx=0,
# 									 plot_type =plot_attr,
# 									 highlight_nodes = highlight_nodes,
# 									 node_thickness=node_thickness)
#
# 	plt.show()
# 	save = input('Save with name: (s+(anything): saves it, e: exit, else pass) ')
# 	if save[0] == 's':
# 		plt, fig = get_trajectory_figure(traj, lims=lims, b_idx=0,
# 										 plot_type =plot_attr,
# 										 highlight_nodes = highlight_nodes,
# 										 node_thickness=node_thickness)
# 		plt.savefig(data_folder + experiment_folder + filename + save[1:] + '_figure.png', dpi=300)
# 		print('Saved as {}.'.format(data_folder + experiment_folder + filename + save[1:] + '_figure.png'))
# 	elif save == 'e':
# 		print('Bye.')
# 		exit()

### Plot gradients for NBA
# import os
# affected_nodes = None
# node_thickness = None
# plot_attr = 'loc'
# limpos = limneg = 1
# lims = [-limneg, limpos]
# lims = None
# n_timesteps = 40
#
# experiment_folder = 'gradient_plots/'
# filenames = os.listdir(data_folder + experiment_folder)
# filenames = [filename[:-9] for filename in filenames if filename.endswith('feats.npy')]
# subnames = ['grads', 'feats', 'player_id', 'latent']
#
# for filename in filenames:
# 	grads = np.load(data_folder + experiment_folder + filename + subnames[0] + '.npy')
# 	feats = np.load(data_folder + experiment_folder + filename + subnames[1] + '.npy')
# 	player_id = np.load(data_folder + experiment_folder + filename + subnames[2] + '.npy')
# 	traj = feats[:, :, :n_timesteps]
# 	grad = grads[:, :, :n_timesteps]
#
# 	plt, fig = get_trajectory_figure(traj, lims=lims, b_idx=0,
# 									 plot_type =plot_attr,
# 									 grads=grads, grad_id=player_id)
#
# 	plt.show()
# 	save = input('Save with name: (s+(anything): saves it, e: exit, else pass) ')
# 	if save[0] == 's':
# 		plt, fig = get_trajectory_figure(traj, lims=lims, b_idx=0,
# 										 plot_type =plot_attr,
# 										 grads=grads, grad_id=player_id)
# 		plt.savefig(data_folder + experiment_folder + filename + save[1:] + '_figure.png', dpi=300)
# 		print('Saved as {}.'.format(data_folder + experiment_folder + filename + save[1:] + '_figure.png'))
# 	elif save == 'e':
# 		print('Bye.')
# 		exit()

### Plot new constraints for NBA
# import os
# affected_nodes = None
# node_thickness = None
# plot_attr = 'vel'
# limpos = limneg = 1
# lims = [-limneg, limpos]
# lims = None
# n_timesteps = 40
# n_timesteps_ini = 3
# iter_id = -1
#
# # b_idx, lims = 10, [0.2,0.8,-0.9,0.1] # example 1
# # b_idx, lims = 3, [-0.7,0.3,-0.8,0.2] # example 2
# b_idx, lims = 0, [-0.8,0.3,-0.8,0.1] # example add 1 # attr b_idx + 3 Note: For the new saved we dont need to specify the batch index.
# # b_idx, lims = 0, [0.25,0.8,-0.8,0.0] # example add 2 (vel) b_idx + 36 + 30
#
# experiment_folder = 'new_constrain/'
# filenames = os.listdir(data_folder + experiment_folder)
# filenames = [filename[:-9] for filename in filenames if filename.endswith('feats.npy')]
# subnames = ['feat_neg_last', 'feat_negs', 'feats']
# print(filenames)
# for filename in filenames:
# 	feat_neg_last = np.load(data_folder + experiment_folder + filename + subnames[0] + '.npy')
# 	feat_negs = np.load(data_folder + experiment_folder + filename + subnames[1] + '.npy')
# 	feats = np.load(data_folder + experiment_folder + filename + subnames[1] + '.npy')
#
# 	traj = feat_negs[iter_id, :, :, :n_timesteps]
#
# 	plt_i, fig_i = get_trajectory_figure(traj, lims=lims, b_idx=b_idx,
# 									 plot_type =plot_attr)
# 	if fig_i is None: continue
#
# 	plt_i.show()
# 	save = input('Save with name: (s+(anything): saves it, e: exit, else pass) ')
# 	if save[0] == 's':
# 		plt, fig = get_trajectory_figure(traj, lims=lims, b_idx=b_idx,
# 										 plot_type =plot_attr)
#
# 		plt.savefig(data_folder + experiment_folder + filename + save[1:] + '_figure.png', dpi=300)
# 		print('Saved as {}.'.format(data_folder + experiment_folder + filename + save[1:] + '_figure.png'))
# 	elif save == 'e':
# 		print('Bye.')
# 		exit()

### Teaser figure
# affected_nodes = None
# node_thickness = None
# plot_attr = 'loc'
#
#
# # b_idx, lims = 10, [0.2,0.8,-0.9,0.1] # example 1
# # b_idx, lims = 3, [-0.7,0.3,-0.8,0.2] # example 2
#
# experiment_folder = 'teaser/'
# data_nba_folder = '/data/Armand/nba-data-large/processed/'
# filename = 'loc_train_NBA_11_len100_sf2.npy'
# b_idx = 3 # examples: 15
# timesteps, timesteps_ini = 100, 0
# feats = np.load(data_nba_folder + filename)[..., :2][..., timesteps_ini:timesteps+timesteps_ini,:]
# feats = np.flip(feats, axis=-2)
# lims = [feats[..., 0].min(),
# 		feats[..., 0].max(),
# 		feats[..., 1].min(),
# 		feats[..., 1].max()]
#
# for b_idx in range(feats.shape[0]):
# 	plt_i, fig_i = get_trajectory_figure(feats, lims=lims, b_idx=b_idx,
# 									 plot_type =plot_attr)
# 	# if fig_i is None: continue
#
# 	plt_i.show()
# 	save = input('Save with name: (s+(anything): saves it, e: exit, else pass) ')
# 	if save[0] == 's':
# 		plt, fig = get_trajectory_figure(feats, lims=lims, b_idx=b_idx,
# 										 plot_type =plot_attr)
#
# 		plt.savefig(data_folder + experiment_folder + save[1:] + '_figure.png', dpi=300, transparent=True)
# 		print('Saved as {}.'.format(data_folder + experiment_folder + save[1:] + '_figure.png'))
# 	elif save == 'e':
# 		print('Bye.')
# 		exit()



#### Pred plots ######
# from numpy import linalg as la
# affected_nodes = None
# node_thickness = None
# plot_attr = 'vel'
# limpos = limneg = 1
# lims = [-limneg, limpos]
# lims = None
#
#
# experiment_folder = 'pred_rec_examples/'
# filenames = os.listdir(data_folder + experiment_folder)
# filenames = [filename[:-len('gt_traj_pred.npy')] for filename in filenames if filename.endswith('gt_traj_pred.npy')]
# subnames = ['pred_all_samples', 'gt_traj_all', 'gt_traj_pred']
#
# for filename in filenames:
# 	pred = np.load(data_folder + experiment_folder + filename + subnames[0] + '.npy')
# 	gt_all = np.load(data_folder + experiment_folder + filename + subnames[1] + '.npy')
# 	# gt_pred = np.load(data_folder + experiment_folder + filename + subnames[2] + '.npy')
#
# 	vel_norm = la.norm(gt_all[..., 2:])
# 	loc_diff_norm = la.norm((gt_all[..., 1:, :2] - gt_all[..., :-1, :2]))
# 	scale = loc_diff_norm / vel_norm
# 	gt_all[..., 2:] *= scale
#
# 	iterate_range = 1
#
# 	for i in range(iterate_range):
#
# 		if iterate_range>1:
# 			traj = np.concatenate([gt_all[:,:,:-pred.shape[3]],pred[i]], axis=2)
# 		else: traj = np.concatenate([gt_all[:,:,:-pred.shape[3]],pred[-1]], axis=2); i = -1
#
# 		mark_point = gt_all.shape[2] - pred.shape[3]
#
# 		plt, fig = get_trajectory_figure(traj, lims=lims, b_idx=0,
# 							  plot_type =plot_attr,mark_point=mark_point)
#
# 		plt.show()
# 		save = input('Save with name: (s+(anything): saves it, e: exit, else pass) ')
# 		if save[0] == 's':
# 			for j, traj_j in enumerate([traj, gt_all]):
# 				plt, fig = get_trajectory_figure(traj_j, lims=lims, b_idx=0,
# 												 plot_type =plot_attr,mark_point=mark_point)
# 				if j == 0: gt = 'pred'
# 				else: gt = 'gt'
# 				plt.savefig(data_folder + experiment_folder + filename + save[1:] + '_' + str(i) + '_' + gt + '_figure.png', dpi=300)
# 				print('Saved as {}.'.format(data_folder + experiment_folder + filename + save[1:] + '_' + gt  + '_' + str(j) + '_figure.png'))
# 		elif save == 'e':
# 			print('Bye.')
# 			exit()


#### Rec plots ######
from numpy import linalg as la
affected_nodes = None
node_thickness = None
plot_attr = 'vel'
limpos = limneg = 1
lims = [-limneg, limpos]
lims = None


experiment_folder = 'pred_rec_examples/'
filenames = os.listdir(data_folder + experiment_folder)
filenames = [filename[:-len('gt_traj_pred.npy')] for filename in filenames if filename.endswith('gt_traj_pred.npy')]
subnames = ['pred_all_samples', 'gt_traj_all', 'gt_traj_pred']

for filename in filenames:
	if filename[2:6] == 'pred':
		continue
	pred = np.load(data_folder + experiment_folder + filename + subnames[0] + '.npy')
	gt_all = np.load(data_folder + experiment_folder + filename + subnames[1] + '.npy')
	# gt_pred = np.load(data_folder + experiment_folder + filename + subnames[2] + '.npy')

	vel_norm = la.norm(gt_all[..., 2:])
	loc_diff_norm = la.norm((gt_all[..., 1:, :2] - gt_all[..., :-1, :2]))
	scale = loc_diff_norm / vel_norm
	gt_all[..., 2:] *= scale

	iterate_range = gt_all.shape[-2]

	for i in range(iterate_range):

		if iterate_range>1:
			traj = np.concatenate([gt_all[:,:,:-pred.shape[3]],pred[i]], axis=2)
		else: traj = np.concatenate([gt_all[:,:,:-pred.shape[3]],pred[-1]], axis=2); i = -1

		# mark_point = gt_all.shape[2] - pred.shape[3]

		plt, fig = get_trajectory_figure(traj, lims=lims, b_idx=0,
							  plot_type =plot_attr)

		plt.show()
		save = input('Save with name: (s+(anything): saves it, e: exit, else pass) ')
		if save[0] == 's':
			for j, traj_j in enumerate([traj, gt_all]):
				plt, fig = get_trajectory_figure(traj_j, lims=lims, b_idx=0,
												 plot_type =plot_attr)
				if j == 0:
					gt = 'pred'
					plt.savefig(data_folder + experiment_folder + filename + '_' + str(i) + '_' + gt + '_figure.png', dpi=300)
				else:
					gt = 'gt'
					plt.savefig(data_folder + experiment_folder + filename  + '_' + gt + '_figure.png', dpi=300)
			print('Saved as {}.'.format(data_folder + experiment_folder + filename + '_' + gt  + '_' + str(j) + '_figure.png'))
		elif save == 'c': break
		elif save == 'e':
			print('Bye.')
			exit()