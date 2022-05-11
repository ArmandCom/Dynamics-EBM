import numpy as np
import pandas as pd
import os
csv_dir = '/data/Armand/nba-movement-data/data/csv/'
file_id = '0021500172.csv'
visualize = False
save = True
for file_id in os.listdir(csv_dir):
	data = pd.read_csv(csv_dir+file_id, usecols=['game_id', 'team_id', 'player_id', 'x_loc', 'y_loc', 'quarter', 'game_clock', 'event_id'])
	data_by_game = data.groupby('game_id')

	def compare_players_sequentially(df_i):
		grouped_by_players = []
		for ii in range(0,len(df_i)-1,11):
			if ii == 0:
				grouped_by_players.append(df_i.iloc[:11])
			if (df_i.iloc[ii:ii+11].get('player_id').to_numpy()  == grouped_by_players[-1].iloc[:11].get('player_id').to_numpy() ).sum() == 11:
				grouped_by_players[-1] =  grouped_by_players[-1].append(df_i.iloc[ii:ii+11])
			else: grouped_by_players.append(df_i.iloc[ii:ii+11])

		return grouped_by_players

	all_seconds = []
	grouped_by_players = []
	lengths = []
	count = 0
	acc_id = 0
	for game_no in data_by_game.groups.keys():
		df_game_i = data_by_game.get_group(game_no)
		data_by_event = df_game_i.groupby('event_id')
		for event_no in data_by_event.groups.keys():
			df_i = data_by_event.get_group(event_no)
			if len(all_seconds) > 0:
				# print(df_i.iloc[1]['team_id'],all_seconds[-1].iloc[1]['team_id'],
				# 	  df_i.iloc[-1]['team_id'],all_seconds[-1].iloc[-1]['team_id'], df_i.iloc[-1]['game_clock'])
				if (df_i.iloc[1]['team_id']==all_seconds[-1].iloc[1]['team_id'] and
					df_i.iloc[-1]['team_id']==all_seconds[-1].iloc[-1]['team_id'] and
					df_i.iloc[-1]['quarter']==all_seconds[-1].iloc[-1]['quarter']):


					try: assert len(df_i) % 11 == 0
					except: print('Not divisible by 11, pass'); continue
					try: assert df_i.iloc[0]['team_id'] == -1
					except: print('Divisible by 11, but first player is not -1'); continue
					groups = compare_players_sequentially(df_i)
					if (df_i.iloc[:11]['player_id'].to_numpy() == all_seconds[-1].iloc[-11:]['player_id'].to_numpy() ).sum()==11:
						# print(all_seconds[-1], groups[0])
						all_seconds[-1] = all_seconds[-1].append(groups[0], ignore_index=True)
						groups = groups[1:]

					else: all_seconds.extend(groups)
				else:
					all_seconds.extend(compare_players_sequentially(df_i))
					print('New')
			if visualize and len(all_seconds)>6:
				break
			else: all_seconds.extend(compare_players_sequentially(df_i))

	final_numpy_list = []
	for sec in all_seconds:
		players_group = sec.groupby('player_id')
		players = []
		team_group = sec.groupby('team_id')
		# for team_id in team_group.groups.keys():
		# 	print(team_id, team_group.get_group(team_id)[['player_id']])
		# 	# print(team_group.get_group(team_id))
		# 	exit()
		for player_id in players_group.groups.keys():
			# print(player_id, players_group.get_group(player_id)[['x_loc', 'y_loc']].shape)
			try:
				array = players_group.get_group(player_id)[['x_loc', 'y_loc']].to_numpy()
				players.append(array)
			except: print('Skipping'); continue
		out = np.stack(players)
		zeros = np.zeros(out[:, :, 0:1].shape)
		zeros[:1] = -1
		zeros[1:6] = 0
		zeros[6:11] = 0
		out = np.concatenate([out, zeros], axis=-1)
		final_numpy_list.append(out)

	[print(elem.shape) for elem in final_numpy_list]
	final_numpy_list = [elem for elem in final_numpy_list if elem.shape[1]>1]

	if save and not visualize:
		suffix = 'all_data'
		for save_id, data in enumerate(final_numpy_list):
			np.save(csv_dir + file_id[:-4] + '_' + suffix + '{}.npy'.format(save_id), data)


import matplotlib.pyplot as plt
def visualize_trajectories(list, list_idx=0, len=50, sr=5):
	fig = plt.figure()
	axes = plt.gca()
	lw = 1.5
	sz_pt = 30
	maps = ['bone', 'magma', 'spring', 'autumn', 'gist_gray', 'afmhot', 'cool', 'Wistia', 'YlGnBu'] #https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
	cmap = maps[4]
	alpha = 0.6
	# if lims is not None:
	# 	axes.set_xlim([lims[0], lims[1]])
	# 	axes.set_ylim([lims[0], lims[1]])
	loc = list[list_idx]
	loc = np.transpose(loc, (1,2,0))[:, :2] # L, F, No
	colors = ['b', 'r', 'c', 'y', 'k', 'm', 'g']
	modes = ['-']*loc.shape[-1]


	for i in range(loc.shape[-1]):
		# plt.scatter(loc[:, 0, i], loc[:, 1, i], s=sz_pt, c=np.arange(loc.shape[1]), cmap=cmap, alpha=alpha)
		# plt.plot(loc[-len*sr::sr, 0, i], loc[-len*sr::sr, 1, i], modes[i], linewidth=lw)
		plt.plot(loc[:len*sr:sr, 0, i], loc[:len*sr:sr, 1, i], modes[i], linewidth=lw)
		plt.plot(loc[0, 0, i], loc[0, 1, i], 'o')
	plt.show()

if visualize:
	visualize_trajectories(final_numpy_list, list_idx=-2, len=20)

# for row in range(len(df_game_i)):
	#
		# count += 1
		# if df_game_i.loc[row+1]['team_id'] == -1:
		# 	if len(all_seconds) > 0:
		# 		print(df_game_i.loc[1]['team_id'],all_seconds[-1].loc[1]['team_id'],
		# 			  df_game_i.loc[acc_id-1]['team_id'],all_seconds[-1].iloc[-1]['team_id'],
		# 			  acc_id, df_game_i.loc[acc_id-1]['game_clock'])
		# 		if (df_game_i.loc[1]['team_id']==all_seconds[-1].loc[1]['team_id'] and
		# 			df_game_i.loc[acc_id-1]['team_id']==all_seconds[-1].iloc[-1]['team_id']):
		# 			all_seconds[-1] = all_seconds[-1].append(df_game_i.loc[acc_id-10:acc_id])
		# 		else:
		# 			all_seconds.append(df_game_i.loc[acc_id-10:acc_id])
		# 			print('hi')
		# 	else: all_seconds.append(df_game_i.loc[acc_id-10:acc_id])
		# 	if count != 11:
		# 		print('Problem with game: {}', game_no)
		# 	lengths.append(count)
		# 	count = 0
		# acc_id += 1



	# time_stamps = df_game_i['game_clock'].unique()
	# df_game =df_game_i.copy()
	#
	# for i in range(0,df_game_i.shape[0],11):
	# 	# print(df_game_i['team_id'][i])
	# 	if df_game_i['team_id'][i] != -1:
	# 		print(i)
	# 		break


		# rows_t = df_game_i.loc[df_game_i['game_clock']==t]
		# df_game[loc, 'player_list'] = set(df_game_i.loc[df_game_i['game_clock']==t,'player_id'])

print('high')
	# for key in data_by_player.groups.keys():
	#     player_data = data_by_player.get_group(key).sort_values(by='game_clock', ascending=False)
	#     data_list.append(player_data.to_numpy())
	# data_np = np.cat(data_list, dim = 0)

# print(data)
# print(data_by_player)