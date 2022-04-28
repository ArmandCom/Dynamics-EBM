import os,shutil
from pathlib import Path


root='/data/Armand/EBM/cachedir/charged/'
destination="/tmp"
directory = os.path.join(root,"experiments")
os.chdir(directory)
min_saved_iteration = 10000

for dir in os.listdir("."):
	found = False
	model_exists = False
	for file in os.listdir("./" + dir):
		try: model_num = int(file[6:-4]); model_exists = True
		except: continue
		if model_num > min_saved_iteration:
			found = True
			break
	if not found and model_exists:
		print('About to remove dir {} \nwith the following files:'.format(dir))
		[print('- ' + filename) for filename in os.listdir("./" + dir)]
		asw = input('Remove?')
		if asw == 'y' or asw == 'Y':
			try:
				shutil.rmtree(Path("./" + dir))
			except OSError as e:
				print("Error: %s : %s" % (dir, e.strerror))
			print('Removed')
		else: print('Not removed')
