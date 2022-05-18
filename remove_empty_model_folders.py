import os,shutil
from pathlib import Path


root='/data/Armand/EBM/cachedir/experiments/charged/'
destination="/tmp"
directory = os.path.join(root, "")
os.chdir(directory)
min_saved_iteration = 10000

def remove_more():
	dirname = input('Other names to be removed? (n: No, dir_name: remove dir)')
	if dirname == 'n':
		print('OK, bye')
		exit()
	else:
		try:
			if dirname in os.listdir("./"):
				asw = input('Sure you want to remove {}?'.format(dirname))
				if asw == 'y':
					shutil.rmtree(Path("./" + dirname))
					print('Removed')
		except OSError as e:
			print('Couldnt remove ' + "./" + dir + '\n')
			print("Error: %s : %s" % (dir, e.strerror))
		remove_more()

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
remove_more()
