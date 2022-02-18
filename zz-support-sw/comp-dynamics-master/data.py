import csv
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

#event_acc = EventAccumulator(sub_folder)
#event_acc.Reload()
# Show all tags in the log file
#print(event_acc.Tags())

# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
#w_times, step_nums, vals = zip(*event_acc.Scalars('Accuracy'))

parser = argparse.ArgumentParser()
parser.add_argument('--mse', default=True, action='store_true')
parser.add_argument('--inf', dest='mse', action='store_false')
parser.add_argument('--inference', dest='mse', action='store_false')
parser.add_argument('--traj', default='both')
args = parser.parse_args()

mse = args.mse
trajectories = args.traj

if mse:
    filename = 'MSE_error'
else:
    filename = 'inference'
if trajectories != 'both':
    filename = filename + '_%s' %trajectories
    array = [int(trajectories)]
else:
    array = [1, 50000]
print(filename)

with open('%s.csv' %filename, mode='w') as file:
    writer = csv.writer(file, delimiter=',')

    if mse:
        writer.writerow(['Trajectories', 'Separate', 'Activation', 'Characteristic', 'Langevin Steps', 'Step Size', 'Training Iterations', 'Position', 'Accuracy'])
    else:
        writer.writerow(['Trajectories', 'Separate', 'Activation', 'Characteristic', 'Langevin Steps', 'Step Size', 'Dataset', 'Training Iterations', 'Incorrect', 'Accuracy'])

    for trajectories in array:
        for activation in ['ReLU', 'Swish']:
            for characteristic in ['None', 'Buffer', 'Inertia', 'Anti', 'All', 'No Backprop']:
                if activation == 'ReLU':
                    folder = '%d_separate_langevin_5_1' %trajectories
                elif activation == 'Swish':
                    folder = '%d_separate_langevin_5_1_swish' %trajectories
                if characteristic in ['Anti', 'All']:
                    folder = folder + '_anti'
                if characteristic in ['Buffer', 'All']:
                    folder = folder + '_persistent'
                if characteristic in ['Inertia', 'All']:
                    folder = folder + '_inertia'
                folder = folder + '_lr5'
                if characteristic == 'No Backprop':
                    folder = folder + '_noprop'
                folder = folder + '/'
                for steps in ([40, 200] if characteristic == 'None' else [40]):
                    for step_size in ['0.1', '1', '10', '100', '1000']:
                        if activation == 'Swish' and step_size == '1000':
                            if steps == 200 or characteristic not in ['None', 'Buffer', 'Inertia']:
                                continue
                        sub_folder = folder + '%d_step%s_noise0/' %(steps, step_size)
                        print(sub_folder)
                        event_acc = EventAccumulator(sub_folder)
                        event_acc.Reload()
                        if mse:
                            for position in [11, 31, 51, 61, 71, 81, 91]:
                                string = ('Train' if trajectories == 1 else 'Test')
                                for i in range(5):
                                    w_times, train_step_nums, vals = zip(*event_acc.Scalars('%s_Random_Error_Step_%d' %(string, position+i)))
                                    for train_step in train_step_nums:
                                        writer.writerow(['%d' %trajectories, '1', '%s' %activation, '%s' %characteristic, '%d' %steps, '%s' %step_size, \
                                                         '%d' %train_step, '%d' %(position+i), '%f' %vals[train_step]])
                        else:
                            for dataset in ['Train', 'Test']:
                                w_times, train_step_nums, vals = zip(*event_acc.Scalars('Num_Incorrect_Mean_%s' %dataset))
                                for train_step in train_step_nums:
                                    incorrect = vals[train_step]
                                    accuracy = 1 - (incorrect / 20.0)
                                    writer.writerow(['%d' %trajectories, '1', '%s' %activation, '%s' %characteristic, '%d' %steps, '%s' %step_size, \
                                                    '%s' %dataset, '%d' %train_step, '%f' %incorrect, '%f' %accuracy])
