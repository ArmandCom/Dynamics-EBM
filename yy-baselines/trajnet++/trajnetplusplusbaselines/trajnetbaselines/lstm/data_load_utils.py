import trajnetplusplustools
import os
import pickle
import numpy as np

def prepare_data(path, subset='/train/', sample=1.0, goals=True):
    """ Prepares the train/val scenes and corresponding goals 
    
    Parameters
    ----------
    subset: String ['/train/', '/val/']
        Determines the subset of data to be processed
    sample: Float (0.0, 1.0]
        Determines the ratio of data to be sampled
    goals: Bool
        If true, the goals of each track are extracted
        The corresponding goal file must be present in the 'goal_files' folder
        The name of the goal file must be the same as the name of the training file

    Returns
    -------
    all_scenes: List
        List of all processed scenes
    all_goals: Dictionary
        Dictionary of goals corresponding to each dataset file.
        None if 'goals' argument is False.
    Flag: Bool
        True if the corresponding folder exists else False.
    """

    ## Check if folder exists
    if not os.path.isdir(path + subset):
        if 'train' in subset:
            print("Train folder does NOT exist")
            exit()
        if 'val' in subset:
            print("Validation folder does NOT exist")
            return None, None, False

    ## read goal files
    all_goals = {}
    all_scenes = []

    ## List file names
    files = [f.split('.')[-2] for f in os.listdir(path + subset) if f.endswith('.ndjson')]
    ## Iterate over file names
    for file in files:
        reader = trajnetplusplustools.Reader(path + subset + file + '.ndjson', scene_type='paths')
        ## Necessary modification of train scene to add filename
        scene = [(file, s_id, s) for s_id, s in reader.scenes(sample=sample)]
        if goals:
            goal_dict = pickle.load(open('goal_files/' + subset + file +'.pkl', "rb"))
            ## Get goals corresponding to train scene
            all_goals[file] = {s_id: [goal_dict[path[0].pedestrian] for path in s] for _, s_id, s in scene}
        all_scenes += scene

    if goals:
        return all_scenes, all_goals, True
    return all_scenes, None, True

def prepare_new_data(path, subset='/train/', sample=1.0, goals=True):
    """ Prepares the train/val scenes and corresponding goals

    Parameters
    ----------
    subset: String ['/train/', '/val/']
        Determines the subset of data to be processed
    sample: Float (0.0, 1.0]
        Determines the ratio of data to be sampled
    goals: Bool
        If true, the goals of each track are extracted
        The corresponding goal file must be present in the 'goal_files' folder
        The name of the goal file must be the same as the name of the training file

    Returns
    -------
    all_scenes: List
        List of all processed scenes
    all_goals: Dictionary
        Dictionary of goals corresponding to each dataset file.
        None if 'goals' argument is False.
    Flag: Bool
        True if the corresponding folder exists else False.
    """

    path = '/data/Armand/NRI/'
    if subset == '/train/': split = 'train'
    elif subset == '/val/': split = 'valid'
    n_objects = 5
    # All hardcoded for now

    suffix = '_springs'+str(n_objects)
    # suffix += 'inter0.1_nowalls_sf100_len5000'
    suffix += 'inter0.1_sf50_lentrain5000_nstrain50000'

    ## Check if folder exists
    # if not os.path.isdir(path + subset + suffix):
    #     if 'train' in subset:
    #         print("Train folder does NOT exist")
    #         exit()
    #     if 'val' in subset:
    #         print("Validation folder does NOT exist")
    #         return None, None, False

    ## read goal files
    all_goals = {}
    all_scenes = []


    loc = np.load(path + 'loc_' + split + suffix + '.npy')
    vel = np.load(path + 'vel_' + split + suffix + '.npy')
    print('Loading from path ', path + 'loc_' + split + suffix + '.npy')

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = loc.shape[3]

    # Note: unnormalize

    loc_max = loc.max()
    loc_min = loc.min()
    vel = vel / 20
    # Note: In simulation our increase in T (delta T) is 0.001.
    #  Then we sample 1/100 generated samples.
    #  Therefore the ratio between loc and velocity is vel/(incrLoc) = 10
    vel_max = vel.max()
    vel_min = vel.min()

    print("Normalized Springs Dataset")
    # Normalize to [-1, 1]
    loc = (loc - loc_min) * 2 / (loc_max - loc_min) - 1
    vel = vel * 2 / (loc_max - loc_min)

    # print("Standardized Springs Dataset")
    # loc_mean = loc.mean()
    # loc_std = loc.std()
    # loc = (loc - loc_mean) / loc_std
    # vel = vel / loc_std

    # print("Unnormalized Spring Dataset")
    # loc_max = None
    # loc_min = None
    # vel_max = None
    # vel_min = None
    # loc = loc / 5.
    # vel = vel / 5.

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc = np.transpose(loc, [0, 1, 3, 2])
    vel = np.transpose(vel, [0, 1, 3, 2])
    # feat = np.concatenate([loc, vel], axis=3)
    feat = loc
    # edges = np.reshape(edges, [-1, num_atoms ** 2])
    # edges = np.array((edges + 1) / 2, dtype=np.int64)

    # feat = torch.FloatTensor(feat)
    # edges = torch.LongTensor(edges)

    # # Exclude self edges
    # off_diag_idx = np.ravel_multi_index(
    #     np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
    #     [num_atoms, num_atoms])
    # edges = edges[:, off_diag_idx]

    ## List file names

    return feat, None, True # TODO: Check how mins and maxes are used