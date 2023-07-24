import pandas as pd
import numpy as np
import scipy.io
from scipy import sparse
import json
import argparse
import os

print('Reading input arguments:')
parser = argparse.ArgumentParser(description='scrath disk path input')
parser.add_argument('--scratch_path')
parser.add_argument('--penalise', type = bool, default=False)
args = parser.parse_args()
scratch_path = args.scratch_path
penalise = args.penalise

with open(os.path.join(scratch_path,'mdp_dict.txt'), 'r') as file:
    mdp_dict = file.read()
mdp_dict = json.loads(mdp_dict)

with open(os.path.join(scratch_path,'init_state.txt'), 'r') as file:
    init_state = file.read()
init_state = json.loads(init_state)

features = list(init_state.keys())
complete_trans = pd.read_csv(os.path.join(scratch_path,'KT3_complete_transitions.csv'))

# In case state type is not string
complete_trans['state'] = complete_trans['state'].astype('int64').astype('str')
complete_trans['next_state'] = complete_trans['next_state'].astype('int64').astype('str')

obs_state = complete_trans['state'].str.split('', expand = True).values[:,1:-1]
obs_nstate = complete_trans['next_state'].str.split('', expand = True).values[:,1:-1]

def get_MDP(mdp_name,feat_to_incl, complete_trans, missing_start_state):
    transitions = pd.DataFrame()
    # Finding feature index
    feat_idx = [features.index(feat) for feat in feat_to_incl]

    # Extracting relevant features
    transitions['state'] = np.sum(obs_state[:,feat_idx], axis = 1)
    transitions['next_state'] = np.sum(obs_nstate[:,feat_idx], axis = 1)
    transitions['reward'] = complete_trans.reward.values
    transitions['action'] = complete_trans.action.values
    transitions['transitions'] = complete_trans.transitions.values

    # Removing non-actionable states
    transitions.set_index(['state', 'action', 'next_state', 'reward'], inplace = True)
    state_unique = transitions.index.get_level_values(0).unique()
    nstate_unique = transitions.index.get_level_values(2).unique()
    counter = 0
    while state_unique.size != nstate_unique.size:
        if state_unique.size > nstate_unique.size:
            disjoint_state = state_unique[np.where([i not in nstate_unique for i in state_unique])]
            if disjoint_state.item() == starting_state:
                print('Starting state not present in next_state')
                missing_start_state = True
                break
        non_action = nstate_unique[np.where([i not in state_unique for i in nstate_unique])]
        print('Dropping transition with non_action state: {}'.format(non_action.tolist()))
        transitions.drop(index = non_action, level = 2, inplace = True)
        state_unique = transitions.index.get_level_values(0).unique()
        nstate_unique = transitions.index.get_level_values(2).unique()
        counter += len(non_action)
    print('Num of states dropped: {}'.format(counter))
    transitions.reset_index(inplace = True)

    MDP = transitions.groupby(['state', 'action','next_state']).sum()['transitions']
    normalizer = MDP.groupby(level = [0,1]).sum().rename('normalizer')
    MDP = pd.DataFrame(MDP).reset_index()
    MDP = MDP.merge(normalizer, how= 'left', left_on = ['state','action'], right_index=True)
    MDP['transition_prob'] = MDP['transitions']/MDP['normalizer']

    rewards = transitions.groupby(['state', 'action','next_state' , 'reward']).sum()['transitions']
    reward_normalizer = rewards.groupby(level = [0,1,2]).sum().rename('reward_normalizer')
    rewards = pd.DataFrame(rewards).reset_index()
    rewards = rewards.merge(reward_normalizer, how = 'left', left_on = ['state', 'action', 'next_state'], 
                        right_index=True)
    rewards['reward_normalizer'] = rewards['transitions']/rewards['reward_normalizer']
    rewards['weighted_reward'] = rewards['reward'] * rewards['reward_normalizer']
    weighted_rewards = rewards.groupby(['state','action','next_state']).sum()['weighted_reward']

    MDP.set_index(['state','action','next_state'], inplace = True)
    MDP['reward'] = weighted_rewards
    MDP.drop(columns = ['normalizer', 'transitions'], inplace = True)

    transitions.rename(columns = {'transitions':'support'}, inplace = True)
    transitions = transitions.groupby(['state', 'action', 'next_state', 'reward']).sum()
    transitions.to_csv('trans_ori/original_transitions_{}.csv'.format(mdp_name))
    del transitions

    MDP.to_csv('trans_agg/agg_trans_{}.csv'.format(mdp_name))

    return MDP, missing_start_state

for mdp_name, feat_to_incl in mdp_dict.items():
    print('Current MDP: {}'.format(mdp_name))
    missing_start_state = False
    starting_state = ''
    for feat in feat_to_incl:
        starting_state += init_state[feat]
    MDP, missing_start_state = get_MDP(mdp_name, feat_to_incl, complete_trans, missing_start_state)
    # create an empty array of NaN of the right dimensions
    shape = list(map(len, MDP.index.levels))
    ind = np.array(list(MDP.index.codes))
    state_idx = MDP.index.levels[0].array
    # To ensure a square matrix and correct indexing
    if missing_start_state:
        shape[2] += 1
        starting_state_idx = np.where(state_idx == starting_state)[0]
        ind[2,:][ind[2,:] >= starting_state_idx] += 1

    if shape[0] != shape[2]:
        raise Exception('Matrix not Square!!')
    P_matrix = np.array([sparse.lil_matrix((shape[0], shape[2]),dtype='float64') for act_dim in range(shape[1])])
     # create an empty array of NaN of the right dimensions
    R_matrix = np.array([sparse.lil_matrix((shape[0], shape[2]),dtype='float64') for act_dim in range(shape[1])])

    unique_states = np.unique(ind[0])
    for a in range(37):
        action_ind = np.where(ind[1] == a)[0]
        temp = ind[::2,action_ind]
        P_matrix[a][tuple(temp)] = MDP['transition_prob'].values.flat[action_ind]
        R_matrix[a][tuple(temp)] = MDP['reward'].values.flat[action_ind]
        if penalise:
            observed_states = ind[0,action_ind]
            OOD_sa = np.setdiff1d(unique_states, observed_states)
            # Penalise OOD sa tuples
            R_matrix[a][OOD_sa,OOD_sa] = -9999
            P_matrix[a][OOD_sa,OOD_sa] = 1



    mdic = {"P": P_matrix, "R": R_matrix, 'state_idx': state_idx.astype('int64').tolist(),
        'name':mdp_name}
    if penalise:
        scipy.io.savemat('matlab_files_penalised/{}.mat'.format(mdp_name), mdic)
    else:
        scipy.io.savemat('matlab_files/{}.mat'.format(mdp_name), mdic)