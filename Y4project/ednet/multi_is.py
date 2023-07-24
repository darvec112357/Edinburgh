import pandas as pd
import numpy as np
import json
import os, re

############# Inputs ###########################3
with open('init_state.txt', 'r') as file:
    init_state = file.read()
init_state = json.loads(init_state)
features = list(init_state.keys())

with open('mdp_dict.txt', 'r') as file:
    mdp_dict = file.read()
mdp_dict = json.loads(mdp_dict)

# Optimum representation for CQL
optimum_representation = 'MDP_aug4687'

nrows = None # For debugging
complete_traj = pd.read_csv('trajectories_KT3_200k.csv', nrows=nrows)

###### Converting states to str in case #######################
complete_traj['state'] = complete_traj['state'] .astype('str')
complete_traj['next_state'] = complete_traj['next_state'] .astype('str')

# ########### Limiting to first x users #############################
limit = 50000
user_subset = complete_traj['username'].unique()[:limit]
complete_traj.query('username in @user_subset', inplace=True)

obs = complete_traj['state'].str.split('', expand = True).values[:,1:-1]
rewards = complete_traj.reward.values
n_obs = complete_traj['next_state'].str.split('', expand = True).values[:,1:-1]
actions = complete_traj.action.values
usernames = complete_traj.username.values

del complete_traj

def get_is(model_name, trajectories, transform_type, policy_path = 'policies', cql = False):

    trajectories = pd.DataFrame(trajectories, columns=['username','state', 'action', 'next_state', 'reward'])
    trajectories['action'] = trajectories['action'].astype('int64')
    trajectories['reward'] = trajectories['reward'].astype('float64')
    
    if 'MDP' in model_name:
        transitions = pd.read_csv('trans_ori/original_transitions_{}.csv'.format(model_name))
    else:
        # CQL policies
        transitions = pd.read_csv('trans_ori/original_transitions_{}.csv'.format(optimum_representation))

    q_vals = pd.read_csv('{}/Q_{}.csv'.format(policy_path, model_name), skiprows=1, header=None)
    target_policy = pd.read_csv('{}/policy_{}.csv'.format(policy_path, model_name))

    transitions['state'] = transitions['state'] .astype('str')
    transitions['next_state'] = transitions['next_state'] .astype('str')

    # Changing state back to str (from MATLAB)
    target_policy['state'] = target_policy['state'].astype('str')

    # Deriving Behaviour Policy
    sa_count = transitions.groupby(['state','action']).sum()['support']
    s_count = transitions.groupby(['state']).sum()['support']
    observed_policy = sa_count/s_count
    observed_policy.rename('observed_policy', inplace = True)

    # Deriving stochastic target policy
    target_policy.rename(columns = {'policy':'action'}, inplace = True)
    target_policy.set_index(['state', 'action'], inplace = True)
    action_size = transitions['action'].unique().shape[0]

    if transform_type == 'hardcode':
        # Hard-code transformation
        eps = 0.1
        target_policy['target_policy'] = 1 - eps
        policies_to_compare = pd.DataFrame(observed_policy)
        policies_to_compare = policies_to_compare.merge(target_policy['target_policy'], how = 'left', left_index = True, right_index = True)
        policies_to_compare.fillna(eps/action_size, inplace = True)
        
    elif transform_type == 'softmax':
        prob_soft_max = np.exp(q_vals*0.01).values/np.exp(q_vals*0.01).values.sum(axis = 1).reshape(-1,1)
        prob_soft_max = pd.DataFrame(prob_soft_max, index = target_policy.index.get_level_values(0))
        prob_soft_max = prob_soft_max.melt(value_vars= prob_soft_max.columns,var_name='action', value_name = 'target_policy', ignore_index=False)
        prob_soft_max.set_index('action', append = True, inplace = True)
        policies_to_compare = pd.DataFrame(observed_policy)
        policies_to_compare = policies_to_compare.merge(prob_soft_max, how = 'left', left_index = True, right_index = True)
        
    trajectories_mod = trajectories.merge(policies_to_compare, how = 'left', left_on = ['state', 'action'], right_index = True)

    # Computing avg. discounted reward for behaviour policy
    discount_rate = 0.99
    episode_steps = trajectories_mod.groupby('username').cumcount()
    step_discount_factor = discount_rate**episode_steps
    trajectories_mod['disc_reward'] = trajectories_mod['reward']*step_discount_factor
    disc_return_per_user = trajectories_mod.groupby('username').sum()['disc_reward']
    av_disc_return = disc_return_per_user.mean()
    print('Average discounted return is {}'.format(av_disc_return))

    ################ Ordinary Importance Sampling ##############################################
    trajectories_mod['norm_prob'] = trajectories_mod['target_policy']/trajectories_mod['observed_policy']
    # NOTE I add epsilon here to avoid 0 prob. trajectories_mod as a result of vanishing problem
    eps = 1e-3
    prob_per_user = trajectories_mod.groupby('username').prod()['norm_prob'] + eps
    # Remove problematic users
    prob_per_user.replace([np.inf, -np.inf], np.nan, inplace=True)
    is_per_user = disc_return_per_user * prob_per_user
    standard_is = is_per_user.mean()
    standard_is_var = is_per_user.var()

    ################ Weighted Importance Sampling ##############################################
    # Filter out problematic users
    prob_per_user.replace([np.inf, -np.inf], np.nan, inplace=True)
    weighted_is = (disc_return_per_user * prob_per_user).sum()/prob_per_user.sum()

    ################ Per-Dis Importance Sampling ##############################################
    cumprob_in_traj = trajectories_mod.groupby('username').cumprod()['norm_prob'] + eps
    trajectories_mod['pdis_per_step'] = cumprob_in_traj * trajectories_mod['disc_reward']
    pdis_per_user = trajectories_mod.groupby('username').sum()['pdis_per_step']
    trajectories_mod.drop(columns = ['pdis_per_step'], inplace = True)
    # Filter out problematic users
    pdis_per_user.replace([np.inf, -np.inf], np.nan, inplace=True)
    pdis = pdis_per_user.mean()
    pdis_var = pdis_per_user.var()

    return standard_is, standard_is_var, pdis, pdis_var, weighted_is


    
### PI policies #################################################################
results = pd.DataFrame()
pen_results = pd.DataFrame()
for mdp_name, feat_to_incl in mdp_dict.items():
    print('Current MDP: {}'.format(mdp_name))
    # Finding feature index
    feat_idx = [features.index(feat) for feat in feat_to_incl]
    trajectories = [usernames, obs[:,feat_idx].sum(axis = 1), actions, n_obs[:,feat_idx].sum(axis = 1), rewards]
    trajectories = np.array(trajectories).transpose()
    
    standard_is, standard_is_var, pdis, pdis_var, weighted_is = \
        get_is(mdp_name, trajectories, transform_type='hardcode', policy_path='policies')

    results.loc[mdp_name,'OIS'] = standard_is
    results.loc[mdp_name,'OIS Var'] = standard_is_var
    results.loc[mdp_name,'PDIS'] = pdis
    results.loc[mdp_name,'PDIS Var'] = pdis_var
    results.loc[mdp_name,'WIS'] = weighted_is


    print('Current Penalised MDP: {}'.format(mdp_name))
    # !! Check whether trajectories is affected
    standard_is, standard_is_var, pdis, pdis_var, weighted_is = \
        get_is(mdp_name, trajectories, transform_type='hardcode', policy_path='policies_penalised')

    pen_results.loc[mdp_name,'OIS'] = standard_is
    pen_results.loc[mdp_name,'OIS Var'] = standard_is_var
    pen_results.loc[mdp_name,'PDIS'] = pdis
    pen_results.loc[mdp_name,'PDIS Var'] = pdis_var
    pen_results.loc[mdp_name,'WIS'] = weighted_is

##### CQL ###############################################
print('CQL')

# Optimal feature set
feat_to_incl = mdp_dict[optimum_representation]
feat_idx = [features.index(feat) for feat in feat_to_incl]
trajectories = [usernames, obs[:,feat_idx].sum(axis = 1), actions, n_obs[:,feat_idx].sum(axis = 1), rewards]
trajectories = np.array(trajectories).transpose()

cql_results = pd.DataFrame()
models = os.listdir('d3rlpy_results/cql/')
for model in models:
    model_name = re.sub(r'_2021.*$',"", model)
    print('Current offline model: {}'.format(model_name))
    standard_is, standard_is_var, pdis, pdis_var, weighted_is = \
            get_is(model_name, trajectories, transform_type='hardcode', policy_path='policies_CQL')

    cql_results.loc[model_name,'OIS'] = standard_is
    cql_results.loc[model_name,'OIS Var'] = standard_is_var
    cql_results.loc[model_name,'PDIS'] = pdis
    cql_results.loc[model_name,'PDIS Var'] = pdis_var
    cql_results.loc[model_name,'WIS'] = weighted_is


with pd.ExcelWriter('Results/importance_sampling_{}k.xlsx'.format(int(limit/1000))) as writer: 
    results.to_excel(writer, 'normal_policy')
    pen_results.to_excel(writer, 'penalised_policy')
    cql_results.to_excel(writer,'cql_policy')
