import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.stats import rv_discrete
import argparse
import os


print('Reading input arguments:')
parser = argparse.ArgumentParser(description='scrath disk path input')
parser.add_argument('--scratch_path')
parser.add_argument('--n_sims', type = int)
parser.add_argument('--sim_length', type = int)
args = parser.parse_args()
scratch_path = args.scratch_path
n_sims = args.n_sims
sim_length = args.sim_length

with open(os.path.join(scratch_path,'init_state.txt'), 'r') as file:
    init_state = file.read()
init_state = json.loads(init_state)

with open(os.path.join(scratch_path,'mdp_dict_chosen.txt'), 'r') as file:
    mdp_dict = file.read()
mdp_dict = json.loads(mdp_dict)
print(mdp_dict)


def generate_strong_weak(mdp_name, feat_to_incl):
    ################## Input files #############################
    MDP = pd.read_csv(os.path.join(scratch_path,'trans_agg','agg_trans_{}.csv'.format(mdp_name)))
    policies = pd.read_csv(os.path.join(scratch_path,'policies_penalised','policy_{}.csv'.format(mdp_name)))

    #################################################################
    MDP['state'] = MDP['state'].astype(str)
    MDP['next_state'] = MDP['next_state'].astype(str)
    MDP.set_index(['state', 'action', 'next_state'], inplace = True)
    # Convert state from int to str (MATLAB exports in int)
    policies['state'] = policies['state'].astype('str')
    policies.set_index('state', inplace = True)
    policy = policies['policy']

    strong_student = MDP.copy()
    strong_student.reset_index(inplace = True)
    strong_student['delta'] = 0

    intermediate_student = MDP.copy()
    intermediate_student.reset_index(inplace = True)
    intermediate_student['delta'] = 0

    weak_student = MDP.copy()
    weak_student.reset_index(inplace = True)
    weak_student['delta'] = 0

    delta = 0.05

    # Note I use strong_student's state idx, which is identical to MDP and weak_student state
    # Topic fam modification
    current_topic_fam = strong_student['state'].str[0].astype('int')
    next_topic_fam = strong_student['next_state'].str[0].astype('int')
    strong_student.loc[next_topic_fam > current_topic_fam, 'delta'] += delta
    intermediate_student.loc[next_topic_fam > current_topic_fam, 'delta'] += delta
    weak_student.loc[next_topic_fam == current_topic_fam, 'delta'] += delta

    # Correct so far modification
    current_correct_ans = strong_student['state'].str[1].astype('int')
    next_correct_ans = strong_student['next_state'].str[1].astype('int')
    strong_student.loc[next_correct_ans >= current_correct_ans, 'delta'] += delta
    intermediate_student.loc[next_correct_ans == current_correct_ans, 'delta'] += delta
    weak_student.loc[next_correct_ans < current_correct_ans, 'delta'] += delta

    # Avg time modification
    current_avg_time = strong_student['state'].str[2].astype('int')
    next_avg_time = strong_student['next_state'].str[2].astype('int')
    strong_student.loc[next_avg_time <= current_avg_time, 'delta'] += delta
    intermediate_student.loc[next_avg_time == current_avg_time, 'delta'] += delta
    weak_student.loc[next_avg_time > current_avg_time, 'delta'] += delta

    if comp == 'final':
        print('Modifying expl received, av fam, prev correct & ssl')
        # Explanations modification
        current_expl = strong_student['state'].str[3].astype('int')
        next_expl = strong_student['next_state'].str[3].astype('int')
        strong_student.loc[next_expl > current_expl, 'delta'] += delta
        intermediate_student.loc[next_expl == current_expl, 'delta'] += delta
        weak_student.loc[next_expl < current_expl, 'delta'] += delta

        # Av fam modification
        current_avfam = strong_student['state'].str[6].astype('int')
        next_avfam = strong_student['next_state'].str[6].astype('int')
        strong_student.loc[next_avfam > current_avfam, 'delta'] += delta
        intermediate_student.loc[next_avfam == current_avfam, 'delta'] += delta
        weak_student.loc[next_avfam < current_avfam, 'delta'] += delta

        # prev_correct modification
        next_pc = strong_student['next_state'].str[5].astype('int')
        strong_student.loc[next_pc == '1', 'delta'] += delta
        weak_student.loc[next_pc == '0', 'delta'] += delta

        # ssl modification
        current_ssl = strong_student['state'].str[4].astype('int')
        next_ssl = strong_student['next_state'].str[4].astype('int')
        strong_student.loc[next_ssl < current_ssl, 'delta'] += delta
        intermediate_student.loc[next_ssl == current_ssl, 'delta'] += delta
        weak_student.loc[next_ssl > current_ssl, 'delta'] += delta

    # Calculating Relative delta
    mean_delta = strong_student.groupby(['state', 'action'], observed = True).mean()['delta'].rename('mean_delta')
    strong_student = strong_student.merge(mean_delta, how = 'left', left_on = ['state', 'action'], right_index = True)
    strong_student['rel_delta'] = strong_student['delta'] - strong_student['mean_delta']
    mean_delta = weak_student.groupby(['state', 'action'], observed = True).mean()['delta'].rename('mean_delta')
    weak_student = weak_student.merge(mean_delta, how = 'left', left_on = ['state', 'action'], right_index = True)
    weak_student['rel_delta'] = weak_student['delta'] - weak_student['mean_delta']
    mean_delta = intermediate_student.groupby(['state', 'action'], observed = True).mean()['delta'].rename('mean_delta')
    intermediate_student = intermediate_student.merge(mean_delta, how = 'left', left_on = ['state', 'action'], right_index = True)
    intermediate_student['rel_delta'] = intermediate_student['delta'] - intermediate_student['mean_delta']

    # Applying rel delta to transition probs
    strong_student['transition_prob'] += strong_student['rel_delta']
    weak_student['transition_prob'] += weak_student['rel_delta']
    intermediate_student['transition_prob'] += intermediate_student['rel_delta']
    strong_student.loc[strong_student['transition_prob'] < 0, 'transition_prob'] = 0
    weak_student.loc[weak_student['transition_prob'] < 0, 'transition_prob'] = 0
    intermediate_student.loc[intermediate_student['transition_prob'] < 0, 'transition_prob'] = 0

    # Normalizing new transition probs.
    normalizer_strong = strong_student.groupby(['state', 'action'], observed = True).sum()['transition_prob'].rename('normalizer')
    normalizer_weak = weak_student.groupby(['state', 'action'], observed = True).sum()['transition_prob'].rename('normalizer')
    normalizer_intermediate = intermediate_student.groupby(['state', 'action'], observed = True).sum()['transition_prob'].rename('normalizer')
    strong_student = strong_student.merge(normalizer_strong, how ='left', left_on = ['state', 'action'], right_index=True)
    weak_student = weak_student.merge(normalizer_weak, how ='left', left_on = ['state', 'action'], right_index=True)
    intermediate_student = intermediate_student.merge(normalizer_intermediate, how ='left', left_on = ['state', 'action'], right_index=True)
    strong_student['transition_prob'] = strong_student['transition_prob']/strong_student['normalizer']
    weak_student['transition_prob'] = weak_student['transition_prob']/weak_student['normalizer']
    intermediate_student['transition_prob'] = intermediate_student['transition_prob']/intermediate_student['normalizer']

    strong_student.drop(columns = ['normalizer','rel_delta','delta', 'mean_delta'], inplace = True)
    weak_student.drop(columns = ['normalizer', 'rel_delta', 'delta', 'mean_delta'], inplace = True)
    intermediate_student.drop(columns = ['normalizer', 'rel_delta', 'delta', 'mean_delta'], inplace = True)
    strong_student.set_index(['state', 'action', 'next_state'], inplace = True)
    weak_student.set_index(['state', 'action', 'next_state'], inplace = True)
    intermediate_student.set_index(['state', 'action', 'next_state'], inplace = True)

    return MDP, strong_student, weak_student, intermediate_student, policy


def simulator(MDP, policy,  sim_length, starting_state = None):
    simulation_length = sim_length
    if starting_state is None:
        state = MDP.index.get_level_values(0)[0]
    else:
        state = starting_state
    cumul_reward = 0
    cumreward_per_step = [0]
    discount_factor = 0.99

    for n in range(simulation_length):
        action = policy.loc[state]
        pk = MDP.loc[state].loc[action]['transition_prob'].values
        # Convert states to int, to accomodate scipy's requirement
        xk = np.array(MDP.loc[state].loc[action].index).astype('int64')
        custm = rv_discrete(name='custm', values=(xk, pk))
        # Convert n_state back to str
        n_state = (custm.rvs(size = 1)[0]).astype('str')
        reward = MDP.loc[state,action,n_state]['reward']
        cumul_reward += reward*discount_factor**n
        state = n_state
        cumreward_per_step = np.append(cumreward_per_step, cumul_reward)

    return cumreward_per_step



def run_plot(original, strong, weak, intermediate,policy, n_sims, sim_steps, ax):
    np.random.seed(100)

    results_original = []
    results_strong = []
    results_intermediate = []
    results_weak = []

    for i in range(n_sims):
        results_original.append(simulator(original, policy, sim_steps, starting_state=starting_state))
        results_strong.append(simulator(strong, policy, sim_steps, starting_state=starting_state))
        results_intermediate.append(simulator(intermediate, policy, sim_steps, starting_state=starting_state))
        results_weak.append(simulator(weak, policy, sim_steps, starting_state=starting_state))

    rewards_record = pd.DataFrame()
    rewards_record['original'] = results_original
    rewards_record['strong'] = results_strong
    rewards_record['intermediate'] = results_intermediate
    rewards_record['weak'] = results_weak
    rewards_record = rewards_record.apply(pd.Series.explode)
    rewards_record['step'] = list(range(sim_length + 1))*n_sims
    rewards_record = rewards_record.melt(value_vars=['original', 'strong', "intermediate",'weak'], id_vars = ['step'], var_name = 'student_type',
        value_name='cumulative reward from initial state')

    rewards_record['cumulative reward from initial state'] = rewards_record['cumulative reward from initial state'].astype('float')

    sns.lineplot(data = rewards_record, x = 'step', y = 'cumulative reward from initial state',
        hue = 'student_type', ax = ax)
    if comp == 'final':
        ax.set_title('{}: All features perturbed'.format(mdp_name))
    else:
        ax.set_title('{}'.format(mdp_name))


comp = 'not_final'

fig, axs = plt.subplots(2, 3, figsize=(15,10))
axs = axs.flatten()
# Defining custom 'ylim' values.
custom_ylim = (0, 400)
plt.setp(axs, ylim=custom_ylim)

for ax, (mdp_name, feat_to_incl) in enumerate(mdp_dict.items()):
    print(mdp_name)
    starting_state = ''
    for feat in feat_to_incl:
        starting_state += init_state[feat]

    original, strong, weak, intermediate,policy = generate_strong_weak(mdp_name = mdp_name, feat_to_incl=feat_to_incl)

    run_plot(original,strong,weak,intermediate,policy,n_sims,sim_length,axs[ax])


# mdp_name= list(mdp_dict.keys())[-1]
# feat_to_incl = mdp_dict[mdp_name]
# print('{}: All feat pert.'.format(mdp_name))
# starting_state = ''
# for feat in feat_to_incl:
#     starting_state += init_state[feat]
#
# comp = 'final' # Set this to 'final' to pass if statement in 'simulator'
# original, strong, weak, intermediate,policy = generate_strong_weak(mdp_name = mdp_name, feat_to_incl=feat_to_incl)
# run_plot(original,strong,weak,intermediate,policy,n_sims,sim_length,axs[ax + 1])


fig.suptitle('Cumulative reward as episode progresses under policy with 95% CI across {} runs'.format(n_sims))
plt.savefig('export_data/strong_weak.pdf')
