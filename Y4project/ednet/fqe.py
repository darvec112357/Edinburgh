from d3rlpy.ope import DiscreteFQE
from d3rlpy.algos import DiscreteCQL
from d3rlpy.algos import DiscreteBCQ
from d3rlpy.dataset import MDPDataset
import numpy as np
import pandas as pd
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
from sklearn.model_selection import train_test_split
from d3rlpy.models.encoders import VectorEncoderFactory
import argparse
import os

print('Reading input arguments:')
parser = argparse.ArgumentParser(description='scrath disk path input')
parser.add_argument('--scratch_path')
parser.add_argument('--exp_name')
parser.add_argument('--n_epochs', type = int, default=10)
parser.add_argument('--algo', default='cql')
parser.add_argument('--batch_size', type = int, default=32)
parser.add_argument('--q_function', default='mean')
parser.add_argument('--dropout_rate', type=float, default=None)
parser.add_argument('--batch_norm', type=bool, default=False)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--target_int', type=int, default=100)
parser.add_argument('--eps', type=float, default=1e-8)

args = parser.parse_args()
scratch_path = args.scratch_path
n_epochs = args.n_epochs
exp_name = args.exp_name
algo = args.algo
batch_size = args.batch_size
q_function = args.q_function
dropout_rate = args.dropout_rate
batch_norm = args.batch_norm
lr = args.lr
target_int = args.target_int
eps = args.eps

##### Loading & Setting up Data ##########################################################################################
dataset = MDPDataset.load(os.path.join(scratch_path,'dataset_KT3_200k.h5'))

train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

######## Loading Policy Iteration/Behaviour Policy ############################################
if algo == 'pi':
    model_name = 'policy_MDP_aug4687'
    PI_policy = pd.read_csv(os.path.join(scratch_path,'policies_penalised', '{}.csv'.format(model_name)))
    PI_policy['state'] = PI_policy['state'].astype('str')
    num_features = len(PI_policy.loc[0,'state'])
    for i in range(num_features):
        PI_policy['obs_{}'.format(i)] = PI_policy['state'].str[i].astype(int)
    PI_policy.set_index(['obs_{}'.format(i) for i in range(num_features)], inplace = True)
    PI_policy.name = model_name
    PI_policy.stochastic = False
elif algo == 'bp':
    model_name = 'MDP_aug4687'
    transitions = pd.read_csv(os.path.join(scratch_path,'trans_ori','original_transitions_{}.csv'.format(model_name)))
    transitions['state'] = transitions['state'] .astype('str')
    transitions['next_state'] = transitions['next_state'] .astype('str')

    sa_count = transitions.groupby(['state','action']).sum()['support']
    s_count = transitions.groupby(['state']).sum()['support']
    behaviour_policy = sa_count/s_count
    behaviour_policy = behaviour_policy.reset_index(level = [1])
    behaviour_policy = behaviour_policy.max(axis = 0, level= 0)['action']
    behaviour_policy.rename('policy', inplace=True)
    behaviour_policy = behaviour_policy.reset_index()
    PI_policy = behaviour_policy.copy()
    PI_policy.name = model_name
    num_features = len(PI_policy['state'][0])
    for i in range(num_features):
        PI_policy['obs_{}'.format(i)] = PI_policy['state'].str[i].astype(int)
    PI_policy.set_index(['obs_{}'.format(i) for i in range(num_features)], inplace = True)
    PI_policy.name = model_name
    PI_policy.stochastic = False
else:
    PI_policy = None

############ Loading & Setting up Algo ###############################################
if algo == 'bcq':
    model = DiscreteBCQ.from_json(os.path.join(scratch_path,'optimum_offline/params.json'))
else:
    model = DiscreteCQL.from_json(os.path.join(scratch_path,'optimum_offline/params.json'))

model.load_model(os.path.join(scratch_path,'optimum_offline/model_10.pt'))

fqe = DiscreteFQE(algo = model, external_policy=PI_policy, use_gpu=True, scaler='min_max', batch_size=batch_size, 
    q_func_factory=q_function, encoder_factory=VectorEncoderFactory(dropout_rate=dropout_rate, use_batch_norm=batch_norm),
    learning_rate=lr, target_update_interval=target_int, eps = eps)

results = fqe.fit(train_episodes, eval_episodes=test_episodes,
        scorers={
           'init_value': initial_state_value_estimation_scorer,
        }, n_epochs=n_epochs, experiment_name=exp_name)