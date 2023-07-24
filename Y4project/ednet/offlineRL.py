from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteCQL
from d3rlpy.algos import DiscreteBCQ
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.metrics.scorer import discrete_action_match_scorer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import glob
import os
import json

print('Reading input arguments:')
parser = argparse.ArgumentParser(description='scrath disk path input')
parser.add_argument('--scratch_path')
parser.add_argument('--exp_name')
parser.add_argument('--data')
parser.add_argument('--n_epochs', type = int, default=10)
parser.add_argument('--batch_size', type = int, default=32)
parser.add_argument('--q_function', default='mean')
parser.add_argument('--dropout_rate', type=float, default=None)
parser.add_argument('--batch_norm', type=bool, default=False)
parser.add_argument('--lr', type=float, default=6.25e-05)
parser.add_argument('--limit_features', type=bool, default = False)
parser.add_argument('--n_steps', type=int, default=1)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--algo', default='cql')


args = parser.parse_args()
scratch_path = args.scratch_path
n_epochs = args.n_epochs
exp_name = args.exp_name
data = args.data
batch_size = args.batch_size
q_function = args.q_function
dropout_rate = args.dropout_rate
batch_norm = args.batch_norm
lr = args.lr
limit_features = args.limit_features
n_steps = args.n_steps
alpha = args.alpha
algo = args.algo


with open(os.path.join(scratch_path, 'init_state.txt'), 'r') as file:
    init_state = file.read()
init_state = json.loads(init_state)
features = list(init_state.keys())


if data == 'write':
    train_data = pd.read_csv(os.path.join(scratch_path, 'CQL_train_data.csv'))
    obs = train_data.drop(columns=['username', 'action', 'reward']).values

    scaler = 'min_max'

    episode_step = train_data.groupby('username').cumcount()
    terminal_step = (episode_step == 0).shift(-1)
    terminal_step.fillna(True, inplace = True)
    terminal_step *= 1

    terminals = terminal_step.values.reshape(-1,1)
    actions = train_data['action'].values.reshape(-1,1)
    rewards = train_data['reward'].values.reshape(-1,1)

    del train_data

    dataset = MDPDataset(observations=obs, terminals=terminals, 
                        actions=actions, rewards=rewards, discrete_action=True)

    dataset.dump('dataset_KT3_200k.h5')
    print('Dataset formed. Obs size = {}'.format(obs.shape))
else:
    dataset = MDPDataset.load(os.path.join(scratch_path, 'dataset_KT3_200k.h5'))
    scaler = 'min_max'

train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)


if algo == 'cql':
    model = DiscreteCQL(use_gpu=True, scaler=scaler, batch_size=batch_size, q_func_factory=q_function, learning_rate=lr,
        encoder_factory=VectorEncoderFactory(dropout_rate=dropout_rate, use_batch_norm=batch_norm), n_steps = n_steps, alpha = alpha)
elif algo == 'bcq':
    model = DiscreteBCQ(use_gpu=True, scaler=scaler, batch_size=batch_size, q_func_factory=q_function, learning_rate=lr,
        encoder_factory=VectorEncoderFactory(dropout_rate=dropout_rate, use_batch_norm=batch_norm), n_steps = n_steps)


# start training
results = model.fit(train_episodes,
        eval_episodes=test_episodes,
        n_epochs = n_epochs,
        scorers={
            'advantage': discounted_sum_of_advantage_scorer, # smaller is better
            'td_error': td_error_scorer, # smaller is better
            'initial_state_val': initial_state_value_estimation_scorer,
            'action_match': discrete_action_match_scorer
        }, 
        experiment_name = exp_name)

# save greedy-policy as TorchScript
model.save_policy('policy_{}.pt'.format(exp_name))