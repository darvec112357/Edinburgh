import pandas as pd
import numpy as np
import os
import time
from itertools import product
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model
import json
import argparse

print('Reading input arguments:')
parser = argparse.ArgumentParser(description='scratch disk path input')
parser.add_argument('--scratch_path')
parser.add_argument('--limit', type=float, default=None)
args = parser.parse_args()
scratch_path = args.scratch_path
limit = args.limit

questions = pd.read_csv(os.path.join(scratch_path,'EdNet-Contents', 'contents', 'questions.csv'))
lectures =  pd.read_csv(os.path.join(scratch_path,'EdNet-Contents', 'contents', 'lectures.csv'))
kt3_files = os.listdir(os.path.join(scratch_path, 'KT3_csv_large'))
kt3_paths = [os.path.join(scratch_path,'KT3_csv_large', file) for file in kt3_files]

# Loading ssl autoencoder
ssl_ae = load_model(os.path.join(scratch_path, 'encoder_ssl.h5'))
# Loading partfam autoencoder
partfam_ae = load_model(os.path.join(scratch_path, 'encoder_partfam.h5'))


num_qs_per_part = questions.groupby('part').count()['question_id'].values
num_lec_per_part = lectures.groupby('part').count()['lecture_id'].values[2:]
lectures.rename(columns = {'part':'part_l'}, inplace = True)

# 0 being lectures of each associated part
q_rank_space = [0,1,2,3,4]
part_space = questions['part'].unique()
action_space = list(product(part_space, q_rank_space))
action_encoder = {j:i for i,j in enumerate(action_space)}
# Adding lectures from part 0 & -1
action_encoder[(0,0)] = len(action_encoder)
action_encoder[(-1,0)] = len(action_encoder)

reward_mapping = {
    0:0, # lecture 
    1:1, # q_rank = 1 & correct
    2:2, # q_rank = 2 & correct
    3:3, # q_rank = 3 & correct
    4:4, # q_rank = 4 & correct
    -1:-8, # q_rank = 1 & incorrect
    -2:-6, # q_rank = 2 & incorrect
    -3:-4, # q_rank = 3 & incorrect
    -4:-2, # q_rank = 4 & incorrect
}

def run_batch(paths, action_encoder = action_encoder, reward_mapping = reward_mapping, limit = None):
    start = time.time()
    li = []
    for i, path in enumerate(paths):
        user_df = pd.read_csv(path)
        print('Size of df part {}: {}'.format(i,len(user_df)))
        li.append(user_df)
    mid = time.time()
    print('Loaded users in {}s'.format(np.round(mid-start,2)))

    # Concatenanting user dfs
    kt3 = pd.concat(li, axis = 0 , ignore_index = True)
    print('Raw size of KT3: {}'.format(len(kt3)))
    
    # Deriving time elapsed
    enter_time = kt3['action_type'] == "enter"
    leave_time = kt3['action_type'] == "submit"
    enter_time = enter_time * kt3['timestamp']
    enter_time[enter_time == 0] = np.nan
    enter_time.fillna(method = 'ffill', inplace = True)
    leave_time = leave_time * kt3['timestamp']
    time_diff = leave_time - enter_time
    time_diff[time_diff < 0] = np.nan
    time_diff.fillna(method = 'bfill', inplace = True)
    kt3['elapsed_time'] = time_diff
    # Only extract question responses, lectures & explanations
    kt3 = kt3[kt3['item_id'].str.match("[qle]\d*")]
        
    # Merging info on correct answers and parts from other dfs
    kt3 = kt3.merge(questions[['question_id', 'correct_answer', 'part']], right_on = 'question_id', 
                    left_on = 'item_id', how = 'left')
    kt3 = kt3.merge(questions[['explanation_id', 'part']], right_on = 'explanation_id', 
                    left_on = 'item_id', how = 'left', suffixes=['_q','_e'])
    kt3 = kt3.merge(lectures[['lecture_id', 'part_l']], left_on = 'item_id', right_on = 'lecture_id', 
                   how = 'left')
    # Removing meta lecture info
    kt3.query('action_type != "quit"', inplace = True)
    kt3['part_q'].fillna(0, inplace = True)
    kt3['part_e'].fillna(0, inplace = True)
    kt3['part_l'].fillna(0, inplace = True)
    kt3['part'] = kt3['part_q'] + kt3['part_e'] + kt3['part_l']
    kt3['part'] = kt3['part'].astype('int')
    # Dealing with meta info
    kt3.loc[kt3['action_type'] == 'respond','action_type'] = 'question'
    kt3.loc[kt3['item_id'].str.match("l\d*"),'action_type'] = 'lecture'
    kt3.loc[kt3['item_id'].str.match("e\d*"),'action_type'] = 'explanation'
    # NOTE I set time elapsed for lectures/explantions as 0
    kt3.loc[kt3['action_type'] != 'question', 'elapsed_time'] = 0
    # Cleaning up
    kt3.drop(columns = ['lecture_id', 'question_id','explanation_id', 'platform', 'part_l', 'part_q', 'part_e','source'], \
             inplace = True)
    # NOTE I am counting unique oscillations (meaning max is 4 if possible answer is abcd)
    answer_ossc = kt3.groupby(['username','item_id']).nunique()['user_answer']
    # NOTE there are multiple attempts at responding/watching content. Currently I take the final record of this
    kt3 = kt3.groupby(['username','item_id']).tail(1).copy()
    kt3.set_index(['username', 'item_id'], inplace = True)
    kt3['answer_ossc'] = answer_ossc
    kt3.reset_index(inplace = True)
    kt3.sort_values(['username','timestamp'], inplace = True)
    kt3.reset_index(inplace = True, drop = True)
    kt3['part'] = kt3['part'].astype('int8')
    kt3_meta = kt3.copy()
    kt3.query('action_type != "explanation"', inplace = True)

    print('Total number of transitions = {}'.format(len(kt3)))

    ############################ Question Correctness ############################
    kt3['mark'] = (kt3['user_answer'] == kt3['correct_answer'])*1
    question_correct = kt3.query('action_type == "question"').groupby(['item_id', 'part'], ).sum()['mark']
    question_count = kt3.query('action_type == "question"').groupby(['item_id','part'], ).count()['mark']
    question_diff = 100 - (question_correct/question_count * 100)

    # Assigning questions to difficulty quantiles
    question_diff = question_diff.reset_index(level = 1)
    q_level_25 = question_diff.groupby('part',).quantile(0.25)
    q_level_50 = question_diff.groupby('part', ).quantile(0.50)
    q_level_75 = question_diff.groupby('part',).quantile(0.75)
    question_diff['q_rank'] = -1
    for part in questions['part'].unique():
        q_part = question_diff.query('part == @part')
        first_q = q_level_25.loc[part,'mark']
        second_q = q_level_50.loc[part,'mark']
        third_q = q_level_75.loc[part,'mark']
        question_rank = np.digitize(q_part['mark'], bins = [0,first_q,second_q,third_q])
        question_diff.loc[q_part.index, 'q_rank'] = question_rank
    kt3_mod = kt3.merge(question_diff['q_rank'], how = 'left', left_on = 'item_id', right_index = True)
    kt3_mod['q_rank'] = kt3_mod['q_rank'].fillna(0)
    kt3_mod['q_rank'] = kt3_mod['q_rank'].astype('int')
    kt3_mod['mark'] = kt3['mark']
    del kt3 # to save memory

    ################## State Features ###########################
    state_features = pd.DataFrame()
    state_features['username'] = kt3_mod['username']
    initial_state = ''

    #### TOPIC FAM ###################################################
    num_content_per_part = num_qs_per_part + num_lec_per_part
    topic_fam = kt3_mod.groupby(['username','part'], ).cumcount()
    topic_len = kt3_mod['part'].apply(lambda x: num_content_per_part[x-1])
    topic_fam_norm = topic_fam/topic_len 
    first_q = topic_fam_norm.quantile(0.25)
    second_q = topic_fam_norm.quantile(0.5)
    third_q = topic_fam_norm.quantile(0.75)
    topic_fam_quantized = np.digitize(topic_fam_norm, [0,first_q, second_q, third_q])
    # NOTE I am not incrementing familiarity in these parts since there are no questions associated
    topic_fam_quantized[(kt3_mod['part'] == -1) | (kt3_mod['part'] == 0)] = 1
    state_features['topic_fam'] = topic_fam_quantized
    initial_state += '1'

    del topic_fam_quantized
    del topic_fam_norm

    ##### PART FAM ####################################
    enc = OneHotEncoder()
    part_fam = enc.fit_transform(kt3_mod['part'].values.reshape(-1,1)).toarray()
    part_fam = (topic_fam + 1).values.reshape(-1,1) * part_fam
    # Not including part -1 & 0 as these have no associated questions (only lectures)
    part_fam = part_fam[:,2:]
    # Forward filling
    part_fam[part_fam == 0] = np.nan
    episode_step = kt3_mod.groupby('username').cumcount()
    begin_mask = np.isnan(part_fam) * (episode_step == 0).values.reshape(-1,1)
    part_fam[begin_mask] = 0
    part_fam = pd.DataFrame(part_fam, columns = range(1,8))
    part_fam.fillna(method = 'ffill', inplace = True)

    # Quantize it
    for part in part_fam.columns:
        dist = part_fam[part]
        first_q = dist.quantile(0.25)
        second_q = dist.quantile(0.5)
        third_q = dist.quantile(0.75)
        # Not sure what to do if first q is 0
        if first_q == 0:
            first_q += 1
            second_q += 1
            third_q += 1
        part_fam[part] = np.digitize(part_fam[part], [0,first_q, second_q, third_q])
    init_conf = [[1,1,1,1,1,1,1]]

    start = time.time()
    parts_fam_enc = partfam_ae.predict(part_fam)
    end = time.time()
    print('\nTime to autoencode part_fam: {}s'.format(np.round(end-start,2)))
    # Quantize it
    n_bins = 8
    bins = [np.quantile(parts_fam_enc, q) for q in np.arange(0,1,1/n_bins)]
    part_fam_quantized = np.digitize(parts_fam_enc, bins)
    state_features['part_fam'] = part_fam_quantized
    start_val = np.digitize(partfam_ae.predict(init_conf), bins).item()
    initial_state += str(start_val)

    del part_fam_quantized
    del parts_fam_enc

    ############### AVG FAM ##################################################
    part_fam_norm = part_fam/num_content_per_part
    av_part_fam = part_fam_norm.mean(axis = 1)
    first_q = av_part_fam.quantile(0.25)
    second_q = av_part_fam.quantile(0.5)
    third_q = av_part_fam.quantile(0.75)
    print('Quantiles are: Q1:{} Q2:{} Q3:{}'.format(np.round(first_q,3), np.round(second_q,3), np.round(third_q,3)))
    av_fam_quantized = np.digitize(av_part_fam, [0,first_q, second_q, third_q])
    state_features['av_fam'] = av_fam_quantized
    initial_state += '1'

    del av_fam_quantized
    del av_part_fam

    ############### PREV CORRECT #################################################
    prev_correct = kt3_mod['mark'].copy()
    prev_correct[kt3_mod['action_type'] == 'lecture'] = 2
    state_features['prev_correct'] = prev_correct.values
    initial_state += '2'

    del prev_correct

    ############### CORRECT SO FAR ############################################
    kt3_mod['correct_attempts'] = kt3_mod['correct_answer'] == kt3_mod['user_answer']
    kt3_mod.loc[kt3_mod['action_type'] == 'lecture', 'correct_attempts'] = False
    attempts = kt3_mod.groupby(['username', 'correct_attempts']).cumcount()
    attempts += 1
    kt3_mod['correct_so_far'] = np.nan
    question_step = kt3_mod.groupby(['username', 'action_type']).cumcount()
    episode_step = kt3_mod.groupby('username').cumcount()
    # question step keeps track of cumulative questions (only) presented 
    question_step[(kt3_mod['action_type'] != 'question') & (episode_step != 0)] = np.nan
    question_step.fillna(method = 'ffill', inplace = True)
    starting_index = episode_step == 0
    kt3_mod.loc[starting_index, 'correct_so_far'] = 0
    # Assign cumulative correct attempts at correct steps
    kt3_mod.loc[kt3_mod['correct_attempts'].values, 'correct_so_far'] = attempts[kt3_mod['correct_attempts']]
    kt3_mod.fillna(method = 'ffill', inplace = True)
    kt3_mod['correct_so_far'] = kt3_mod['correct_so_far'].astype('int')
    correct_so_far = kt3_mod['correct_so_far'] /(question_step + 1)
    kt3_mod.drop(columns = ['correct_so_far', 'correct_attempts'], inplace = True)
    first_q = correct_so_far.quantile(0.25)
    second_q = correct_so_far.quantile(0.5)
    third_q = correct_so_far.quantile(0.75)
    state_features['correct_so_far'] = np.digitize(correct_so_far, [0, first_q, second_q, third_q])
    initial_state += '1'

    del correct_so_far

    ################ AVG TIME ####################################
    av_elapsed_time = kt3_mod.groupby('username').cumsum()['elapsed_time']/(question_step + 1)
    first_q = av_elapsed_time.quantile(0.25)
    second_q = av_elapsed_time.quantile(0.5)
    third_q = av_elapsed_time.quantile(0.75)
    av_time_quantized = np.digitize(av_elapsed_time, [0,first_q, second_q, third_q])
    state_features['av_time'] = av_time_quantized
    initial_state += '1'

    del av_time_quantized
    del av_elapsed_time

    ################## SLOW ANSWER ###################################
    question_time_av = kt3_mod.groupby('item_id', ).mean()['elapsed_time']
    question_time_av.rename('question_time_av', inplace = True)
    if 'question_time_av' not in kt3_mod.columns:
        kt3_mod = kt3_mod.merge(question_time_av, how = 'left', left_on = 'item_id', right_index = True)
    state_features['slow_answer'] = ((kt3_mod['elapsed_time'] > kt3_mod['question_time_av'])*1).values
    initial_state += '0'

    ################ LECTS CONSUMED ##################################
    lectures_consumed = kt3_mod.groupby(['username', 'action_type'], ).cumcount() + 1
    lectures_consumed[(kt3_mod['action_type'] == 'question')] = np.nan
    lectures_consumed[(episode_step == 0) & lectures_consumed.isna()] = 0
    lectures_consumed.fillna(method = 'ffill', inplace = True)
    lects_per_student = kt3_mod.groupby(['username', 'action_type'], ).count().reset_index('action_type')
    lects_per_student = lects_per_student.query('action_type == "lecture"')['item_id'].value_counts()
    first_q = lects_per_student.quantile(0.25)
    second_q = lects_per_student.quantile(0.5)
    third_q = lects_per_student.quantile(0.75)
    lects_quantized = np.digitize(lectures_consumed, [0, first_q, second_q, third_q])
    state_features['lects_consumed'] = lects_quantized
    initial_state += '1'

    del lectures_consumed
    del lects_quantized

    ############### TIME IN PART #################################
    con_steps = (kt3_mod['part'] != kt3_mod['part'].shift(1)).cumsum()
    temp = kt3_mod.copy()
    temp['con_steps'] = con_steps
    con_steps_dist = temp.groupby(['username', 'part', 'con_steps'], ).count()['item_id']
    con_steps_counter = temp.groupby(['username', 'con_steps'], ).cumcount() + 1
    first_q = con_steps_dist.quantile(0.3) # since 0.25 is 1
    second_q = con_steps_dist.quantile(0.5)
    third_q = con_steps_dist.quantile(0.75)
    time_in_part = np.digitize(con_steps_counter, [0, first_q, second_q, third_q])
    state_features['time_in_part'] = time_in_part
    initial_state += '1'

    del temp
    del time_in_part
    del con_steps_dist
    del con_steps_counter
    del con_steps

    ################## EXPLANATIONS RECEIVED ###############################
    episode_step = kt3_meta.groupby('username').cumcount()
    expl_count = kt3_meta.groupby(['username', 'action_type'], ).cumcount()
    expl_count[(kt3_meta['action_type'] != 'explanation') & (episode_step != 0)] = np.nan
    expl_count += 1
    expl_count.fillna(method = 'ffill', inplace = True)
    expl_count_dist = kt3_meta.groupby(['username', 'action_type'], ).count().reset_index() \
        .query('action_type == "explanation"')['item_id']
    first_q = expl_count_dist.quantile(0.25)
    second_q = expl_count_dist.quantile(0.5)
    third_q = expl_count_dist.quantile(0.75)
    expl_quantized = np.digitize(expl_count, [0, first_q, second_q, third_q])
    state_features['expl_received'] = expl_quantized[kt3_meta.reset_index().query('action_type != "explanation"').index]
    initial_state += '1'

    del expl_count
    del expl_quantized
    del expl_count_dist

    ################ SECOND GUESSING ########################################
    kt3_mod['cumul_sc'] = kt3_mod['answer_ossc'] - 1
    kt3_mod.loc[kt3_mod['cumul_sc'] < 0, 'cumul_sc'] = 0
    cumul_sc = kt3_mod.groupby('username').cumsum()['cumul_sc']
    normalize = True
    if normalize:
        cumul_sc = cumul_sc/(question_step + 1)
    first_q = cumul_sc.quantile(0.25)
    second_q = cumul_sc.quantile(0.5)
    third_q = cumul_sc.quantile(0.75)
    cumulsc_quantized = np.digitize(cumul_sc, [0, first_q, second_q, third_q])
    state_features['sec_guess'] = cumulsc_quantized
    initial_state += '1'

    del cumul_sc
    del cumulsc_quantized

    ################ STEPS SINCE LAST #######################################
    temp = enc.fit_transform(kt3_mod['part'].values.reshape(-1,1)).toarray()
    episode_step = kt3_mod.groupby('username').cumcount()
    temp = temp * (episode_step + 1).values.reshape(-1,1)
    # Not including part -1 & 0 as these have no associated questions (only lectures)
    temp = temp[:,2:]
    # Forward filling
    temp[temp == 0] = np.nan
    begin_mask = np.isnan(temp) * (episode_step == 0).values.reshape(-1,1)
    temp[begin_mask] = -200
    temp = pd.DataFrame(temp, columns = range(1,8))
    temp.fillna(method = 'ffill', inplace = True)
    steps_since_last = (episode_step + 1).values.reshape(-1,1) - temp.values
    steps_since_last[steps_since_last>200] = 200
    # Quantize
    first_q = np.quantile(steps_since_last, 0.25)
    second_q = np.quantile(steps_since_last, 0.5)
    third_q = np.quantile(steps_since_last, 0.75)
    ssl_quantized = np.digitize(steps_since_last, [0, first_q, second_q, third_q])
    # Encoding ssl
    start = time.time()
    ssl_enc = ssl_ae.predict(ssl_quantized)
    end = time.time()
    print('\nTime to autoencode ssl: {}s'.format(np.round(end-start,2)))
    # Quantize it
    n_bins = 8
    bins = [np.quantile(ssl_enc, q) for q in np.arange(0,1,1/n_bins)]
    ssl = np.digitize(ssl_enc, bins)
    # [4,4,4,4,4,4,4] is initial since all parts are not recently covered
    start_val = np.digitize(ssl_ae.predict([[4,4,4,4,4,4,4]]), bins).item()
    state_features['ssl'] = ssl
    initial_state += str(start_val)

    del ssl
    del ssl_quantized
    del ssl_enc
    
    # Creating Initial state value dict
    features = state_features.columns.to_list()
    features.remove('username')
    init_state_dic = {state:init_value for state,init_value in zip(features,initial_state)}

    ################## TRAJECTORIES ####################################
    trajectories =  state_features.copy()
    trajectories['next_state'] = state_features.drop(columns = ['username']).astype('str').sum(axis = 1).astype('int64')
    trajectories['next_state'] = trajectories['next_state'].astype(str)
    trajectories['state'] = trajectories['next_state'].shift(1)
    trajectories.loc[starting_index,'state'] = initial_state
    trajectories =  trajectories[['username','state', 'next_state']]
    del state_features # Delete to save memory

    ############# ACTION ENCODING ################################
    start = time.time()
    trajectories['action'] = list(zip(kt3_mod['part'], kt3_mod['q_rank']))
    trajectories['action'] = trajectories['action'].map(action_encoder)
    end = time.time()
    print('Time taken to encode actions: {}s'.format(np.round(end-start,2)))


    ############## REWARD MAPPING ####################################
    kt3_mod.loc[kt3_mod['mark'] == 0, 'mark'] = -1
    trajectories['reward'] = kt3_mod['mark'] * kt3_mod['q_rank']
    trajectories['reward'] = trajectories['reward'].apply(lambda x: reward_mapping[x])
    

    ######## TRANSITION COUNTS #######################################
    transitions = trajectories.groupby(['state', 'action','next_state', 'reward']).count()['username']
    transitions.rename('transitions', inplace = True)
    print('There are {} rows in trajectories'.format(len(trajectories)))
    print('There are {} NA usernames in trajectories'.format(trajectories['username'].isna().sum()))
    print('There are {} transitions in this batch'.format(transitions.sum()))

    if limit is not None:
        limit = int(limit)
        print('trajectories is limited to : {} users'.format(limit))
        limited_users = trajectories.username.unique()[:limit]
        trajectories.query('username in @limited_users', inplace = True)
    
    return transitions, init_state_dic, trajectories

transitions,init_state_dict,trajectories = run_batch(kt3_paths, limit=limit)
print('Finished processing')
transitions.to_csv('KT3_complete_transitions.csv')
if limit is None:
    trajectories.to_csv('trajectories_KT3_complete.csv', index=False)
else:
    trajectories.to_csv('trajectories_KT3_{}k.csv'.format(int(limit/1000)), index=False)

with open('init_state.txt', 'w') as file:
    file.write(json.dumps(init_state_dict))


# Setting up dataset for training CQL
features = list(init_state_dict.keys())
# Optimal feature set
feat_to_incl = ["topic_fam", "correct_so_far","av_time","expl_received", "ssl", "prev_correct","av_fam"]
trajectories['state'] = trajectories['state'].astype('str')
feat_indices = [features.index(feat) for feat in feat_to_incl]

obs = trajectories['state'].str.split('', expand = True).values[:,1:-1].astype(int)
obs = obs[:, feat_indices]
train_data = pd.DataFrame(obs, columns = ['obs_{}'.format(i) for i in range(len(feat_to_incl))])
train_data['reward'] = trajectories['reward'].values
train_data['username'] = trajectories['username'].values
train_data['action'] = trajectories['action'].values
del trajectories

train_data.to_csv('CQL_train_data.csv', index=False)



print('Finished exporting')