import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import argparse

parser = argparse.ArgumentParser(description='scrath disk path input')
parser.add_argument('--scratch_path')
args = parser.parse_args()
scratch_path = args.scratch_path

questions = pd.read_csv(os.path.join(scratch_path,'EdNet-Contents', 'contents', 'questions.csv'))
lectures =  pd.read_csv(os.path.join(scratch_path,'EdNet-Contents', 'contents', 'lectures.csv'))

kt3_files = os.listdir(os.path.join(scratch_path,'KT3_csv_large'))
kt3_paths = [os.path.join(scratch_path,'KT3_csv_large', file) for file in kt3_files]

num_qs_per_part = questions.groupby('part').count()['question_id'].values
num_lec_per_part = lectures.groupby('part').count()['lecture_id'].values[2:]
lectures.rename(columns = {'part':'part_l'}, inplace = True)

# Limit for debug
nrows = None
# Read the first 100k subset
kt3 = pd.read_csv(kt3_paths[0], nrows=nrows)

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
kt3.query('action_type != "quit"', inplace=True)
kt3['part_q'].fillna(0, inplace = True)
kt3['part_e'].fillna(0, inplace = True)
kt3['part_l'].fillna(0, inplace = True)
kt3['part'] = kt3['part_q'] + kt3['part_e'] + kt3['part_l']
kt3['part'] = kt3['part'].astype('int')
# Dealing with meta info
kt3.loc[kt3['action_type'] == 'respond','action_type'] = 'question'
kt3.loc[kt3['item_id'].str.match("l\d*"),'action_type'] = 'lecture'
kt3.loc[kt3['item_id'].str.match("e\d*"),'action_type'] = 'explanation'

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


############### Prepare Part_fam Data ####################################################################
num_content_per_part = num_qs_per_part + num_lec_per_part
enc = OneHotEncoder()
topic_fam = kt3.groupby(['username','part'], observed = True).cumcount()
part_fam = enc.fit_transform(kt3['part'].values.reshape(-1,1)).toarray()
part_fam = (topic_fam + 1).values.reshape(-1,1) * part_fam

# Not including part -1 & 0
part_fam = part_fam[:,2:]

# Forward filling
part_fam[part_fam == 0] = np.nan
episode_step = kt3.groupby('username', observed=True).cumcount()
begin_mask = np.isnan(part_fam) * (episode_step == 0).values.reshape(-1,1)
part_fam[begin_mask] = 0
part_fam = pd.DataFrame(part_fam, columns = range(1,8))
part_fam.fillna(method = 'ffill', inplace = True)
part_fam = part_fam.astype(int)

# Quantize it
for part in part_fam.columns:
    dist = part_fam[part]
    #dist = dist[dist > 0]
    first_q = dist.quantile(0.25)
    second_q = dist.quantile(0.5)
    third_q = dist.quantile(0.75)
    print('PF Quantiles are: Q1:{} Q2:{} Q3:{}'.format(np.round(first_q,3), np.round(second_q,3), np.round(third_q,3)))
    # If first q is 0
    if first_q == 0:
        first_q += 1
        second_q += 1
        third_q += 1
    part_fam[part] = np.digitize(part_fam[part], [0,first_q, second_q, third_q])


############### Prepare ssl Data ####################################################################
enc = OneHotEncoder()
temp = enc.fit_transform(kt3['part'].values.reshape(-1,1)).toarray()
episode_step = kt3.groupby('username').cumcount()
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

# quantize
first_q = np.quantile(steps_since_last, 0.25)
second_q = np.quantile(steps_since_last, 0.5)
third_q = np.quantile(steps_since_last, 0.75)

steps_since_last = np.digitize(steps_since_last, [0, first_q, second_q, third_q])

print('SSL quantiles are: Q1:{} Q2:{} Q3:{}'.format(np.round(first_q,3), np.round(second_q,3), np.round(third_q,3)))

################ Prepare train test data ###############################################
val = True
def split_scale(data, validate = val, scale = False):
    if validate:
        X_train, X_test = train_test_split(data, shuffle = False)
        if scale:
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        return X_train, X_test
    else:
        X_train = data
        if scale:
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
        return X_train

def get_encoder(data, val, feat_name):
    ############ Build the Autoencoder Model ###################################################
    n_inputs = 7
    # define encoder
    visible = Input(shape=(n_inputs,))
    # encoder level 1
    e = Dense(n_inputs*2, activation = 'relu')(visible)
    e = BatchNormalization()(e)

    # bottleneck
    n_bottleneck = 1
    bottleneck = Dense(n_bottleneck, activation = 'relu')(e)

    # define decoder, level 1
    d = Dense(n_inputs*2, activation = 'relu')(bottleneck)
    d = BatchNormalization()(d)

    # output layer
    output = Dense(n_inputs, activation='linear')(d)

    # define autoencoder model
    autoenc = Model(inputs=visible, outputs=output)
    # compile autoencoder model
    autoenc.compile(optimizer='adam', loss='mse')

    print(autoenc.summary())
    plot_model(autoenc, 'encoder_compress.png', show_shapes=True)

    ############################### Split data #######################################
    X_train, X_test = split_scale(data, validate = val, scale = True)

    ############################## Train Model ##########################################
    if val:
        history = autoenc.fit(X_train, X_train, epochs=20, batch_size=32, verbose=2, validation_data=(X_test,X_test))
    else:
        history = autoenc.fit(X_train, X_train, epochs=20, batch_size=32, verbose=2)

    # plot loss
    plt.figure(figsize=(7,7))
    plt.plot(history.history['loss'], label='train')
    if val:
        plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig('autoencoder_loss_{}.pdf'.format(feat_name))
    plt.show()

    # define an encoder model (without the decoder)
    encoder = Model(inputs=visible, outputs=bottleneck)

    # save the encoder to file
    encoder.save('encoder_{}.h5'.format(feat_name))

val = True
get_encoder(part_fam, val, 'partfam')
get_encoder(steps_since_last, val, 'ssl')