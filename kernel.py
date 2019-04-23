
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from librosa import mfcc
import scipy.signal
import scipy.stats

import os
train = pd.read_csv('train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

# Like in RNN starter kernel we choose first
sec_to_fail = 50085877
test = train.iloc[:sec_to_fail, :]
train = train.iloc[sec_to_fail:, :]

# We use the same sliding window method as https://www.kaggle.com/latimerb/earthquake-prediction-getting-started
fs = 4000000 # Hz
window_time = 0.0375 # seconds
offset = 0.01 # seconds
window_size = int(window_time*fs)

def extract_feature(dataset, length):
    feat_mat = np.zeros((length, 85))

    for i in tqdm_notebook(np.arange(0,feat_mat.shape[0])):
        start = int(i*offset*fs)
        stop = int(window_size+i*offset*fs)
        seg = dataset.iloc[start:stop,0]

        feat_mat[i,0] = np.mean(seg)
        feat_mat[i,1] = np.var(seg)
        feat_mat[i,2] = np.max(seg)
        feat_mat[i,3] = np.min(seg)
        # From here I added mfcc features
        mfcc_f = mfcc(y=np.array(seg, dtype=np.float32), n_mfcc=20)
        # For each of 10 mfcc features I extracted mean and variance
        for j in range(mfcc_f.shape[0]):
            feat_mat[i, 4 + 4*j] = np.mean(mfcc_f[j])
            feat_mat[i, 4 + 4*j + 1] = np.var(mfcc_f[j])
            feat_mat[i, 4 + 4*j + 2] = np.max(mfcc_f[j])
            feat_mat[i, 4 + 4*j + 3] = np.min(mfcc_f[j])
        feat_mat[i,-1] = dataset.iloc[stop+1,1]

    return feat_mat

test = train.iloc[:sec_to_fail, :]
train = train.iloc[sec_to_fail:, :]
feat_mat = extract_feature(train, 13500)
test_feat_mat = extract_feature(test, 1100)

df = pd.DataFrame(feat_mat,dtype=np.float64)
test_df = pd.DataFrame(test_feat_mat,dtype=np.float64)
dataset = df.values
test_set = test_df.values

xgr = xgb.XGBRegressor(booster='gbtree', n_estimators=100, max_depth=4, n_jobs=72, learning_rate=0.1)
xgr.fit(dataset[:,:-1], dataset[:,-1])

from tqdm import tqdm_notebook

submission = pd.read_csv('sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})
X_test = pd.DataFrame(columns=df.columns, dtype=np.float64, index=submission.index)

for i, seg_id in enumerate(tqdm_notebook(X_test.index)):
    seg = pd.read_csv('./test/' + seg_id + '.csv')
    X_test.loc[seg_id, '0'] = np.mean(seg.values)
    X_test.loc[seg_id, '1'] = np.var(seg.values)
    X_test.loc[seg_id, '2'] = np.max(seg.values)
    X_test.loc[seg_id, '3'] = np.min(seg.values)
    mfcc_f = mfcc(y=np.array(seg, dtype=np.float32).flatten(), n_mfcc=20)
    # For each of 10 mfcc features I extracted mean and variance
    for j in range(mfcc_f.shape[0]):
        X_test.loc[seg_id, 4 + 4*j] = np.mean(mfcc_f[j])
        X_test.loc[seg_id, 4 + 4*j + 1] = np.var(mfcc_f[j])
        X_test.loc[seg_id, 4 + 4*j + 2] = np.max(mfcc_f[j])
        X_test.loc[seg_id, 4 + 4*j + 3] = np.min(mfcc_f[j])

pred = xgr.predict(X_test.values)

submission['time_to_failure'] = pred
submission.to_csv('submission.csv')
