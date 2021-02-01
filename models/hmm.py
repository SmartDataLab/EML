#%%
from __future__ import division

import json
import numpy as np
import pandas as pd
import os
import six
import sys

from hmmlearn.hmm import MultinomialHMM

sys.modules["sklearn.externals.six"] = six

# 0:不玩手机， 1:开始玩手机, 2:持续玩手机, 3:结束玩手机
def get_ubie_label(raw_label_series):
    states = {}
    label = np.array(raw_label_series)
    flag = 0.0
    state_list = [0] * len(label)
    for i in range(len(label)):
        state = np.zeros((4,))
        if label[i] == 1.0:
            if flag == 0.0:
                states[i] = np.array([0, 1, 0, 0])
                state_list[i] = 1
            else:
                states[i] = np.array([0, 0, 1, 0])
                state_list[i] = 2
        else:
            if flag == 1.0 and i > 0:
                states[i - 1] = np.array([0, 0, 0, 1])
                state_list[i - 1] = 3
            states[i] = np.array([1, 0, 0, 0])
            state_list[i] = 0
        flag = label[i]
    return state, state_list


def get_probability_box(pred_series):
    df_train = pd.DataFrame()
    df_train["low_2"] = pred_series < 0.1
    df_train["low_1"] = (pred_series >= 0.1) * (pred_series < 0.3)
    df_train["low_0"] = (pred_series >= 0.3) * (pred_series < 0.5)
    df_train["high_0"] = (pred_series >= 0.5) * (pred_series < 0.7)
    df_train["high_1"] = (pred_series >= 0.7) * (pred_series < 0.9)
    df_train["high_2"] = pred_series >= 0.9
    return df_train.astype(np.int8)


def map_pred(x):
    if x < 0.1:
        return 0
    elif x < 0.3:
        return 1
    elif x < 0.5:
        return 2
    elif x < 0.7:
        return 3
    elif x < 0.9:
        return 4
    else:
        return 5


def get_emission(pred_list, state_list):
    count_array = np.zeros((4, 6))
    for idx in range(len(pred_list)):
        count_array[state_list[idx], pred_list[idx]] += 1
    return (count_array.T / count_array.sum(axis=1)).T


def get_transmat(state_list):
    count_array = np.zeros((4, 4))
    for idx in range(len(state_list) - 1):
        count_array[state_list[idx], state_list[idx + 1]] += 1
    return (count_array.T / count_array.sum(axis=1)).T


def get_pred_for_hmm(pred_series):
    return [map_pred(x) for x in pred_series]


def get_hmm(df, n_components, n_features):
    _, state_list = get_ubie_label(df["label"])
    pred_list = get_pred_for_hmm(df["pred"])
    clf = MultinomialHMM(n_components=n_components)
    clf.n_features = n_features
    clf.transmat_ = get_transmat(state_list)
    clf.emissionprob_ = get_emission(pred_list, state_list)
    clf.startprob_ = np.array([0.5, 0.05, 0.4, 0.05])
    return clf


#%%

if __name__ == "__main__":

    folder_path = "/Users/su/data/phone_play/13522117899/"
    file_list = os.listdir(folder_path)
    prefix_list = [x.split(".")[0] for x in file_list if x.split(".")[-1] == "pb"]
    df = pd.read_csv("../data/%s-pred.csv" % prefix_list[0])
    clf = get_hmm(df, n_components=4, n_features=6)
    pred_list = get_pred_for_hmm(df["pred"])
    hmm_pred = clf.predict(pd.DataFrame({"pred_box": pred_list}))

    # def test_two():
    label_01 = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
    get_ubie_label(label_01)
# %%
