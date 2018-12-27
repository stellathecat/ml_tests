# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 18:49:54 2018

@author: boros
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from sklearn.utils import shuffle
import sklearn
import os
import multiprocessing as mp

def X_y(q, process_type, preprocessed_data, time_, volume, model_type, df_h1_High_minus_Low, df_h1_High_minus_Close, df_h1_Close_minus_Low):
    # subfunction of create_x_y_parallel in order to create features for ML algorithm parallel (with multiple CPUs)
    result = []
    
    len_pr_data = len(preprocessed_data)
    len_X = 25
    
    for j in range(0, len_pr_data):
        len_pr_data_j = len(preprocessed_data[j])
        for i in range(len_X + 1, len_pr_data_j):
            if process_type == 'X':
                result.append(preprocessed_data[j][i-len_X:i])
            if process_type == 'y':   
                result.append(preprocessed_data[j][i])
            if process_type == 'hours':
                result.append(time_[j].iloc[i].hour)
            if process_type == 'days':
                result.append(time_[j].iloc[i].day)
            if process_type == 'months':
                result.append(time_[j].iloc[i].month)
            if process_type == 'volumes':
                result.append(volume[j].iloc[i-1])
            if process_type == 'mean_feature':
                result.append(np.mean(preprocessed_data[j][i-len_X:i]))
            if process_type == 'max_feature':
                result.append(np.max(preprocessed_data[j][i-len_X:i]))
            if process_type == 'min_feature':
                result.append(np.min(preprocessed_data[j][i-len_X:i]))
            if process_type == 'last_second_dif':
                result.append(np.diff(preprocessed_data[j][i-len_X:i].tail(2)))
            if process_type == 'high_minus_low':   
                result.append(df_h1_High_minus_Low[j][i-1])
            if process_type == 'high_minus_close':   
                result.append(df_h1_High_minus_Close[j][i-1])
            if process_type == 'close_minus_low':   
                result.append(df_h1_Close_minus_Low[j][i-1])
    
    q.put(result)

def create_x_y_parallel(preprocessed_data, time_, volume, model_type, df_h1_High_minus_Low, df_h1_High_minus_Close, df_h1_Close_minus_Low):
    # Creating X and y for ML algorithm with parallel processing (multiple CPUs), it is important
    # because preprocessing was pretty long time
    # We are creating more features for ML into X, e.g. hour, day, volume, etc.
    
    preprocessed_data = get_new_preprocessed_data(model_type, preprocessed_data)
    ctx = mp.get_context('spawn')

    queues = []
    p = []
    for i in range(0, 10):
        queues.append(mp.Queue()) # part of multiprocessing
    
    p.append(mp.Process(target=X_y, args=(queues[0], 'X', preprocessed_data, 
                    time_, volume, 'dir', df_h1_High_minus_Low, df_h1_High_minus_Close, df_h1_Close_minus_Low)))
    p[-1].start() # we start the process (on one CPU)
    p.append(mp.Process(target=X_y, args=(queues[1], 'y', preprocessed_data, 
                    time_, volume, 'dir', df_h1_High_minus_Low, df_h1_High_minus_Close, df_h1_Close_minus_Low)))
    p[-1].start() # we start another process (on other CPU)
    p.append(mp.Process(target=X_y, args=(queues[2], 'hours', preprocessed_data, 
                    time_, volume, 'dir', df_h1_High_minus_Low, df_h1_High_minus_Close, df_h1_Close_minus_Low)))
    p[-1].start()
    p.append(mp.Process(target=X_y, args=(queues[3], 'days', preprocessed_data, 
                    time_, volume, 'dir', df_h1_High_minus_Low, df_h1_High_minus_Close, df_h1_Close_minus_Low)))
    p[-1].start()
    p.append(mp.Process(target=X_y, args=(queues[4], 'months', preprocessed_data, 
                    time_, volume, 'dir', df_h1_High_minus_Low, df_h1_High_minus_Close, df_h1_Close_minus_Low)))
    p[-1].start()
    p.append(mp.Process(target=X_y, args=(queues[5], 'volumes', preprocessed_data, 
                    time_, volume, 'dir', df_h1_High_minus_Low, df_h1_High_minus_Close, df_h1_Close_minus_Low)))
    p[-1].start()
    p.append(mp.Process(target=X_y, args=(queues[6], 'mean_feature', preprocessed_data, 
                    time_, volume, 'dir', df_h1_High_minus_Low, df_h1_High_minus_Close, df_h1_Close_minus_Low)))
    p[-1].start()
    p.append(mp.Process(target=X_y, args=(queues[7], 'max_feature', preprocessed_data, 
                    time_, volume, 'dir', df_h1_High_minus_Low, df_h1_High_minus_Close, df_h1_Close_minus_Low)))
    p[-1].start()
    p.append(mp.Process(target=X_y, args=(queues[8], 'min_feature', preprocessed_data, 
                    time_, volume, 'dir', df_h1_High_minus_Low, df_h1_High_minus_Close, df_h1_Close_minus_Low)))
    p[-1].start()
    p.append(mp.Process(target=X_y, args=(queues[9], 'last_second_dif', preprocessed_data, 
                    time_, volume, 'dir', df_h1_High_minus_Low, df_h1_High_minus_Close, df_h1_Close_minus_Low)))
    p[-1].start()
    
    # We get the results of the ready processes
    X = queues[0].get() 
    y = queues[1].get()
    hours = queues[2].get()
    days = queues[3].get()
    months = queues[4].get()
    volumes = queues[5].get()
    mean_feature = queues[6].get()
    max_feature = queues[7].get()
    min_feature = queues[8].get()
    last_second_dif = queues[9].get()
    
    for i in range(0, len(p)):
        p[i].join()
    
    # I have 10 CPUs on my server therefore I could process only 10 processes parallel
    # after done of 10 processes we continue with the following processes:
    
    queues = []
    p = []
    for i in range(0, 10):
        queues.append(mp.Queue())
        
    p.append(mp.Process(target=X_y, args=(queues[0], 'high_minus_low', preprocessed_data, 
                    time_, volume, 'dir', df_h1_High_minus_Low, df_h1_High_minus_Close, df_h1_Close_minus_Low)))
    p[-1].start()
    p.append(mp.Process(target=X_y, args=(queues[1], 'high_minus_close', preprocessed_data, 
                    time_, volume, 'dir', df_h1_High_minus_Low, df_h1_High_minus_Close, df_h1_Close_minus_Low)))
    p[-1].start()
    p.append(mp.Process(target=X_y, args=(queues[2], 'close_minus_low', preprocessed_data, 
                    time_, volume, 'dir', df_h1_High_minus_Low, df_h1_High_minus_Close, df_h1_Close_minus_Low)))
    p[-1].start()
        
    high_minus_low = queues[0].get()
    high_minus_close = queues[1].get()
    close_minus_low = queues[2].get()
    
    for i in range(0, len(p)):
        p[i].join()
        
    # we are creating X from features
    X = pd.DataFrame(np.column_stack([high_minus_low, high_minus_close, close_minus_low, mean_feature, max_feature, min_feature, last_second_dif, volumes, days, hours, months, X]))
    y = pd.Series(y)

    # and we are creating 0 or 1 labels depending on the current value of y
    if model_type == 'dir':
        y[y >= 0] = 1
        y[y < 0] = 0
        
    print('X, y finish')

    return X, y

start = time.time() # for measuring the time
date = '2016.12.31 24:00:W00' # it is the threshold before that: training set, after that: test set
all_c_test = import_curr_test(date) 
all_c_train = import_curr_train(date)

# creating new lists for data
df_h1_dif = [] 
abs_df_h1_dif = []
scaled_df_h1_dif = []
time_ = []
volume = []

until = -1

all_c_test_current = [all_c_test[0][0:until]]
# explanation
# all_c_test[0] into a list as the first element of list
#all_c_test_current = [all_c_test[0]] # all_c_test[0] it means: first currency from all currencies, it is important because this system can handle more currencies in order to learn algorithm general information about movement of prices in FOREX

df_h1_dif, abs_df_h1_dif, scaled_df_h1_dif, time_, volume, df_h1_High_minus_Low, df_h1_High_minus_Close, df_h1_Close_minus_Low = data_preprocessing(all_c_test_current)
# explanation
# outputs of function: df_h1_dif, abs_df_h1_dif, etc
# input of function: all_c_test_current
# data_preprocessing: name of function

import time
start = time.time()
X_test_final, y_test_final = create_x_y_parallel(df_h1_dif, 
                time_, volume, 'final', df_h1_High_minus_Low, df_h1_High_minus_Close, df_h1_Close_minus_Low)    
import time
end = time.time()
print(end - start)
# explanation
# it is a parallel function so it is using parallel 10 cores (later this number can be a parameter)
# we are creating here feature data set and target data set for testing)
# final y-s contain pip movement of price as y_test_dir contain only the target direction (long or short)

