import numpy as np
import argparse
import tqdm
import pickle
import pandas as pd
import os
import datetime
import warnings
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from utility.utility import *
from utility.training import callback
import tensorflow as tf

import configparser

# turn off system warning 
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def main(config, args):
    start_time = datetime.datetime.now()

    datetime_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # basic setting
    make_dir(config)
    maybe_make_dir(f'logs/{args.mode}/{args.model_type}')

    # logging    
    print(f'Training Start')

    # reading data
    print(f'Load Data Begain')
    df = pickle.load(open('data.pk', 'rb'))
    # df = read_data(config, args.mode)
    print(f'Load Data Finish')

    X_train=[]
    y_train=[]
    for i in range(0,len(df.trainSet)):
        X_train.append(df.trainSet[i]['target_history'])
        y_train.append(df.trainSet[i]['target_price'])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_valid=[]
    y_valid=[]
    for i in range(0,len(df.validSet)):
        X_valid.append(df.validSet[i]['target_history'])
        y_valid.append(df.validSet[i]['target_price'])
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)

    model = load_model(X_valid.shape, args.model_type)
    # print(args.mode) 

    if args.mode == 'train':
        history = model.fit(
            X_train, y_train, 
            epochs=int(config['MODEL']['epoch']),
            verbose = 1,
            batch_size=4,
            validation_data=(X_valid, y_valid), 
            callbacks = callback(config, args, datetime_prefix)
        )
        y_pred = model.predict(X_valid)

        model.save_weights(f'mi_model/{args.model_type}')

        valid_mse = mean_squared_error(y_valid, y_pred, squared=False)   
        print('mse',valid_mse)
        pd.DataFrame(history.history).to_csv(f'logs/csv_logger/{args.model_type}/{datetime_prefix}_{valid_mse}.csv')

    if args.mode == 'test':
        model.load_weights(f'model/{args.model_type}/{config["MODEL_WEIGHTS"][args.model_type]}')
        y_pred = model.predict(X_scaler)
        y_inverse = inverse_predict(y_pred, config)
        mse = mean_squared_error(y[int(config['MODEL']['slide']):], y_inverse, squared=False)   
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=100)

        sns.lineplot(data = y[int(config['MODEL']['slide']):], ax=ax, color='r',label='y_t')
        sns.lineplot(data = y_inverse, ax=ax,color='c',label='r')
        plt.show()
        print(args.model_type,'mse',mse)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default = 'train', required=False, help='either "train" or "test"')
    # parser.add_argument('-e', '--episode', type=int, default=10, help='number of episode to run')
    parser.add_argument('-t', '--model_type', type=str, default='lstm', required=False, help='"dnn", "conv1d", "conv2d", "lstm" or "transformer"')
    # parser.add_argument('-s', '--stock', type=str, required=False, default='TWII', help='stock index')
    parser.add_argument('-w', '--weight', type=str, required=False, help='stock portfolios')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read('config.ini')

    main(config, args)
    