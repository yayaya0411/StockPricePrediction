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
    print(f'Load Data Begin')
    df = pickle.load(open('data.pk', 'rb'))
    # df = read_data(config, args.mode)
    if args.mode == 'train':
        train  = df.trainSet 
        valid  = df.validSet 
        if args.model_type == 'milstm':
            target_train=[]
            pos_train=[]
            neg_train=[]
            index_train=[]
            y_train=[]
            for i in range(0,len(train)):
                target_train.append(train[i]['target_history'])
                pos_train.append(train[i]['pos_history'])
                neg_train.append(train[i]['neg_history'])
                index_train.append(train[i]['index_history'])
                y_train.append(train[i]['target_price'])

            X_train = {
                'target_history':target_train,
                'pos_history':pos_train,
                'neg_history':neg_train,
                'index_history':index_train,
            }
            y_train = np.array(y_train) 

            target_valid=[]
            pos_valid=[]
            neg_valid=[]
            index_valid=[]
            y_valid=[]
            for i in range(0,len(valid)):
                target_valid.append(valid[i]['target_history'])
                pos_valid.append(valid[i]['pos_history'])
                neg_valid.append(valid[i]['neg_history'])
                index_valid.append(valid[i]['index_history'])
                y_valid.append(valid[i]['target_price'])

            X_valid = {
                'target_history':target_valid,
                'pos_history':pos_valid,
                'neg_history':neg_valid,
                'index_history':index_valid,
            }
            y_valid = np.array(y_valid)

            model = load_model(X_valid, args.model_type)

        else:    
            X_train=[]
            y_train=[]
            for i in range(0,len(train)):
                X_train.append(train[i]['target_history'])
                y_train.append(train[i]['target_price'])
            X_train = np.array(X_train)
            y_train = np.array(y_train)

            X_valid=[]
            y_valid=[]
            for i in range(0,len(valid)):
                X_valid.append(valid[i]['target_history'])
                y_valid.append(valid[i]['target_price'])
            X_valid = np.array(X_valid)
            y_valid = np.array(y_valid)

            model = load_model(X_valid.shape, args.model_type)
        
    if args.mode == 'test':
        test  = df.testSet 
        X_test=[]
        y_test=[]
        for i in range(0,len(test)):
            X_test.append(test[i]['target_history'])
            y_test.append(test[i]['target_price'])
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        model = load_model(X_test.shape, args.model_type)

    print(f'Load Data Finish')

    target = np.array(X_train['target_history']).shape
    print('x train shape:',target)
    print('y train shape:',y_train.shape)
    # print(args.mode) 
    if args.mode == 'train':
        history = model.fit(
            X_train, y_train, 
            epochs=int(config['MODEL']['epoch']),
            verbose = 1,
            batch_size=512,
            validation_data=(X_valid, y_valid), 
            callbacks = callback(config, args, datetime_prefix)
        )
        y_pred = model.predict(X_valid)

        model.save_weights(f'mi_model/{args.model_type}/{args.model_type}')

        valid_mse = mean_squared_error(y_valid.reshape(-1,1), y_pred, squared=False)   
        print(args.model_type, 'mse',valid_mse)
        pd.DataFrame(history.history).to_csv(f'logs/csv_logger/{args.model_type}/{datetime_prefix}_{valid_mse}.csv')

    if args.mode == 'test':
        model.load_weights(f'mi_model/{args.model_type}/{args.model_type}')
        y_pred = model.predict(X_test)
        # y_inverse = inverse_predict(y_pred, config)
        mse = mean_squared_error(y_test.reshape(-1,1), y_pred, squared=False)   
        
        print(args.model_type,'mse',mse)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default = 'train', required=False, help='either "train" or "test"')
    parser.add_argument('-t', '--model_type', type=str, default='milstm', required=False, help='"dnn", "conv1d", "conv2d", "lstm" or "transformer"')
    parser.add_argument('-w', '--weight', type=str, required=False, help='stock portfolios')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read('config.ini')

    main(config, args)
    