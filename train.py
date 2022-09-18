from calendar import EPOCH
import pickle
import time
import numpy as np
import argparse
import re
import logging
import tqdm
import pickle
import pandas as pd
import os
import datetime
import warnings
from sklearn.preprocessing import StandardScaler

from utility.utility import maybe_make_dir, make_dir
from utility.techIndex import talib_index
from utility.model import dnn, lstm, conv1d, conv2d, transformer
from utility.training import callback

import configparser

# turn off system warning 
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def log(config, args):
    # configure logging
    timestamp = time.strftime('%Y%m%d%H%M')
    logging.basicConfig(
        filename=f'logs/{args.mode}/{args.model_type}/{timestamp}_{config["STOCK"]["stock"]}.log', 
        filemode='w',
        format='[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s', 
        datefmt='%m/%d/%Y %H:%M:%S', 
        level=logging.INFO
        )
    
    logging.info(f'Mode:                     {args.mode}')
    logging.info(f'Model Type:               {args.model_type}')
    logging.info(f'Training Object:          {config["STOCK"]["stock"]}')
    logging.info(f'Training Year:            {config["STOCK"]["train_year"]}')
    logging.info(f'Testing Year:             {config["STOCK"]["test_year"]}')
    logging.info(f'Window Slide:             {config["MODEL"]["slide"]} days')
    logging.info(f'Training Episode:         {config["MODEL"]["epoch"]}')
    logging.info(f'='*30)
    return logging

def read_data(config, args):
    # reading data by config
    if args.mode == 'train':
        year = config['STOCK']['train_year']   
    if args.mode == 'test':
        year = config['STOCK']['test_year']   
    index_list = config['STOCK']['index'].split(',')
    train_path = os.path.join('data', config['STOCK']['stock']+'_'+ year +'.csv')

    df_stock = pd.read_csv(train_path, index_col = 0)
    df_index = talib_index(df_stock)
    df_index = df_index[index_list]
    return df_index

def label(df, n = 1):
    # add predict label
    df['label'] = df['Close'].shift(n)  
    df = df.iloc[n:,:]
    return df.drop(columns='label'), df['label']

def scaler(X, y , config, args):
    scalerX_file = os.path.join('scaler', config['STOCK']['stock'] + config['STOCK']['scaler_X'])
    scalery_file = os.path.join('scaler', config['STOCK']['stock'] + config['STOCK']['scaler_y'])
    if args.mode == 'train':
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X = scaler_X.fit_transform(X.values)
        y = scaler_y.fit_transform(y.values.reshape(-1,1))
        pickle.dump(scaler_X, open(scalerX_file, 'wb'))
        pickle.dump(scaler_y, open(scalery_file, 'wb'))
    if args.mode == 'test':
        scaler_X = pickle.load(open(scalerX_file), 'rb')
        scaler_y = pickle.load(open(scalery_file), 'rb')
        X = scaler_X.transform(X.values)
        y = scaler_y.transform(y.values.reshape(-1,1))
    return X, y

def training_window(X, y, config):
    time_slide = int(config['MODEL']['slide'])
    X_=[]
    for i in range(time_slide, X.shape[0]): 
        tmp = X[i-time_slide:i]
        X_.append(tmp)
    y_ = y[time_slide:]    
    return np.array(X_), np.array(y_)

def dump_model(X, args):
    if args.model_type == 'dnn':
        return dnn(X)
    if args.model_type == 'conv1d':
        return conv1d(X)
    if args.model_type == 'conv2d':
        return conv2d(X)
    if args.model_type == 'lstm':
        return lstm(X)
    if args.model_type == 'transformer':
        return transformer(X)

def main(config, args):
    start_time = datetime.datetime.now()

    # basic setting
    make_dir(config)
    maybe_make_dir(f'logs/{args.mode}/{args.model_type}')

    # logging    
    logging = log(config, args)
    logging.info(f'Training Start')

    # reading data
    df = read_data(config, args)
    logging.info(f'Reading Data Finish')
    
    X, y = label(df)
    logging.info(f'Make Data Label')

    X, y = scaler(X, y, config, args)
    logging.info(f'Scale Data')

    X, y = training_window(X, y, config)
    logging.info(f'Make Training Windows')
    
    model = dump_model(X.shape, args)
    model.fit(X, y, epochs=int(config['MODEL']['epoch']))
    print(X.shape,y.shape)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default = 'train', required=False, help='either "train" or "test"')
    parser.add_argument('-e', '--episode', type=int, default=10, help='number of episode to run')
    parser.add_argument('-t', '--model_type', type=str, default='transformer', required=False, help='"dnn", "conv1d" or "lstm"')
    parser.add_argument('-s', '--stock', type=str, required=False, default='tech', help='stock portfolios')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read('config.ini')

    main(config, args)
    