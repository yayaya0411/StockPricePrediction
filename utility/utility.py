import os
import logging
import time
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from utility.techIndex import talib_index
from utility.model import dnn, lstm, conv1d, conv2d, transformer

def log(config):
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

def maybe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_dir(config):
    directorys = config['DIRECTORY']['directory'].split(',')
    model_directorys = config['DIRECTORY']['model_type'].split(',')
    for directory in directorys:
        if directory == 'logs':
            for sub_dir in model_directorys:
                maybe_make_dir(os.path.join(directory, 'csv_logger', sub_dir))
                maybe_make_dir(os.path.join(directory, 'tensorboard', sub_dir))
        if directory == 'model':
            for sub_dir in model_directorys:
                maybe_make_dir(os.path.join(directory, sub_dir))
        maybe_make_dir(directory)

def read_data(config, args):
    # reading data by config
    if args == 'train':
        year = config['STOCK']['train_year']   
    if args == 'test':
        year = config['STOCK']['test_year']   
    index_list = config['STOCK']['index'].split(',')
    train_path = os.path.join('data', config['STOCK']['stock']+'_'+ year +'.csv')
    print(train_path)
    df_stock = pd.read_csv(train_path, index_col = 0)
    df_index = talib_index(df_stock)
    df_index = df_index[index_list]
    return df_index

def label(df, n = 1):
    # add predict label
    df['label'] = df['Close'].shift(n)  
    df = df.iloc[n:,:]
    return df.drop(columns='label'), df['label']

def scaler(X, y ,config, args):
    scalerX_file = os.path.join('scaler', config['STOCK']['stock'] + config['STOCK']['scaler_X'])
    scalery_file = os.path.join('scaler', config['STOCK']['stock'] + config['STOCK']['scaler_y'])
    if args == 'train':
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X = scaler_X.fit_transform(X.values)
        y = scaler_y.fit_transform(y.values.reshape(-1,1))
        pickle.dump(scaler_X, open(scalerX_file, 'wb'))
        pickle.dump(scaler_y, open(scalery_file, 'wb'))
    if args == 'test':
        scaler_X = pickle.load(open(scalerX_file, 'rb'))
        scaler_y = pickle.load(open(scalery_file, 'rb'))
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

def split_valid(X, y, config):
    split_ratio = float(config['STOCK']['valid_ratio'])
    split_ind = int((X.shape[0] * (1 - split_ratio)))
    return X[:split_ind], y[:split_ind], X[split_ind:], y[split_ind:], split_ind

def load_model(X, args):
    if args == 'dnn':
        return dnn(X)
    if args == 'conv1d':
        return conv1d(X)
    if args == 'conv2d':
        return conv2d(X)
    if args == 'lstm':
        return lstm(X)
    if args == 'transformer':
        return transformer(X)

def inverse_predict(y, config):
    scalery_file = os.path.join('scaler', config['STOCK']['stock'] + config['STOCK']['scaler_y'])
    scaler_y = pickle.load(open(scalery_file, 'rb'))
    y = scaler_y.inverse_transform(y)
    return y

