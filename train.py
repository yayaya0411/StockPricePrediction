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

from datetime import datetime
from utility.utility import maybe_make_dir
import configparser

from config import *


def main(config, args):
    stock = config['STOCK']['stock']
    train_year = config['STOCK']['train_year']
    test_year = config['STOCK']['test_year']
    # print(config['STOCK'])
    train = os.path.join(config['DIRECTORY']['stock'],stock+'_'+train_year+'.csv')
    df_stock = pd.read_csv(train)
    print(df_stock)
    # return

    # configure logging
    logging.basicConfig(
        filename=f'logs/{args.mode}/{args.model_type}/{model_prefix}_{args.mode}.log', 
        filemode='w',
        format='[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s', 
        datefmt='%m/%d/%Y %H:%M:%S', 
        level=logging.INFO
        )
    logging.info(f'Mode:                     {args.mode}')
    logging.info(f'Model Type:               {args.model_type}')
    logging.info(f'Training Object:          {stock_name}')
    logging.info(f'Portfolio Stock:          {stock_code}')
    logging.info(f'Window Slide:             {slide} days')
    logging.info(f'Turn to Ratio:            {to_ratio}')
    logging.info(f'Turn to Gray:             {to_gray}')
    logging.info(f'Buy/Sell Stocks:          {env.buy_stock} per action')
    logging.info(f'Model Weights:            {args.weights}')
    logging.info(f'Training Episode:         {args.episode}')
    logging.info(f'Initial Invest Value:    ${args.initial_invest:,}')
    logging.info(f'='*30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--episode', type=int, default=50, help='number of episode to run')
    parser.add_argument('-m', '--mode', type=str, required=True, help='either "train" or "test"')
    parser.add_argument('-t', '--model_type', type=str, required=True, help='"dnn", "conv1d" or "lstm"')
    parser.add_argument('-s', '--stock', type=str, required=True, default='tech', help='stock portfolios')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read('config.ini')

    main(config, args)
    