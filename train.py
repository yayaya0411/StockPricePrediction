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


def main(config):
    stock = config['STOCK']['stock']
    train_year = config['STOCK']['train_year']
    test_year = config['STOCK']['test_year']
    # print(config['STOCK'])
    train = os.path.join(config['DIRECTORY']['stock'],stock+'_'+train_year+'.csv')
    df_stock = pd.read_csv(train)
    print(df_stock)
    # return

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    main(config)
    