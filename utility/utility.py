import os
import logging
import time

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

