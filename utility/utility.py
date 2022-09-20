import os

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

