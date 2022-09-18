import os

def maybe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_dir(config):
    directorys = config['DIRECTORY']['directory'].split(',')
    for directory in directorys:
        maybe_make_dir(directory)
