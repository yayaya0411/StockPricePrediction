import os
import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, TensorBoard
from tqdm.keras import TqdmCallback

def callback(config, args, datetime_prefix=None):

    stopping = EarlyStopping(monitor="val_loss", patience=100, verbose=1, mode="auto")

    logdir = os.path.join(f'logs/tensorboard/{args.model_type}/{datetime_prefix}')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    
    # csv_logger = CSVLogger(f'logs/csv_logger/{args.model_type}/{datetime_prefix}', separator = ',', append = False)
    model_setting = config['MODEL']
    checkpoint = ModelCheckpoint(
        filepath = os.path.join(f'model/{args.model_type}/{datetime_prefix}_e{model_setting["epoch"]}_s{model_setting["slide"]}.h5'),
        monitor = 'val_loss',
        verbose = 1,
        save_best_only = True,
    )

    return [stopping, TqdmCallback(verbose=2), tensorboard_callback, checkpoint]