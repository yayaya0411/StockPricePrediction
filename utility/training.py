import os
import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

def callback( model_name):
    callback = EarlyStopping(monitor="val_loss", patience=100, verbose=1, mode="auto")
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    checkpoint = ModelCheckpoint(
        filepath = model_name,
        monitor = 'val_loss',
        verbose = 1,
        save_best_only = True,
    )
    return [callback, tensorboard_callback,checkpoint]