from sklearn.ensemble import AdaBoostClassifier

import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import MaxPooling2D, Conv1D,MaxPooling1D,TimeDistributed
from tensorflow.keras.layers import Dense ,Dropout , LSTM ,TimeDistributed,Flatten, Conv1D, MaxPooling1D,BatchNormalization
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras import layers
from tensorflow.keras.layers import LayerNormalization
# from tensorflow.python.keras.layers import Input, Embedding, Dot, Reshape, Dense
# from tensorflow.python.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input,Embedding,LSTM,Dense
from tensorflow.keras.models import Model
# from tensorflow.keras import backend as K
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor,KerasClassifier
from inspect import signature

def lstm4_clf_v1(input_shape , n_classes):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(input_shape[0],input_shape[1])))
    model.add(LSTM(25,return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(12,return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(6,return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(3,return_sequences=True))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(90, activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.0001)))
    model.add(BatchNormalization())
    model.add(Dense(25, activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.0001)))
    model.add(BatchNormalization())
    model.add(Dense(2, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(l=0.0001)))
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0001, decay_steps=20, decay_rate=0.9)
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=lr_schedule), metrics=['accuracy'])
    model.summary()
    return model

def lstm4_clf_v2(input_shape , n_classes):
    model = Sequential()
    model.add(LSTM(64,return_sequences=True,input_shape=(input_shape[0],input_shape[1])))
    model.add(LSTM(128,return_sequences=True))
    model.add(LSTM(128,return_sequences=True))
    model.add(LSTM(64,return_sequences=True))
    model.add(Flatten())
    # model.add(TimeDistributed(Dense(64, activation='relu')))
    # model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(Dense(2, activation='softmax'))
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0001, decay_steps=20, decay_rate=0.9)
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=lr_schedule), metrics=['accuracy'])
    model.summary()
    return model

def lstm4_clf_v3(input_shape , n_classes):
    model = Sequential()
    model.add(LSTM(50,input_shape=(input_shape[0],input_shape[1]), return_sequences=True))
    model.add(LSTM(50,return_sequences=True))
    model.add(keras.layers.LayerNormalization(axis=1 , center=True , scale=True))
    model.add(LSTM(50,return_sequences=True))
    model.add(keras.layers.LayerNormalization(axis=1 , center=True , scale=True))
    model.add(LSTM(50,return_sequences=True))
    model.add(keras.layers.LayerNormalization(axis=1 , center=True , scale=True))
    model.add(LSTM(50,return_sequences=True))
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(l=0.001)))
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=20, decay_rate=0.9)
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=lr_schedule), metrics=['accuracy'])
    model.summary()
    return model

def lstm_clf_v4(inputs,
                output,
                loss="binary_crossentropy",
                metrics = ['accuracy'],
                ):
    input = keras.Input(shape=(inputs.shape[1], inputs.shape[2]))
    x = LSTM(64, dropout = 0.2, return_sequences=True)(input)
    x = LSTM(128, dropout = 0.2, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = LSTM(128, dropout = 0.2, return_sequences=True)(x)
    x = LSTM(64, dropout = 0.2, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = LSTM(32, dropout = 0.2, return_sequences=False)(x)
    z = (Dense(output.shape[1], activation='sigmoid', kernel_regularizer=keras.regularizers.l2(l=0.001)))(x)
    model = keras.Model(inputs = input, outputs= z)
    
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=10, decay_rate=0.9)
    # optimizer = Adam(learning_rate=lr_schedule)
    # model.compile(loss=loss, metrics = metrics, optimizer = optimizer)
    # model.summary()
    return model, 'lstm_v4'
    
def lstm_reg_v1(inputs,
                output,
                loss="mean_squared_error",
                metrics = ['mean_squared_error'],
                optimizer="adam"):
    input = keras.Input(shape=(inputs.shape[1], inputs.shape[2]))
    x = LSTM(64, dropout = 0.2, return_sequences=True)(input)
    x = LSTM(128, dropout = 0.2, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = LSTM(128, dropout = 0.2, return_sequences=True)(x)
    x = LSTM(64, dropout = 0.2, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = LSTM(32, dropout = 0.2, return_sequences=False)(x)
    z = (Dense(output.shape[1], activation = 'linear'))(x)
    model = keras.Model(inputs = input, outputs= z)
    # model.compile(loss=loss, metrics = metrics, optimizer = optimizer)
    # model.summary()
    return model, 'lstm_v1'

def conv1d_reg_v1(
    inputs,
    output,
    loss="mean_squared_error",
    metrics = ['mean_squared_error'],
    optimizer="adam",
    kernel_size=2,
    strides=1,
    padding='same',
    dropout=0.2
    ):
    input = keras.Input(shape=(inputs.shape[1], inputs.shape[2]))
    x = Conv1D(filters = 64, kernel_size=kernel_size, strides=strides, padding=padding, activation = 'relu')(input)
    x = Conv1D(filters = 128, kernel_size=kernel_size, strides=strides, padding=padding, activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = MaxPooling1D()(x)
    x = Conv1D(filters = 128, kernel_size=kernel_size, strides=strides, padding=padding, activation = 'relu')(x)
    x = Conv1D(filters = 64, kernel_size=kernel_size, strides=strides, padding=padding, activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = MaxPooling1D()(x)
    x = Conv1D(filters = 32, kernel_size=kernel_size, strides=strides, padding=padding, activation = 'relu')(x)
    x = Flatten()(x)
    z = Dense(output.shape[1], name='output')(x)
    model = keras.Model(inputs = input, outputs= z)
    # model.compile(loss=loss, metrics = metrics, optimizer = optimizer)
    # model.summary()
    return model, 'conv1d_v1'


def lstm_ada_v1():
    
    # model = Sequential()
    # model.add(LSTM(64,dropout = 0.2,return_sequences=True,input_shape=(60,22)))  # shape = (time_slide, feature)
    # model.add(LSTM(128,dropout = 0.2,return_sequences=True))
    # model.add(BatchNormalization())
    # model.add(LSTM(128,dropout = 0.2,return_sequences=True))
    # model.add(LSTM(64,dropout = 0.2,return_sequences=True))
    # model.add(BatchNormalization())
    # model.add(LSTM(32,dropout = 0.2,return_sequences=True))
    # model.add(Flatten())
    # model.add(Dense(1, activation='sigmoid'))
    
    input = keras.Input(shape=(60, 22)) # shape = (time_slide, feature)
    x = LSTM(64, dropout = 0.2, return_sequences=True)(input)
    x = LSTM(128, dropout = 0.2, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = LSTM(128, dropout = 0.2, return_sequences=True)(x)
    x = LSTM(64, dropout = 0.2, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = LSTM(32, dropout = 0.2, return_sequences=False)(x)
    z = Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs = input, outputs= z)
    
    model.compile(loss="binary_crossentropy",metrics = ["accuracy"],optimizer = 'adam')
    return model

class CustomKerasClassifier(KerasClassifier):
  def fit(self, x, y, sample_weight=None, **kwargs):
    if sample_weight is not None:
        kwargs['sample_weight'] = sample_weight
    else:
        kwargs['sample_weight'] = [0.5,0.5] 
        # print(type(sample_weight))
    return super(CustomKerasClassifier, self).fit(x, y, **kwargs)
    #return super(KerasClassifier, self).fit(x, y, sample_weight=sample_weight)

def adaboost_clf():
    # each model in keras
    estimator = CustomKerasClassifier(build_fn= lstm_ada_v1, epochs=50, batch_size=256, verbose=2,sample_weight=None)
    boosted = AdaBoostClassifier(base_estimator=estimator, n_estimators=30,random_state=0 )
    # print(signature(boosted.fit),'\n')
    return boosted




