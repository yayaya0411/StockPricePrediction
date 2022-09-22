import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, Conv2D, Flatten, BatchNormalization, MaxPooling1D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import os
import datetime
import numpy as np

batch_size = 32
seq_len = 30

d_k = 256
d_v = 256
n_heads = 12
ff_dim = 256

'''
dnn 
'''
def dnn(n_obs):
    """ A multi-layer perceptron """
    # print('\n',n_obs[0],'\n')
    model = Sequential()
    model.add(Dense(units=256, input_shape=[n_obs[1]], activation="relu"))
    model.add(Dense(units=512, activation="relu"))
    model.add(Dropout(0.3))
    # model.add(Dense(units=1024, activation="relu"))
    # model.add(Dense(1, activation="linear"))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer='adam', metrics=['mse','mae', 'mape'])
    print(model.summary())
    return model

'''
Conv1d 
'''
def conv1d(n_obs):
    kernel_size=2
    strides=1
    padding = 'same'
    model = Sequential()
    model.add(Conv1D(filters = 64, kernel_size=kernel_size, strides=strides, padding=padding, activation = 'relu',input_shape=(n_obs[1],n_obs[2])))
    model.add(Conv1D(filters = 128, kernel_size=kernel_size, strides=strides, padding=padding, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    # model.add(Conv1D(filters = 512, kernel_size=kernel_size, strides=strides, padding=padding, activation = 'relu'))
    # model.add(Dropout(0.3))
    # model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss="mse", optimizer='adam', metrics=['mse','mae', 'mape'])
    print(model.summary())
    return model

'''
Conv2d 
'''
def conv2d(n_obs):
    kernel_size=(2,2)
    # strides=(1,1)
    padding = 'same'
    model = Sequential()
    model.add(Conv2D(filters = 64, kernel_size=kernel_size,  padding=padding, activation = 'relu',input_shape=(n_obs[1],n_obs[2],1)))
    model.add(Conv2D(filters = 128, kernel_size=kernel_size,  padding=padding, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))
    model.add(Flatten())
    model.add(Dense(1))
    # model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['mae', 'mape'])
    model.compile(loss="mse", optimizer='adam', metrics=['mse','mae', 'mape'])
    print(model.summary())
    return model

'''
LSTM 
'''
def lstm(n_obs):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True,input_shape=(n_obs[1],n_obs[2])))
    # model.add(LSTM(128, dropout=0.2, return_sequences=True))
    model.add(LSTM(128, return_sequences=True,dropout=0.3))
    # model.add(LSTM(256, return_sequences=True,dropout=0.3))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer='adam', metrics=['mse', 'mae', 'mape'])
    print(model.summary())
    return model

'''
Transformer 
'''


class Time2Vector(Layer):
    def __init__(self, seq_len, **kwargs):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        '''Initialize weights and biases with shape (batch, seq_len)'''
        self.weights_linear = self.add_weight(name='weight_linear',
                                              shape=(int(self.seq_len),),
                                              initializer='uniform',
                                              trainable=True)

        self.bias_linear = self.add_weight(name='bias_linear',
                                           shape=(int(self.seq_len),),
                                           initializer='uniform',
                                           trainable=True)

        self.weights_periodic = self.add_weight(name='weight_periodic',
                                                shape=(int(self.seq_len),),
                                                initializer='uniform',
                                                trainable=True)

        self.bias_periodic = self.add_weight(name='bias_periodic',
                                             shape=(int(self.seq_len),),
                                             initializer='uniform',
                                             trainable=True)

    def call(self, x):
        '''Calculate linear and periodic time features'''
        x = tf.math.reduce_mean(x[:, :, :4], axis=-1)
        time_linear = self.weights_linear * x + self.bias_linear  # Linear time feature
        # Add dimension (batch, seq_len, 1)
        time_linear = tf.expand_dims(time_linear, axis=-1)

        time_periodic = tf.math.sin(tf.multiply(
            x, self.weights_periodic) + self.bias_periodic)
        # Add dimension (batch, seq_len, 1)
        time_periodic = tf.expand_dims(time_periodic, axis=-1)
        # shape = (batch, seq_len, 2)
        return tf.concat([time_linear, time_periodic], axis=-1)

    def get_config(self):  # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'seq_len': self.seq_len})
        return config


class SingleAttention(Layer):
    def __init__(self, d_k, d_v):
        super(SingleAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        self.query = Dense(self.d_k,
                           input_shape=input_shape,
                           kernel_initializer='glorot_uniform',
                           bias_initializer='glorot_uniform')

        self.key = Dense(self.d_k,
                         input_shape=input_shape,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='glorot_uniform')

        self.value = Dense(self.d_v,
                           input_shape=input_shape,
                           kernel_initializer='glorot_uniform',
                           bias_initializer='glorot_uniform')

    def call(self, inputs):  # inputs = (in_seq, in_seq, in_seq)
        q = self.query(inputs[0])
        k = self.key(inputs[1])

        attn_weights = tf.matmul(q, k, transpose_b=True)
        attn_weights = tf.map_fn(lambda x: x/np.sqrt(self.d_k), attn_weights)
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        v = self.value(inputs[2])
        attn_out = tf.matmul(attn_weights, v)
        return attn_out

#############################################################################


class MultiAttention(Layer):
    def __init__(self, d_k, d_v, n_heads):
        super(MultiAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.attn_heads = list()

    def build(self, input_shape):
        for n in range(self.n_heads):
            self.attn_heads.append(SingleAttention(self.d_k, self.d_v))

        # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7
        self.linear = Dense(input_shape[0][-1],
                            input_shape=input_shape,
                            kernel_initializer='glorot_uniform',
                            bias_initializer='glorot_uniform')

    def call(self, inputs):
        attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
        concat_attn = tf.concat(attn, axis=-1)
        multi_linear = self.linear(concat_attn)
        return multi_linear

#############################################################################


class TransformerEncoder(Layer):
    def __init__(self, d_k, d_v, n_heads, ff_dim, dropout=0.1, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.attn_heads = list()
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)
        self.attn_dropout = Dropout(self.dropout_rate)
        self.attn_normalize = LayerNormalization(
            input_shape=input_shape, epsilon=1e-6)

        self.ff_conv1D_1 = Conv1D(
            filters=self.ff_dim, kernel_size=1, activation='relu')
        # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1] = 7
        self.ff_conv1D_2 = Conv1D(filters=input_shape[0][-1], kernel_size=1)
        self.ff_dropout = Dropout(self.dropout_rate)
        self.ff_normalize = LayerNormalization(
            input_shape=input_shape, epsilon=1e-6)

    def call(self, inputs):  # inputs = (in_seq, in_seq, in_seq)
        attn_layer = self.attn_multi(inputs)
        attn_layer = self.attn_dropout(attn_layer)
        attn_layer = self.attn_normalize(inputs[0] + attn_layer)

        ff_layer = self.ff_conv1D_1(attn_layer)
        ff_layer = self.ff_conv1D_2(ff_layer)
        ff_layer = self.ff_dropout(ff_layer)
        ff_layer = self.ff_normalize(inputs[0] + ff_layer)
        return ff_layer

    def get_config(self):  # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'d_k': self.d_k,
                       'd_v': self.d_v,
                       'n_heads': self.n_heads,
                       'ff_dim': self.ff_dim,
                       'attn_heads': self.attn_heads,
                       'dropout_rate': self.dropout_rate})
        return config


def transformer(n_obs):
    '''Initialize time and transformer layers'''
    time_embedding = Time2Vector(seq_len)
    attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

    '''Construct model'''
    # in_seq = Input(shape=(seq_len, 22))
    in_seq = Input(shape=(n_obs[1],n_obs[2]))
    x = time_embedding(in_seq)
    x = Concatenate(axis=-1)([in_seq, x])
    x = attn_layer1((x, x, x))
    x = attn_layer2((x, x, x))
    x = attn_layer3((x, x, x))
    x = GlobalAveragePooling1D(data_format='channels_first')(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    out = Dense(1)(x)

    model = Model(inputs=in_seq, outputs=out)
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    # model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['mae', 'mape'])
    print(model.summary())
    return model

