from __future__ import print_function
import keras
import numpy as np
from keras import optimizers
import tensorflow as tf
import keras.backend as k
from keras.callbacks import LambdaCallback


model = keras.models.load_model('../var_files/word_pred_Model4.hdf5')

inps = np.load('../raw_io_files/input_train_sample5.txt.npy')
outps = np.load('../raw_io_files/input_target_sample5.txt.npy')

model.fit(x=inps, y=outps, batch_size=128, epochs=400, verbose=1)

model.save('../var_files/temp_word_pred_Model4.hdf5')


# fisher=np_fisher, star_vars=np_weights, lamda=100
