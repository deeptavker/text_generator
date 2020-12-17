
from keras.preprocessing.text import Tokenizer
import numpy as np
from numpy.testing import assert_allclose
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from pickle import dump, load
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.models import load_model, Input, Model
from keras.layers import Dropout
import json
import h5py

with open("../var_files/vocab.json") as vocab:
    vocabulary_size = len(json.load(vocab)[0])

seq_len = 3

visible = Input(shape=(seq_len,))
embedding = Embedding(
    vocabulary_size + 1,
    seq_len,
    input_length=seq_len)(visible)
hidden1 = LSTM(200, return_sequences=True)(embedding)
hidden2 = LSTM(200)(hidden1)
hidden3 = Dense(200, activation='relu')(hidden2)

output = Dense(vocabulary_size + 1, activation='softmax')(hidden3)
model = Model(inputs=visible, outputs=output)
opt_adam = optimizers.Adam(lr=0.001)
print(model.summary())
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt_adam,
    metrics=['accuracy'])

path = 'word_pred_Model4.hdf5'
checkpoint = ModelCheckpoint(
    path,
    monitor='loss',
    verbose=1,
    save_best_only=True,
    mode='min')
# model.fit(train_inputs,train_targets,epochs=600,verbose=1,callbacks=[checkpoint])

model.save('../var_files/word_pred_Model4.hdf5')
