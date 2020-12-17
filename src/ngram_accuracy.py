import json
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model, Model
from pickle import load
import keras
import tensorflow as tf
import numpy as np
import keras.backend as K


model = load_model('../var_files/word_pred_Model4.hdf5', compile=False)

tokenizer = load(open('../var_files/tokenizer_Model4', 'rb'))
seq_len = 3

fnames = ['input1.txt', 'input2.txt']


res = 0
total = 0


for name in fnames:
    with open('../ngrams/4gram_' + name + '.json') as ng:
        ngrams = json.load(ng)

    total += len(ngrams)
    for elem in ngrams:

        inp = " ".join(elem[:3])
        encoded_text = tokenizer.texts_to_sequences([inp])[0]
        pad_encoded = pad_sequences(
            [encoded_text],
            maxlen=seq_len,
            truncating='pre')
        pred_word_ind = model.predict(pad_encoded, verbose=0)[0]
        pred_word_ind1 = pred_word_ind.argmax(axis=-1)
        pred_word = tokenizer.index_word[pred_word_ind1]

        res += (pred_word == elem[3])

print("4-Gram Accuracy = {}".format(1.0 * res / total))
