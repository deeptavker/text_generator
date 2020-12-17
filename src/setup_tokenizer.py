from keras.preprocessing.text import Tokenizer
import numpy as np
import re
from keras.utils import to_categorical
import io
import json
from pickle import dump, load

with open("../var_files/vocab.json") as vocab:
    v = json.load(vocab)
    v = v[0]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(v)

dump(tokenizer, open('../var_files/tokenizer_Model4', 'wb'))

np.save('../var_files/batch_counter', np.array([0]))

shape = len(v), len(v)
mapping_2gram = np.zeros(shape=shape)

np.save('../var_files/mapping_2gram', mapping_2gram)
