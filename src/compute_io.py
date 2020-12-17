from keras.preprocessing.text import Tokenizer
import numpy as np
import re
from keras.utils import to_categorical
import io
import json
import random
from pickle import load
from keras.models import load_model


fname = input()

with open("../text_seq/text_sequences_4gram_" + fname + ".json") as seq_json:
    text_sequences_4gram = json.load(seq_json)

with open("../var_files/vocab.json") as vocab:
    vocabulary_size = len(json.load(vocab)[0])

tokenizer = load(open('../var_files/tokenizer_Model4', 'rb'))


##### 4 gram #####

sequences = tokenizer.texts_to_sequences(text_sequences_4gram)
unique_words = tokenizer.index_word
unique_wordsApp = tokenizer.word_counts
n_sequences = np.empty([len(sequences), 4], dtype='int32')

for i in range(len(sequences)):
    n_sequences[i] = sequences[i]


train_inputs_4gram = n_sequences[:, :-1]
train_targets_4gram = n_sequences[:, -1]
train_targets_4gram = to_categorical(
    train_targets_4gram,
    num_classes=vocabulary_size + 1)


np.save('../raw_io_files/input_train_' + fname, train_inputs_4gram)
np.save('../raw_io_files/input_target_' + fname, train_targets_4gram)
