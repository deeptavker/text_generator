from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model, Model
from pickle import load
import numpy as np
import keras.utils
from keras.utils import np_utils
import pandas.util.testing as tm
from prettytable import PrettyTable


model = load_model('../var_files/word_pred_Model4.hdf5')

tokenizer = load(open('../var_files/tokenizer_Model4', 'rb'))

inp_words_seq_len = 3


def gen_text(model, tokenizer, seq_len, seed_text):
    output_text = []
    input_text = seed_text
    words = []
    encoded_text = tokenizer.texts_to_sequences([input_text])[0]
    pad_encoded = pad_sequences(
        [encoded_text],
        maxlen=seq_len,
        truncating='pre')
    pred_word_ind = model.predict(pad_encoded, verbose=0)[0]
    pred_word_ind1 = pred_word_ind.argmax(axis=-1)
    pred_word, prob = tokenizer.index_word[pred_word_ind1], pred_word_ind[pred_word_ind1]
    words.append([prob, pred_word])

    return words


print('\n\n Enter input text')
while True:
    print("-----------------------------------------------------------")
    seed_text = input('\n\nEnter input string : ')

    if seed_text.lower() == '--exit':
        break
    else:
        out = gen_text(
            model,
            tokenizer,
            seq_len=inp_words_seq_len,
            seed_text=seed_text)
        t = PrettyTable(['Next word', 'Probability'])
        for words in out:
            t.add_row([words[1], words[0]])
        print(t)
