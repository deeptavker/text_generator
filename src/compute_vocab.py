from keras.preprocessing.text import Tokenizer
import numpy as np
import re
from keras.utils import to_categorical
import io
import json
import nltk
from nltk.util import ngrams
import scipy.stats as st

#computes vocab (adds to it if already exists and creates text_sequences)

fname = input()
d = io.open('../training_text/' + fname, errors='ignore')
data = d.readlines()

############################################################
################# Cleaning the text ########################
############################################################
# Method that will clean the data:
def clean_text(text):
    text = text.lower() #convert all the chracters into small letters
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'ll", "will", text)
    text = re.sub(r"\'re", "are", text)
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"n't", "not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}'_+=|.!?,]", "", text)
    text = text.replace("[", "")
    text = text.replace("]", "")
    return text

clean_data = []
for text in data:
    a = re.sub(r'[^a-zA-z ]+', '', text).strip()
    if len(a)>0:
        clean_data.append(clean_text(a))
    else:
        None

############################################################
################# Tokenization and Section data ############
############################################################
tokens = "\n".join(clean_data).split()


############################################################
####### Structuring sequence wise data for processing ######
############################################################
train_len = 3+1 # 1 for output  
text_sequences = []
for i in range(train_len, len(tokens)):
    seq = tokens[i - train_len:i]
    text_sequences.append(seq)
    
with open('../text_seq/text_sequences_4gram_' + fname + '.json','w') as fp:
    json.dump(text_sequences,fp)


############################################################
### Setting up input and output arrays via Tokenization ####
############################################################


    
res = list(set(tokens))

with open("../var_files/vocab.json") as vocab:
    a = json.load(vocab)

    if len(a[0]) == 0:
        a = [res]
    else:
        res_t = a[0]
        res.extend(res_t)
        res = list(set(res))
        a = [res]
        
with open("../var_files/vocab.json", 'w') as vocab:
    json.dump(a, vocab)


### N grams generator

ngram_tokenize = nltk.word_tokenize(' '.join(tokens))

fourgrams = list(ngrams(ngram_tokenize, 4))
seq = {}
for elem in fourgrams:
    seq[elem[:3]] = []
for elem in fourgrams:
    seq[elem[:3]].append(elem[3])

a = []
for key in seq.keys():
    a.append(list(key) + [st.mode(seq[key])[0][0]])
    
with open('../ngrams/4gram_'+ fname + '.json', 'w') as gr:
    json.dump(a, gr)
