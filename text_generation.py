from statistics import mode
import spacy
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding
from pickle import dump,load
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input,Activation,Dense,Permute,Dropout,add,dot,concatenate,LSTM
import tensorflow as tf
import random
from keras.models import load_model


from keras.preprocessing.text import Tokenizer
def read_file(file_path):
    with open(file_path) as f:
        str_text = f.read()
        
    return str_text

nlp = spacy.load('en_core_web_sm', disable=['parser','tagger', 'ner'])

nlp.max_length = 1198623

def seperate_punc(doc_text):
    return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n \n\n\n\n\n \n \n\n\n\n !"#$%&()*+,--./:;<=>?@[\\]^_`{|}~\',\t\n']

d = read_file('MINI_PROJECT\\song.txt')

tokens = seperate_punc(d)

# print(len(tokens))
print(tokens)

# 25words --> network predicts 26th word
train_len = 25+1
text_sequences = []

for i in range(train_len,len(tokens)):
    seq = tokens[i-train_len:i]
    text_sequences.append(seq)

type(text_sequences)

' '.join(text_sequences[0])
print([' '.join(text_sequences[2])])


tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences) #Updates the intial vocabulary

sequences = tokenizer.texts_to_sequences(text_sequences) #converts words into integers
sequences[0]

tokenizer.index_word # 13441: '357', 13442: 'precedes', 13443: 'prophesies', 13444: 'tains', 13445: 'explosion', 13446: 'graceful', 13447: 'serpentines', 13448: '-lines', 13449: 'halters', 13450: 'realise', 13451: 'poker', 13452: '1892', 13453: 'ob', 13454: 'toronto', 13455: 'stewart', 13456: 'bp', 13457: 'bequeatbefc', 13458: 'tuniverait', 13459: '2384', 13460: 'sltbrars', 13461: 'm6', 13462: 'ipurcbasefc'
for i in sequences[0]:
    print(f"{i} : {tokenizer.index_word[i]}")

tokenizer.word_counts

vocabulary_size = len(tokenizer.word_counts)
vocabulary_size

type(sequences)

sequences = np.array(sequences)
sequences #[[13462,    16,  6365, ...,  2011,     2,  4135]

X = sequences[:,:-1] #[[13462, 16,  6365, ..., 1,  2011, 2]
y = sequences[:,-1]

y = to_categorical(y,num_classes = vocabulary_size+1)
y #[0., 0., 0., ..., 0., 1., 0.]

seq_len = X.shape[1]

seq_len #25
X.shape #(112773, 25)

def create_model(vocabulary_size, seq_len):

    model = Sequential()
    model.add(Embedding(vocabulary_size,seq_len, input_length=seq_len)) # input_dim, output_dim,
    model.add(LSTM(seq_len*2, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(50, activation='relu'))

    model.add(Dense(vocabulary_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model

model = create_model(vocabulary_size+1, seq_len)

model.fit(X,y,batch_size=128,epochs=200,verbose=1)


# model.save('C:\\Users\\vramt\\OneDrive\\Desktop\\Programming\\MINI_PROJECT\\mymodel3.h5')
# model = load_model('C:\\Users\\vramt\\OneDrive\\Desktop\\Programming\\MINI_PROJECT\\mymodel2.h5')

# dump(tokenizer,open("C:\\Users\\vramt\\OneDrive\\Desktop\\Programming\\MINI_PROJECT\\tokenizer", 'wb'))


def generate_text(model, tokenizer, seq_len, seed_text, num_gen_words):
    output_text = []
    input_text = seed_text
    #25 words predict 1 word add ot 
    for i in range(num_gen_words):
        encode_text = tokenizer.texts_to_sequences([input_text])[0]

        pad_encoded = pad_sequences([encode_text], maxlen = seq_len, truncating='pre')
        # print(pad_encoded)

        predict_x=model.predict(pad_encoded, verbose=0) 
        classes_x=np.argmax(predict_x,axis=1)
        # pred_word_ind = model.predict_classes(pad_encoded, verbose=0)[1]
        # print(classes_x)
        pred_word = tokenizer.index_word[classes_x[0]]
        # print(pred_word)
        input_text += ' '+pred_word

        output_text.append(pred_word)

    return ' '.join(output_text) 

# genrating sequence words
# # random.seed(101)
# random_pick = random.randint(0, len(text_sequences))
# random_seed_text = text_sequences[random_pick]
# seed_text = ' '.join(random_seed_text)
seed_text = input()
seed_text

print(generate_text(model, tokenizer, seq_len, seed_text, 25))

























# print(20)