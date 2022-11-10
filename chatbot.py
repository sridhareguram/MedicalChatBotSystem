import imp
from operator import le
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential,Model
from keras.layers.embeddings import  Embedding
from keras.layers import Input,Activation,Dense,Permute,Dropout,add,dot,concatenate,LSTM

with open("MINI_PROJECT\\Dataset\\train\\train_qa.txt",'rb') as f:
    train_data = pickle.load(f)

with open('MINI_PROJECT\\Dataset\\train\\test_qa.txt', 'rb') as f:
    test_data = pickle.load(f)

type(test_data)

len(train_data)
len(test_data)
train_data[0] # story 0, question 1, answer 2
' '.join(train_data[0][0])
train_data[0][2]

all_data = train_data + test_data
len(all_data)

vocab = set()

for story,question,answer in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))

vocab.add('no')
vocab.add('yes')
vocab_len = len(vocab)+1

# Longest Story
all_story_lens = [len(data[0]) for data in all_data]
max_story_len = max(all_story_lens)
max_story_len

max_question_len = max([len(data[1]) for data in all_data])
max_question_len

tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)

tokenizer.word_index

train_story_text = []
train_question_text = []
train_answers = []
for story,question,answer in train_data:
    train_story_text.append(story)
    train_question_text.append(question)
    train_answers.append(answer)

train_story_seq = tokenizer.texts_to_sequences(train_story_text)
len(train_story_seq)
# train_story_seq

def vectorize_stories(data, word_index = tokenizer.word_index, max_story_len = max_story_len, max_question_len = max_question_len):
    # Stories
    X = []
    # questions
    Xq = []
    # Correct answer (yes/no)
    Y = []

    for story,question,answer in data:

        x = [word_index[word.lower()] for word in story]
        q = [word_index[word.lower()] for word in question]

        y = np.zeros(len(word_index)+1)

        y[word_index[answer]] = 1
        X.append(x)
        Xq.append(q)
        Y.append(y)

    return (pad_sequences(X,maxlen=max_story_len), pad_sequences(Xq,maxlen=max_question_len), np.array(Y) )
    
input_train , queries_train, answers_train = vectorize_stories(train_data)

input_train

answers_train
sum(answers_train)
   
input_test , queries_test, answers_test = vectorize_stories(test_data)



