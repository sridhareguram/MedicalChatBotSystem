
import random
import json
import pickle
from unittest import result
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("MINI_PROJECT\\intents.json").read())

words = pickle.load(open('MINI_PROJECT\\folder\\words.pkl', 'rb'))
classes = pickle.load(open('MINI_PROJECT\\folder\\classes.pkl','rb'))

model = load_model('MINI_PROJECT\\folder\\chatbot.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0] #returns the predicted classes
    ERROR_THRESHOLD = 0.25 # keeping some threshold value
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD] # Keeping only some threshold values

    results.sort(key=lambda x: x[1], reverse=True) # sorting based on some probabilities
    return_list = []
    for r in results:
        return_list.append({'intent' : classes[r[0]], 'probability': str(r[1])}) # returning classes and probabilities
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Go! Bot is running!")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)