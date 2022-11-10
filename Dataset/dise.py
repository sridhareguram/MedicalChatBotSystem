import pickle

clf = pickle.load(open("MINI_PROJECT\\Dataset\\disease_symptom.p",'rb'))
print(clf)