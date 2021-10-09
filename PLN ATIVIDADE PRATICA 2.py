import nltk

def gender_features(word):
    return {'last_letter' : word[-1]}

from nltk.corpus import names
lblnames = ([(name,'M') for name in names.words('male.txt')] +
         [(name,'F') for name in names.words('female.txt')])

import random
random.shuffle(lblnames)

print("Masculino:", len(names.words('male.txt')), "Feminino:", len(names.words('female.txt')))
print(len(names.words("male.txt"))/len(names.words('female.txt')))

featuresets = [(gender_features(n),gender) for (n,gender) in lblnames]

train_size = int(0.9 * len(featuresets))
train_set,test_set = featuresets[:train_size], featuresets[train_size:]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print("Accuracy:", '{:04.2f}'.format(nltk.classify.accuracy(classifier,test_set)))

matriz = [tag for (feat,tag) in test_set]

test=[]
for(feat,tag) in test_set:
    test += classifier.classify(feat)

mat = nltk.ConfusionMatrix(matriz, test)
print(mat.pretty_format(sort_by_count=True,truncate=9))