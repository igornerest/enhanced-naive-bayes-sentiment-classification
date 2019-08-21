import time
import nltk
import os
import re
import numpy
from collections import Counter

start_program = time.time()

regex = r"[-'a-zA-ZÀ-ÖØ-öø-ÿ]+"
negationRe = r"not|no|\w*n't"

def readAndTokenize(directory):
    return [nltk.tokenize.regexp_tokenize(open(directory+'/'+filename, 'r', encoding='UTF-8').read().lower(), regex) for filename in os.listdir(directory)]

def negation_handling_ngram(words):
    negated = False
    words_list = list()

    for word in words:
        if (re.fullmatch(negationRe, word)):
            negated = not negated
        elif (negated):
            word = "not_" + word
        words_list.append(word)

    return words_list

start_read = time.time()
print("Início da leitura dos arquivos.")
posTrainDir = "./IMDB_dataset/train/pos"
posTrainTokenList = readAndTokenize(posTrainDir)
posTrainSize = len(posTrainTokenList)

negTrainDir = "./IMDB_dataset/train/neg"
negTrainTokenList = readAndTokenize(negTrainDir)
negTranSize =  len(negTrainTokenList)

posTestDir = "./IMDB_dataset/test/pos"
posTestTokenList = readAndTokenize(posTestDir)
posTestSize = len(posTestTokenList)

negTestDir = "./IMDB_dataset/test/neg"
negTestTokenList = readAndTokenize(negTestDir)
negTestSize =  len(negTestTokenList)

print("Fim da leitura dos arquivos.")
print("--- %s seconds ---" % (time.time() - start_read))

print("Textos de treino com avaliação positiva")
print("Total: {}".format(posTrainSize))

print("Textos de treino com avaliação negativa:")
print("Total: {}".format(len(os.listdir(negTrainDir))))

print("Textos de treino com avaliação positiva")
print("Total: {}".format(posTrainSize))

print("Textos de treino com avaliação negativa:")
print("Total: {}".format(len(os.listdir(negTrainDir))))


trainDocuments = [(x, 'pos') for x in posTrainTokenList] + [(x, 'neg') for x in negTrainTokenList]
testDocuments = [(x, 'pos') for x in posTestTokenList] + [(x, 'neg') for x in negTestTokenList]

def document_features(document):
    dict = Counter()
    for word in document:
        dict[word] += 1
    return dict

trainFeaturesets = [(document_features(d), c) for (d,c) in trainDocuments]
testeFeaturesets = [(document_features(d), c) for (d,c) in testDocuments]

train_set, test_set = trainFeaturesets, testeFeaturesets

start_train = time.time()
print("Início do treinamento.")
classifier = nltk.NaiveBayesClassifier.train(train_set)
print("Fim do treinamento.")
print("--- %s seconds ---" % (time.time() - start_train))

print("Início dos testes.")
start_test = time.time()
print('Accuracy: ' + str(nltk.classify.accuracy(classifier, test_set)*100) + '%')
print("Fim dos testes.")
print("--- %s seconds ---" % (time.time() - start_test))

print("--- Tempo total: %s seconds ---" % (time.time() - start_program))

avaliation = '';
print('Digite sua avaliação (0 para sair)\n')
while avaliation != '0':
    avaliation = input()
    if avaliation != '0':
        print('Sua avaliação é ' + classifier.classify(document_features(avaliation)) +'\n')


#classifier.show_most_informative_features(5)