import sys
import time
import nltk
import os
import re
import numpy
from collections import Counter
from nltk.util import ngrams
from collections import defaultdict as ddict

start_program = time.time()

regex = r"[-'a-zA-ZÀ-ÖØ-öø-ÿ]+|[!?:;.,]"
#negationRe = r"not|no|\w*n't"

print("Digite o valor desejado para n-grams (Máximo 3):")
n = int(input())
if n > 3 or n < 1:
    sys.exit("Entrada inválida")

print("Deseja utilizar feature selection? (S/N)")

#addNeg = "n"
useFeatureSelection = input().lower()
#if(useFeatureSelection == "s"):
    #print("Deseja adicionar a negação às features? (S/N)")
    #addNeg = input().lower()

print("Deseja carregar as features dos arquivos? (S/N)")
reload = input().lower()

def getFeatures(docClass, dir, path = ""):
    if(os.path.exists(path) and path != "" and reload != "n"):
        return numpy.load(path, allow_pickle=True).tolist()
    else:
        feats = list(create_token_list(dir, docClass))
        if path != "":
            numpy.save(path, feats)
        return feats


def nGramAndNegate(document):
    features = Counter()
    punctuation = "?.,!:;"
    words = nltk.tokenize.regexp_tokenize(document.lower(), regex)
    inNegation = False
    oneBefore = None
    twoBefore = None
    for word in words:
        if any([word.find(p) != -1 for p in punctuation]):
            inNegation  = False
        else:
            if inNegation:
                actual = "not_" + word
            else:
                actual = word

            features[actual] += 1
            #if addNeg == "s":
            #    features["not_"+actual] += 1                      #Demora muito sem feature selection
            if oneBefore and n >= 2:
                bigram = oneBefore + " " + actual
                features[bigram] += 1
            #    if addNeg == "s":
            #        features["not_"+bigram] += 1                   #Demora muito sem feature selection
                if twoBefore and n >= 3:
                    trigram = twoBefore + " " + bigram
                    features[trigram] += 1
            #        if addNeg == "s":
            #            features["not_"+trigram] += 1              #Demora muito sem feature selection
                twoBefore = oneBefore
            oneBefore = actual

            #if "n't" in word or word == "not" or word == "no":
            if any(neg in ["not", "n't", "no"] for neg in word):
                inNegation  = not inNegation 

    return features

def feature_selection(features):
    return { k: v for k, v in features.items() if v > 1 }

def create_token_list(directory, docClass):
    for document in os.listdir(directory):
        documentText = open(directory+document, encoding = "utf8").read().lower()
        yield (nGramAndNegate(documentText), docClass) if useFeatureSelection == "n" else (feature_selection(nGramAndNegate(documentText)), docClass)

start_read = time.time()
print("Início da leitura, tokenização, n-gramas e negation handling.")

posTrainDir = "./IMDB_dataset/train/pos/"
posTrainDirDumpPath = "posTrainFeatures{}{}.npy".format(n, useFeatureSelection)
posTrainFeatures = getFeatures('pos', posTrainDir, posTrainDirDumpPath)

posTrainSize = len(posTrainFeatures)

negTrainDir = "./IMDB_dataset/train/neg/"
negTrainDirDumpPath = "negTrainFeatures{}{}.npy".format(n, useFeatureSelection)
negTrainFeatures = getFeatures('neg', negTrainDir, negTrainDirDumpPath)
negTrainSize =  len(negTrainFeatures)

posTestDir = "./IMDB_dataset/test/pos/"
posTestDirDumpPath = "posTestFeatures{}{}.npy".format(n, useFeatureSelection)
posTestFeatures = getFeatures('pos', posTestDir, posTestDirDumpPath)
posTestSize = len(posTestFeatures)

negTestDir = "./IMDB_dataset/test/neg/"
negTestDirDumpPath = "negTestFeatures{}{}.npy".format(n, useFeatureSelection)
negTestFeatures = getFeatures('neg', negTestDir, negTestDirDumpPath)
negTestSize =  len(negTestFeatures)

print("Fim da leitura dos arquivos.")
print("--- %.2f segundos ---" % (time.time() - start_read))

print("Textos de treino com avaliação positiva:")
print("Total: {}".format(posTrainSize))

print("Textos de treino com avaliação negativa:")
print("Total: {}".format(negTrainSize))

print("Textos de teste com avaliação positiva:")
print("Total: {}".format(posTestSize))

print("Textos de teste com avaliação negativa:")
print("Total: {}".format(negTestSize))

trainFeaturesets = posTrainFeatures + negTrainFeatures
testeFeaturesets = posTestFeatures + negTestFeatures
train_set, test_set = trainFeaturesets, testeFeaturesets

start_train = time.time()
print("Início do treinamento.")
classifier = nltk.NaiveBayesClassifier.train(train_set)

print("Fim do treinamento.")
print("--- %.2f segundos ---" % (time.time() - start_train))

print("Início dos testes.")
start_test = time.time()
posAccuracy = nltk.classify.accuracy(classifier, [(key, value) for key, value in test_set if value == 'pos'])
negAccuracy = nltk.classify.accuracy(classifier, [(key, value) for key, value in test_set if value == 'neg'])
totalAccuracy = (posAccuracy+negAccuracy)/2
print('Acurácia da classe positiva: %.2f%%' % (posAccuracy*100))
print('Acurácia da classe negativa: %.2f%%' % (negAccuracy*100))
print('Acurácia: %.2f%%' % (totalAccuracy*100))
print("Fim dos testes.")
print("--- %.2f segundos ---" % (time.time() - start_test))

print("--- Tempo total: %.2f segundos ---" % (time.time() - start_program))

avaliation = ''
print('Digite sua avaliação (0 para sair)\n')
while avaliation != '0':
    avaliation = input()
    if avaliation != '0':
        print('Sua avaliação é ' + classifier.classify(nGramAndNegate(avaliation)) +'\n')