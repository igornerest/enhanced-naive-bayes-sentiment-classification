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
negationRe = r"not|no|\w*n't"

print("Digite o valor desejado para n-grams:")
n = int(input())


def readFile(directory, filename):
    return open(directory+'/'+filename, 'r', encoding='UTF-8').read().lower()

def readAndCountNgrams(directory, documentClass):
    return [getFeatures(readFile(directory, filename), documentClass) for filename in os.listdir(directory)]

def readAndTokenize(directory):
    return [nltk.tokenize.regexp_tokenize(open(directory+'/'+filename, 'r', encoding='UTF-8').read().lower(), regex) for filename in os.listdir(directory)]

def getFeatures(document, documentClass):
    return (Counter(ngrams(nltk.word_tokenize(document), n)), documentClass)

def makeNGrams(max_n, wordList):
    counts = dict()
    # Caso especial: tupla vazia (util para o metodo 'probability')
    # O valor eh igual ao numero de palavras

    # Conta os n-gramas para todos os tamanhos. Para trigramas: (), (    w), (w,w), (w,w,w) 
    n_range = range(1, max_n + 1)
    for i, word in enumerate(wordList):
        for n in n_range:
            if i+n <= len(wordList):
                stri = ""
                for s in wordList[i:i+n]:
                    stri += s + " "
                if str not in counts:
                    counts[stri] = 0
                counts[stri] += 1
    
    return counts

def negate_sequence(words): 
    negated = False 
    punctuationMark = ["!", "?", ":", ";", ".", ","]    
    ans = []
    
    for word in words:
        if word in punctuationMark:
            negated = False
        else:
            unigram = "not_" + word if negated else word
            ans += [unigram]

            if word == "not" or word == "n't":
                negated = not negated

    return ans

start_read = time.time()
print("Início da leitura dos arquivos.")
posTrainDir = "./IMDB_dataset/train/pos"
posTrainTokenList = [()]
for tokens in readAndTokenize(posTrainDir):
    posTrainTokenList +=[(makeNGrams(n, negate_sequence(tokens)), 'pos')]
posTrainSize = len(posTrainTokenList)

negTrainDir = "./IMDB_dataset/train/neg"
negTrainTokenList = [()] 
for tokens in readAndTokenize(negTrainDir):
    negTrainTokenList += [(makeNGrams(n, negate_sequence(tokens)), 'neg')]
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

trainFeaturesets = posTrainTokenList + negTrainTokenList
testeFeaturesets = posTestTokenList + negTestTokenList
train_set, test_set = trainFeaturesets, testeFeaturesets


print(train_set)

start_train = time.time()
print("Início do treinamento.")
classifier = nltk.NaiveBayesClassifier.train(train_set)
print("Fim do treinamento.")
print("--- %s seconds ---" % (time.time() - start_train))

print("Início dos testes.")
#start_test = time.time()
#print('Accuracy: ' + str(nltk.classify.accuracy(classifier, test_set)*100) + '%')
print("Fim dos testes.")
#print("--- %s seconds ---" % (time.time() - start_test))

#print("--- Tempo total: %s seconds ---" % (time.time() - start_program))

avaliation = ''
#print('Digite sua avaliação (0 para sair)\n')
#while avaliation != '0':
#    avaliation = input()
#    if avaliation != '0':
#        print('Sua avaliação é ' + classifier.classify(document_features(avaliation)) +'\n')


#classifier.show_most_informative_features(5)
