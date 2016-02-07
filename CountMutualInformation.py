import csv
import nltk

__author__ = 'mikhail'


from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from operator import itemgetter
from sklearn.metrics import classification_report
from sympy.physics.quantum.circuitplot import np
import csv

from nltk.tokenize import RegexpTokenizer
import os


nyt = open('/home/mikhail/Documents/research/TestEBM3.csv')
nyt_data = []
nyt_labels = []
csv_reader = csv.reader(nyt)

for line in csv_reader:
    nyt_labels.append((line[1]))
    nyt_data.append(line[0])

nyt.close()

trainset_size = int(round(len(nyt_data) * 100))

print 'The training set size for this classifier is ' + str(trainset_size) + '\n'

X_train = np.array([''.join(el) for el in nyt_data[0:trainset_size]])
y_train = np.array([el for el in nyt_labels[0:trainset_size]])






from  gensim import corpora, models, similarities
from gensim.parsing.preprocessing import STOPWORDS

from sets import Set

f1 = Set()

from nltk import PorterStemmer

stemmer = PorterStemmer()

tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

for l in nyt_data:
    tokens = tokenizer.tokenize(l.decode('utf8', 'ignore').lower())
    for token in tokens:
        if token not in STOPWORDS:
            stemm = stemmer.stem(token)
            if (len(stemm) > 3) & (len(stemm) < 10):
                #punctuation = re.compile(r'[-.?!,":;()|0-9]')
                f1.add(stemm)
                print(stemm)

import math

label = ['double-blind:randomized:','open:non-randomized:','open:randomized:','single-blind:randomized:','observational','double-blind:non-randomized:','single-blind:non-randomized:']

word = []
measure = []
f = open('/home/mikhail/Documents/research/MIStemEBM.csv', 'wt')

try:
    writer = csv.writer(f)
    for f in f1:


        for la in label:

            N10 = 1
            N11 = 1
            N01 = 1
            N00 = 1
            for l in range(0, len(nyt_data), 1):


                if (f in nyt_data[l].decode('utf8', 'ignore')) and (nyt_labels[l] == la):
                    N11 += 1
                if (f not in nyt_data[l].decode('utf8', 'ignore')) and (nyt_labels[l] == la):
                    N01 += 1
                if (f in nyt_data[l].decode('utf8', 'ignore') ) and (nyt_labels[l] != la):
                    N10 += 1
                if (f not in nyt_data[l].decode('utf8', 'ignore') ) and (nyt_labels[l] != la):
                    N00 += 1

            word.append(f)
            fe = (len(nyt_data) * N11) / float((N10 + N11) * (N11 + N01))
            print "param", len(nyt_data), N11, fe
            print fe
            ferst = N11 / float(len(nyt_data)) * math.log(fe, 2)
            print ferst

            se = (len(nyt_data) * N01) / float((N01 + N00) * (N11 + N01))
            print se
            second = N01 / float(len(nyt_data)) * math.log(se, 2)
            print second
            th = (len(nyt_data) * N10) / float((N11 + N10) * (N10 + N00))
            print th

            therd = N10 / float(len(nyt_data)) * math.log(th, 2)
            print therd
            fo = (len(nyt_data) * N00) / float((N01 + N00) * (N10 + N00))
            print(fo)
            print "forth", len(nyt_data), N00, fo
            forth = N00 / float(len(nyt_data)) * math.log(fo, 2)
            print forth
            measure.append(ferst + second + therd + forth)
            c = ferst + second + therd + forth
            try:
                writer.writerow((f, c, la))
            except Exception, e:
                print e
                pass
finally:
    f.close()