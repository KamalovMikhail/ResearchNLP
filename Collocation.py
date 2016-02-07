import nltk

__author__ = 'mikhail'

from nltk.collocations import *
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim.parsing.preprocessing import STOPWORDS
import csv

nyt = open('/home/mikhail/Documents/research/TestEBM3.csv') # check the structure of this file!
nyt_data = []
nyt_labels = []
csv_reader = csv.reader(nyt)

for line in csv_reader:
    nyt_labels.append((line[1]))
    nyt_data.append(line[0].decode("ascii", "ignore"))
nyt.close()

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
p_stemmer = PorterStemmer()


for k in range(0,len(nyt_data)):
    texts = []
    raw = nyt_data[k].lower()
    qwe = RegexpTokenizer(r'\w+')
    tokens = qwe.tokenize(raw)
    stemmer = PorterStemmer()

    stopped_tokens = [i for i in tokens if not i in STOPWORDS]
    for token in stopped_tokens:
        stemm = stemmer.stem(token)
        if 10>len(stemm)>4:
            #print(token)
            texts.append(stemm)
    finder = BigramCollocationFinder.from_words(texts)
    print finder.nbest(bigram_measures.pmi, 10)
    print(nyt_labels[k])

    #scored = finder.score_ngrams(bigram_measures.raw_freq)

    #print(texts)




   # finder = BigramCollocationFinder.from_words(tokens)
    #bigram_measures = BigramAssocMeasures()
    #print finder.nbest(bigram_measures.pmi, 10)

    #student_t = {k:v for k,v in finder.score_ngrams(bigram_measures.student_t)}
    #print student_t[u'blood', u'glucose']


