import csv
import os

__author__ = 'mikhail'

import gensim.models.word2vec


model = gensim.models.Word2Vec()

nyt1 = open('/home/mikhail/Documents/research/NCTSLINK2015.csv')
nyt_data1 = []

csv_reader = csv.reader(nyt1)

for line in csv_reader:
    #nyt_labels.append(line[1])
    nyt_data1.append(line[1].decode("ascii", "ignore"))

print(nyt_data1.__sizeof__())


model.build_vocab(nyt_data1[0:20000])
model.train(nyt_data1[20001:30000])

print model['drug']

#model[]

"""
class MySentences(object):
   def __init__(self, dirname):
        self.dirname = dirname
   def __iter__(self):
         for fname in os.listdir(self.dirname):
             for line in open(os.path.join(self.dirname, fname)):
                 yield line.split()

sentences = MySentences('/home/mikhail/Documents/research/FileSet/')
print sentences
model = gensim.models.Word2Vec(sentences)
print model

#model.most_similar(positive=['Selenium'], negative=['ITD'], topn=1)
"""