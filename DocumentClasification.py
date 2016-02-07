from gensim.models import ldamodel, TfidfModel
from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer


__author__ = 'mikhail'
import csv
from  gensim import corpora, models, similarities
from gensim.parsing.preprocessing import STOPWORDS
from os import listdir
from os.path import isfile, join

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
nyt = open('/home/mikhail/Documents/research/sampleWithISRCTNArticles.csv') # check the structure of this file!
nyt_data = []
nyt_labels = []
csv_reader = csv.reader(nyt)

for line in csv_reader:
    nyt_labels.append((line[2]))
    nyt_data.append(line[3])
nyt.close()


stoplist = set('for a of the and to in'.split())

texts = [[word for word in document.lower().split() if word not in STOPWORDS]
         for document in nyt_data]


from collections import defaultdict







texts = [[token for token in text if 15>len(token)> 4]
         for text in texts]



#for f in f1:
    #print(f)



dictionary = corpora.Dictionary(texts)




dictionary.save('/home/mikhail/Documents/research/deerwester.dict')


corpus = [dictionary.doc2bow(text) for text in texts]

corpora.MmCorpus.serialize('/home/mikhail/Documents/deerwester.mm', corpus)

#lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=9)

#lsi.print_topics(num_topics=9, num_words=5)



lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=9)

lda.print_topics(num_topics=9, num_words=12)

#tfidf = TfidfModel(corpus)

#pca = TruncatedSVD(n_components=2)
#X_reduced = pca.fit_transform(tfidf)

#print(X_reduced)