from collections import defaultdict
import csv

__author__ = 'mikhail'
from gensim.parsing.preprocessing import STOPWORDS

nyt = open('/home/mikhail/Documents/research/NCTSLINK2015.csv')
nyt1 = open('/home/mikhail/Documents/research/sampleWithoutNCTArticles.csv') # check the structure of this file!

nyt_data = []
nyt_labels = []
csv_reader = csv.reader(nyt)

for line in csv_reader:
    nyt_labels.append((line[2]))
    nyt_data.append(line[1])
nyt.close()


nyt_data1 = []
nyt_labels1 = []
nyt_name1=[]
csv_reader1 = csv.reader(nyt1)

for line in csv_reader1:
    nyt_labels1.append((line[4]))
    nyt_data1.append(line[3])
    nyt_name1.append(line[2])
nyt1.close()




texts = [[word for word in document.lower().split() if word not in STOPWORDS]
         for document in nyt_data]

texts1 = [[word for word in document.lower().split() if word not in STOPWORDS]
         for document in nyt_data1]





frequency = defaultdict(int)
for text in texts:
    for token in text:
        #print token
        frequency[token] += 1


frequency1 = defaultdict(int)
for text in texts1:
    for token in text:
        #print token
        frequency1[token] += 1


from sets import Set
f1 = Set()


texts = [[token for token in text if frequency[token] > 1]
         for text in texts]

texts1 = [[token for token in text if frequency1[token] > 1]
         for text in texts1]

for text in texts:
    for token in text:
        f1.add(token)

for text in texts1:
    for token in text:
        f1.add(token)

v1=[]
v2=[]
for text in texts1:
    for t in texts:
        sum=0
        for f in f1:
            v1.append(frequency[f]/t.count())
            v2.append(frequency1[f]/text.count())
         for x,y in v1,v2:
            sum=sum+x*y log(x*y)



