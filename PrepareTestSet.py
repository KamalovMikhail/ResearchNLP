from nltk import RegexpTokenizer

__author__ = 'mikhail'


from gensim.models import ldamodel, TfidfModel
from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer


__author__ = 'mikhail'
import csv
from  gensim import corpora, models, similarities
from gensim.parsing.preprocessing import STOPWORDS
from os import listdir
from os.path import isfile, join
import re
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
nyt = open('/home/mikhail/Documents/research/sampleWithNCT_ISRCTNArticles2.csv') # check the structure of this file!
nyt_data1 = []
nyt_data2 = []
nyt_labels = []
nyt_obs = []
st1 = ""
st2 = ""
st3 = ""
cop = ""
csv_reader = csv.reader(nyt)

for line in csv_reader:
    nyt_data1.append(line[2])
    nyt_data2.append(line[3])
    nyt_labels.append(line[6])
    nyt_obs.append(line[5])
nyt.close()
f = open('/home/mikhail/Documents/research/OnlyRandomized.csv', 'wt')
writer = csv.writer(f)
countRand =0
countNonRand = 0
print(len(nyt_data1))
for i in range(0,len(nyt_data1),1):
    #print(i)
    answer = []
    test = nyt_data1[i]+" "+nyt_data2[i]

    randomized = RegexpTokenizer("randomized")
    nonrandomized = RegexpTokenizer("non[-]{0,17}[ ]{0,1}randomized")

    nrt = nonrandomized.tokenize(nyt_labels[i])
    rat = randomized.tokenize(nyt_labels[i])

    if len(nrt) > 0 :
       countNonRand += 1
       answer.append(nrt[0].replace(" ","-").replace("nonrandomized","non-randomized"))
    elif len(rat) > 0 :
        countRand += 1
        answer.append(rat[0])

    if len(answer) > 0 :
        writer.writerow(("".join(test).replace("\'","").replace("\"",""),"".join(answer).replace(" ","-")))
        print("".join(answer).replace(" ","-"))
print(countNonRand)
print(countRand)
f.close()







