from Cheetah.Parser import end
from numpy import unique
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
import nltk

__author__ = 'mikhail'


from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from operator import itemgetter
from sklearn.metrics import classification_report
from sympy.physics.quantum.circuitplot import np
import csv
import os


nyt = open('/home/mikhail/Documents/research/NCTSLINK2015.csv')
nyt_data = []
nyt_labels = []
csv_reader = csv.reader(nyt)

for line in csv_reader:
    nyt_labels.append((line[2]))
    nyt_data.append(line[1])

nyt.close()





trainset_size = int(round(len(nyt_data)*100))
print 'The training set size for this classifier is ' + str(trainset_size) + '\n'

X_train = np.array([''.join(el) for el in nyt_data[0:trainset_size]])
y_train = np.array([el for el in nyt_labels[0:trainset_size]])



#print(X_train)

vectorizer = TfidfVectorizer(min_df=3,

 use_idf=True,
 smooth_idf=True,
max_df=0.5, stop_words='english',
 strip_accents='unicode'
 )



X_train = vectorizer.fit_transform(X_train)

from  gensim import corpora, models, similarities
from gensim.parsing.preprocessing import STOPWORDS


from sets import Set
f1 = Set()

listChar = [",",".",";",":","!","?","-","_"]
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

for l in nyt_data:
    tokens = nltk.word_tokenize(l.decode('utf8', 'ignore'))
    for token in tokens:
        if token not in STOPWORDS:
            if token not in listChar:
                to=(stemmer.stem(token))
                f1.add(to)








for f in f1:
    print f











sentence = """At eight running? on and, Thursday morning  Arthur didn't feel very good."""


stemToken = []







"""
lsa = TruncatedSVD(n_components=1250)
X_reduced = lsa.fit_transform(X_train)
"""




km = KMeans(n_clusters=9, init='k-means++', max_iter=200, n_init=1)
y=km.fit(X_train)


print km


clusters = km.labels_.tolist()

Result= []
"""
agl = AgglomerativeClustering(n_clusters=9,affinity='euclidean', linkage='ward')
rez = agl.fit(X_reduced)
clusters = agl.labels_.tolist()
"""


drugs = [0,0,0,0,0,0,0,0,0]

device = [0,0,0,0,0,0,0,0,0]
diet = [0,0,0,0,0,0,0,0,0]
proced = [0,0,0,0,0,0,0,0,0]
other = [0,0,0,0,0,0,0,0,0]
behav  = [0,0,0,0,0,0,0,0,0]
bio = [0,0,0,0,0,0,0,0,0]
gen = [0,0,0,0,0,0,0,0,0]
radiation = [0,0,0,0,0,0,0,0,0]





for i in range(0,9,1):
    for c in range(1,3535,1):
        if clusters[c]==i:
            if nyt_labels[c]== 'Drug':
                drugs[i] = drugs[i]+ 1
            if nyt_labels[c]== 'Device':
                device[i] = device[i]+ 1
            if nyt_labels[c]== 'Dietary Supplement':
                diet[i] = diet[i]+ 1
            if nyt_labels[c]== 'Procedure':
                proced[i] = proced[i]+ 1
            if nyt_labels[c]== 'Other':
                other[i] = other[i]+ 1
            if nyt_labels[c]== 'Behavioral':
                behav[i] = behav[i]+ 1
            if nyt_labels[c]== 'Biological':
                bio[i] = bio[i]+ 1
            if nyt_labels[c]== 'Genetic':
                gen[i] = gen[i]+ 1
            if nyt_labels[c]== 'Radiation':
                radiation[i] = radiation[i]+ 1

           # print nyt_labels[c], clusters[c]
import matplotlib.pyplot as plt


xbins = []

x= drugs
for i in range(0,9,1):
    xbins.append(i)


plt.bar(xbins, x, color='#2200CC')


plt.bar(xbins, behav, color='#D9007E')


plt.bar(xbins, diet, color='#FF6600' )

plt.bar(xbins, device, color='#FFCC00' )

plt.bar(xbins, bio, color='#ACE600' )
plt.bar(xbins, gen, color='#0099CC' )

plt.bar(xbins, radiation, color='#8900CC' )
plt.bar(xbins, other, color='#FF0000' )
plt.bar(xbins, proced, color='#FF9900' )

plt.show()



"""

import numpy as np
import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt


a = np.array([[0.1,   2.5],
              [1.5,   .4 ],
              [0.3,   1  ],
              [1  ,   .8 ],
              [0.5,   0  ],
              [0  ,   0.5],
              [0.5,   0.5],
              [2.7,   2  ],
              [2.2,   3.1],
              [3  ,   2  ],
              [3.2,   1.3]])

fig, axes23 = plt.subplots(2, 3)

for method, axes in zip(['single', 'complete'], axes23):
    z = hac.linkage(X_reduced, method=method)

    # Plotting
    axes[0].plot(range(1, len(z)+1), z[::-1, 2])
    knee = np.diff(z[::-1, 2], 2)
    axes[0].plot(range(2, len(z)), knee)

    num_clust1 = 10
    knee[knee.argmax()] = 0
    num_clust2 = 10

    axes[0].text(num_clust1, z[::-1, 2][num_clust1-1], 'possible\n<- knee point')

    part1 = hac.fcluster(z, 9, 'maxclust')
    part2 = hac.fcluster(z, 9, 'maxclust')

    clr = ['#2200CC' ,'#D9007E' ,'#FF6600' ,'#FFCC00' ,'#ACE600' ,'#0099CC' ,
    '#8900CC' ,'#FF0000' ,'#FF9900' ,'#FFFF00' ,'#00CC01' ,'#0055CC']

    for part, ax in zip([part1, part2], axes[1:]):
        for cluster in set(part):
            ax.scatter(X_reduced[part == cluster, 0], X_reduced[part == cluster, 1],
                       color=clr[cluster])

    m = '\n(method: {})'.format(method)
    plt.setp(axes[0], title='Screeplot{}'.format(m), xlabel='partition',
             ylabel='{}\ncluster distance'.format(m))
    plt.setp(axes[1], title='{} Clusters'.format(9))
    plt.setp(axes[2], title='{} Clusters'.format(9))

plt.tight_layout()
plt.show()




MatrixTF =[]

X_reduced.shape
for xt in X_reduced:
    print xt

"""
