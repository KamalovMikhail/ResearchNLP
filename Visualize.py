__author__ = 'mikhail'

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from os import listdir
from os.path import isfile, join
import glob

gl = glob.glob('/home/mikhail/Documents/research/FileSet/*.txt')

onlyfiles = [ f for f in listdir('/home/mikhail/Documents/research/FileSet/') if isfile(join('/home/mikhail/Documents/research/FileSet/',f)) ]

filenames = ['/home/mikhail/Documents/research/FileSet/NCT00006392_Drug.txt',
             '/home/mikhail/Documents/research/FileSet/NCT00000470_Procedure.txt',
             '/home/mikhail/Documents/research/FileSet/NCT00021255_Drug.txt',
             '/home/mikhail/Documents/research/FileSet/NCT00029146_Procedure.txt',
             '/home/mikhail/Documents/research/FileSet/NCT00031460_Drug.txt']

vectorizer = CountVectorizer(input='filename')
dtm = vectorizer.fit_transform(gl)
vocab = vectorizer.get_feature_names()

type(dtm)

dtm = dtm.toarray()
vocab = np.array(vocab)

house_idx = list(vocab).index('randomly')
print(dtm[1, house_idx])

n, _ = dtm.shape
dist = np.zeros((n, n))
for i in range(n):
         for j in range(n):
             x, y = dtm[i, :], dtm[j, :]
             dist[i, j] = np.sqrt(np.sum((x - y)**2))
from sklearn.metrics.pairwise import euclidean_distances
dist = euclidean_distances(dtm)
print(np.round(dist, 1))

import os
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)
xs, ys = pos[:, 0], pos[:, 1]
names = [os.path.basename(fn).replace('.txt', '') for fn in gl]
for x, y, name in zip(xs, ys, names):
         color = 'red' if "Drug" in name else 'skyblue'
         plt.scatter(x, y, c=color)
         plt.text(x, y, name)
plt.show()


mds = MDS(n_components=3, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])

for x, y, z, s in zip(pos[:, 0], pos[:, 1], pos[:, 2], names):
         ax.text(x, y, z, s)
plt.show()


from scipy.cluster.hierarchy import ward, dendrogram

linkage_matrix = ward(dist)
dendrogram(linkage_matrix, orientation="right", labels=names);
plt.tight_layout()
plt.show()