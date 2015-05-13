import csv

__author__ = 'mikhail'


from sklearn.cluster import KMeans

num_clusters = 50

km = KMeans(n_clusters=num_clusters)


from sklearn.externals import joblib

nyt = open('/home/mikhail/Documents/research/NCTSLINK2015.csv') # check the structure of this file!
nyt_data = []
nyt_labels = []
csv_reader = csv.reader(nyt)

for line in csv_reader:
    nyt_labels.append((line[2]))
    nyt_data.append(line[1])
nyt.close()

km = joblib.load(nyt_data)

clusters = km.labels_.tolist()