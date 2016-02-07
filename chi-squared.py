import csv
from sklearn.feature_extraction.text import TfidfVectorizer

__author__ = 'mikhail'


from sklearn.feature_selection import SelectKBest, chi2

from sympy.physics.quantum.circuitplot import np


nyt = open('/home/mikhail/Documents/research/NCTSLINK2015.csv')
nyt_data = []
nyt_labels = []
csv_reader = csv.reader(nyt)
train = open('/home/mikhail/Documents/research/Header.arff', 'w')





for line in csv_reader:
    nyt_labels.append(line[2])
    nyt_data.append(line[1].replace("'"," "))

nyt.close()





trainset_size = int(round(len(nyt_data)*100))
print 'The training set size for this classifier is ' + str(trainset_size) + '\n'

X_train = np.array([''.join(el) for el in nyt_data[0:trainset_size]])
y_train = np.array([el for el in nyt_labels[0:trainset_size]])





vectorizer = TfidfVectorizer(min_df=3,

 use_idf=True,
 smooth_idf=True,
max_df=0.5, stop_words='english',
 strip_accents='unicode'
 )



X_train = vectorizer.fit_transform(X_train)
a = vectorizer.get_feature_names()

train.write("@RELATION workfile"+"\n")
train.write("@ATTRIBUTE "+"startproniyqwe"+" REAL"+"\n")

for x in a:

    train.write("@ATTRIBUTE "+x.encode('ascii', 'ignore')+"qwe"+" REAL"+"\n")
    print a

train.write("@ATTRIBUTE class {Other,Device,Dietary-Supplement,Biological,Drug,Radiation,Behavioral,Genetic,Procedure}"+"\n")

train.write("@DATA")

x = X_train.toarray()
for i in range(0,len(x)):
    s="0.0"
    for a in x[i]:
        s= s+','+(str(a))
    train.write(s+","+y_train[i]+"\n")





"""
print("Extracting %d best features by a chi-squared test")

ch2 = SelectKBest(chi2, k='all')
X_train = ch2.fit_transform(X_train,y_train)
X_test = ch2.transform(X_train)

print (X_test)
print()
"""