from sympy.physics.quantum.circuitplot import np
from numpy import unique

__author__ = 'mikhail'
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from operator import itemgetter
from sklearn.metrics import classification_report
import csv
import os
import gensim
from sympy.physics.quantum.circuitplot import np
from numpy import unique

__author__ = 'mikhail'
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from operator import itemgetter
from sklearn.metrics import classification_report
import csv
import os
import gensim

import pandas as pd
df=pd.read_csv('/home/mikhail/Documents/ML/german.csv', sep=',',header=None)
df.values
print("pandas")
print(df[0:19])

data = np.genfromtxt('/home/mikhail/Documents/ML/german.csv', dtype=int, delimiter=',', names=True)



nyt = open('/home/mikhail/Documents/ML/german.csv')
train_data = []
train_labels = []
csv_reader = csv.reader(nyt)


for line in csv_reader:
    train_labels.append(float(line[20]))

    train_data.append((line[0:19]))



print("train")
print(train_data)
print("!!!!!")
print(train_labels)
nyt.close()






nyt1 = open('/home/mikhail/Documents/ML/german1.csv')
test_data = []
test_labels = []
csv_reader = csv.reader(nyt1)

for line in csv_reader:
    test_labels.append(float(line[20]))
    test_data.append((line[0:19]))

nyt1.close()

print(test_data)









from sklearn.svm import LinearSVC

svm_classifier = LinearSVC().fit(train_data, train_labels)

y_svm_predicted = svm_classifier.predict(test_data)


#print 'The precision  ' + str(metrics.precision_score(test_labels, y_svm_predicted))
#print 'The recall ' + str(metrics.recall_score(test_labels, y_svm_predicted))
#print 'The f1  ' + str(metrics.f1_score(test_labels, y_svm_predicted))
#print 'The accuracy' + str(metrics.accuracy_score(test_labels, y_svm_predicted))
