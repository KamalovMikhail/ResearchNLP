from sympy import true

__author__ = 'mikhail'
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

from sklearn.svm import LinearSVC



os.chdir('/home/mikhail/Documents/research')

train = open('/home/mikhail/Documents/research/hierarchical_classification/TRAIN1R.csv')  # check the structure of this file!
test = open('/home/mikhail/Documents/research/hierarchical_classification/TEST1.csv')
train_nyt_data = []
train_nyt_labels = []

test_nyt_data1 = []
test_nyt_labels1 = []

csv_train = csv.reader(train)

for line in csv_train:
    train_nyt_data.append(line[0])
    train_nyt_labels.append(line[1])

train.close()

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
ytrain = lb.fit_transform(train_nyt_labels)



csv_test = csv.reader(test)

for line1 in csv_test:
    test_nyt_data1.append(line1[0])
    test_nyt_labels1.append(line1[1])

test.close()

y_test = np.array([''.join(el1) for el1 in test_nyt_labels1])


# print(X_train)

vectorizer = TfidfVectorizer(min_df=1, max_df = 1,
                             use_idf=True, smooth_idf=True,
                             stop_words='english',
                             strip_accents='unicode',
                             norm='l2')


X_train = vectorizer.fit_transform(train_nyt_data)
X_test = vectorizer.transform(test_nyt_data1)

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components = 500)
X_train = svd.fit_transform(X_train)
X_test = svd.transform(X_test)

svm_classifier = LinearSVC(class_weight={1:8.3, 0:10}).fit(X_train, ytrain)

proba = svm_classifier._predict_proba_lr(X_test)

print(proba)

for i in proba:
    print(i)

ytest = lb.transform(y_test)

y_svm_predicted = svm_classifier.predict(X_test)
TP0 = 0
TP1 = 0
for i in range(0, len(proba)) :
    if ytest[i] == 0 and  y_svm_predicted[i] == 0:
        TP0 =TP0 +1
    if ytest[i] == 1 and  y_svm_predicted[i] == 1:
        TP1 =TP1 +1
    if proba[i][0] - proba[i][1] < 0.1 and proba[i][0] - proba[i][1] > 0 :
        print(ytest[i], y_svm_predicted[i], 1, proba[i][0], proba[i][1])
    else :
        print(ytest[i], y_svm_predicted[i], proba[i][0], proba[i][1])



print(TP0 , TP1)
print(y_svm_predicted)
print(y_test)

print "MODEL: Linear SVC\n"

print 'The precision for this classifier is ' + str(metrics.precision_score(ytest, y_svm_predicted))
print 'The recall for this classifier is ' + str(metrics.recall_score(ytest, y_svm_predicted))
print 'The f1 for this classifier is ' + str(metrics.f1_score(ytest, y_svm_predicted))
print 'The accuracy for this classifier is ' + str(metrics.accuracy_score(ytest, y_svm_predicted))

print '\nHere is the classification report:'

#f = open('/home/mikhail/Documents/research/ResultClassification.csv', 'wt')
#try:
 #   writer = csv.writer(f)
  #  for i in l:
   #     writer.writerow((y_test), (y_svm_predicted))
    #    t[i], y_svm_predicted[i]))
     #   finally:
      #  f.close()