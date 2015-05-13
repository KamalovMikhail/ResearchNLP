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

os.chdir('/home/mikhail/Documents/research')

nyt = open('/home/mikhail/Documents/research/sample.csv')  # check the structure of this file!
nyt1 = open('/home/mikhail/Documents/research/sampleTest.csv')
nyt_data = []
nyt_labels = []

nyt_data1 = []
nyt_labels1 = []
csv_reader = csv.reader(nyt)

for line in csv_reader:
    nyt_labels.append(line[1])
    nyt_data.append(line[0])

nyt.close()

trainset_size = int(round(len(nyt_data) * 0.75))  # i chose this threshold arbitrarily...to discuss
print 'The training set size for this classifier is ' + str(trainset_size) + '\n'

X_train = np.array([''.join(el) for el in nyt_data])
y_train = np.array([el for el in nyt_labels])

csv_reader1 = csv.reader(nyt1)
for line1 in csv_reader1:
    nyt_labels1.append(line1[1])
    nyt_data1.append(line1[0])

X_test = np.array([''.join(el1) for el1 in nyt_data1])
y_test = np.array([''.join(el1) for el1 in nyt_data1])

# print(X_train)

vectorizer = TfidfVectorizer(min_df=2,
                             ngram_range=(1, 2),
                             use_idf=True, smooth_idf=True,
                             stop_words='english',
                             strip_accents='unicode',
                             norm='l2')

test_string = unicode(nyt_data[0])

print "Example string: " + test_string
print "Preprocessed string: " + vectorizer.build_preprocessor()(test_string)
print "Tokenized string:" + str(vectorizer.build_tokenizer()(test_string))
print "N-gram data string:" + str(vectorizer.build_analyzer()(test_string))
print "\n"

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

nb_classifier = MultinomialNB().fit(X_train, y_train)

y_nb_predicted = nb_classifier.predict(X_test)

print "MODEL: Multinomial Naive Bayes\n"

print 'The precision for this classifier is ' + str(metrics.precision_score(y_test, y_nb_predicted))
print 'The recall for this classifier is ' + str(metrics.recall_score(y_test, y_nb_predicted))
print 'The f1 for this classifier is ' + str(metrics.f1_score(y_test, y_nb_predicted))
print 'The accuracy for this classifier is ' + str(metrics.accuracy_score(y_test, y_nb_predicted))

print '\nHere is the classification report:'

print '\nHere is the confusion matrix:'
print metrics.confusion_matrix(y_test, y_nb_predicted, labels=unique(nyt_labels))

N = 7
vocabulary = np.array([t for t, i in sorted(vectorizer.vocabulary_.iteritems(), key=itemgetter(1))])

for i, label in enumerate(nyt_labels):
    if i == 7:
        break
    topN = np.argsort(nb_classifier.coef_[i])[-N:]
    print "\nThe top %d most informative features for topic code %s: \n%s" % (N, label, " ".join(vocabulary[topN]))


from sklearn.svm import LinearSVC

svm_classifier = LinearSVC().fit(X_train, y_train)

y_svm_predicted = svm_classifier.predict(X_test)
print "MODEL: Linear SVC\n"

print 'The precision for this classifier is ' + str(metrics.precision_score(y_test, y_svm_predicted))
print 'The recall for this classifier is ' + str(metrics.recall_score(y_test, y_svm_predicted))
print 'The f1 for this classifier is ' + str(metrics.f1_score(y_test, y_svm_predicted))
print 'The accuracy for this classifier is ' + str(metrics.accuracy_score(y_test, y_svm_predicted))

print '\nHere is the classification report:'

l = range(len(y_test))
#f = open('/home/mikhail/Documents/research/ResultClassification.csv', 'wt')
#try:
 #   writer = csv.writer(f)
  #  for i in l:
   #     writer.writerow((y_test), (y_svm_predicted))
    #    t[i], y_svm_predicted[i]))
     #   finally:
      #  f.close()
print '\nHere is the confusion matrix:'
print metrics.confusion_matrix(y_test, y_svm_predicted, labels=unique(nyt_labels))




#What are the top N most predictive features per class?
N = 10
vocabulary = np.array([t for t, i in sorted(vectorizer.vocabulary_.iteritems(), key=itemgetter(1))])

for i, label in enumerate(nyt_labels):
    if i == 7:  # hack...
        break
    topN = np.argsort(svm_classifier.coef_[i])[-N:]
print "\nThe top %d most informative features for topic code %s: \n%s" % (N, label, " ".join(vocabulary[topN]))
#print topN



from sklearn.linear_model import LogisticRegression

maxent_classifier = LogisticRegression().fit(X_train, y_train)

y_maxent_predicted = maxent_classifier.predict(X_test)
print "MODEL: Maximum Entropy\n"

print 'The precision for this classifier is ' + str(metrics.precision_score(y_test, y_maxent_predicted))
print 'The recall for this classifier is ' + str(metrics.recall_score(y_test, y_maxent_predicted))
print 'The f1 for this classifier is ' + str(metrics.f1_score(y_test, y_maxent_predicted))
print 'The accuracy for this classifier is ' + str(metrics.accuracy_score(y_test, y_maxent_predicted))

print '\nHere is the classification report:'
#print classification_report(y_test, y_maxent_predicted)

#simple thing to do would be to up the n-grams to bigrams; try varying ngram_range from (1, 1) to (1, 2)
#we could also modify the vectorizer to stem or lemmatize
print '\nHere is the confusion matrix:'
print metrics.confusion_matrix(y_test, y_maxent_predicted, labels=unique(nyt_labels))



#What are the top N most predictive features per class?
N = 10
vocabulary = np.array([t for t, i in sorted(vectorizer.vocabulary_.iteritems(), key=itemgetter(1))])

for i, label in enumerate(nyt_labels):
    if i == 7:  # hack...
        break
    topN = np.argsort(maxent_classifier.coef_[i])[-N:]
print "\nThe top %d most informative features for topic code %s: \n%s" % (N, label, " ".join(vocabulary[topN]))
#print topN