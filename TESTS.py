from _elementtree import ElementTree
import csv
import re
import urllib2
from gensim.parsing import STOPWORDS
from nltk import RegexpTokenizer

__author__ = 'mikhail'


tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
t =tokenizer.tokenize('Eighty-seven 19 miles to go, yet.  Onward!')
print(t)



import lda



