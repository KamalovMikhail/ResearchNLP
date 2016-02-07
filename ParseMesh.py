from xml.etree import ElementTree

__author__ = 'mikhail'
from nltk.tokenize import RegexpTokenizer


qwe = RegexpTokenizer("non[-]{0,17}[ ]{0,1}randomized")
t = "Allocation: non-randomized, Endpoint Classification: Safety/Efficacy Study, Intervention Model: single Group Assignment, Masking: open Label, Primary Purpose: Treatment"

ar = qwe.tokenize(t)

print(len(ar))

