__author__ = 'mikhail'

from bs4 import BeautifulSoup
from BeautifulSoup import BeautifulStoneSoup, Tag, NavigableString

import urllib2
response = urllib2.urlopen('http://www.isrctn.com/ISRCTN15266438?q=ISRCTN15266438&filters=&sort=&offset=1&totalResults=1&page=1&pageSize=10&searchType=basic-search')
html_doc = response.read()

soup = BeautifulSoup(html_doc)


from itertools import takewhile

h3s = soup('h3')
for h in h3s:
    if h.text == 'Intervention type':
        f =h.next_sibling.next_sibling.text
        print(f.strip())

#print " ".join(soup.h3.text.split())

