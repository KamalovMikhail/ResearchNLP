import csv

__author__ = 'mikhail'
"""
from xml.etree import ElementTree

import requests

page = requests.get('http://clinicaltrials.gov/show/NCT00423098?resultsxml=true')
document = ElementTree.parse(page)
users = document.find( 'intervention')
for user in users.getchildren():
    if user.tag == 'intervention_type':
        print(user.text)
"""
import urllib2
from xml.etree import ElementTree


nct =[]
nyt = open('/home/mikhail/Documents/research/sampleNCTArticles.csv')
csv_reader = csv.reader(nyt)


f = open('/home/mikhail/Documents/research/NCTSLINK2015.csv', 'wt')
try:

    writer = csv.writer(f)
    for line in csv_reader:
        try:
            page = urllib2.urlopen('http://clinicaltrials.gov/show/'+line[0]+'?resultsxml=true')
            print line[0]
            document = ElementTree.parse(page)
            page_content = page.read()
            users = document.iterfind('clinical_study')
            name =[]
            type=[]
           # for group in document.findall( 'intervention/intervention_name' ):
            #    print(group.text)
             #   name.append(group.text)

            group = document.find( 'intervention/intervention_type' )

            writer.writerow( (line[0],line[4],group.text) )
        except Exception,e:
            print e
            pass





finally:
    f.close()