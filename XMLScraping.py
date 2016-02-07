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




# f = open('/home/mikhail/Documents/research/NCTSLINK2015.csv', 'wt')
# try:

#   writer = csv.writer(f)

try:
    page = urllib2.urlopen('http://clinicaltrials.gov/show/NCT01343589?resultsxml=true')
    document = ElementTree.parse(page)
    page_content = page.read()
    #users = document.iterfind('clinical_study')
    name = []
    users1 = document.findtext('study_design') #study_type
    print(users1)
    type = []
    # for group in document.findall( 'intervention/intervention_name' ):
    #    print(group.text)
    #   name.append(group.text)
    #print(users.text)
    group = document.find('intervention/intervention_type')
    print(group.text)
    # writer.writerow( (line[0],line[4],group.text) )
except Exception, e:
    print e
    pass





    #finally:
    #  f.close()