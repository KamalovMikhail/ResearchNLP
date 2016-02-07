import urllib2

__author__ = 'mikhail'

from nltk import RegexpTokenizer

from xml.etree import ElementTree
import csv

nyt = open('/home/mikhail/Documents/research/errorNCT2Articles.csv') # check the structure of this file!
nyt_data = []
nyt_data1 = []
nyt_labels = []


csv_reader = csv.reader(nyt)

for line in csv_reader:
    nyt_data.append(line[2])
    nyt_data1.append(line[3])
    nyt_labels.append(line[4])
nyt.close()

f = open('/home/mikhail/Documents/research/hierarchical_classification/Inter_Observ/Interv.csv', 'a')

writer = csv.writer(f)
countNCT = 0
countSCRT = 0
ope = 0

try:
    for i in range(0,len(nyt_data),1):
        text = nyt_data[i]+" "+nyt_data1[i]
        try:
            observ = RegexpTokenizer("NCT[0-9]{8}")
            obs = observ.tokenize(nyt_labels[i])
            if len(obs) > 0:
                page = urllib2.urlopen('http://clinicaltrials.gov/show/'+obs[0]+'?resultsxml=true')
                document = ElementTree.parse(page)
                page_content = page.read()

                study_design = document.findtext('study_type')
                writer.writerow((text.replace("\'","").replace("\"","").replace("\\","").replace("\/",""),study_design))
                print("NORMAL"+" "+nyt_labels[i]+" "+study_design)
                countNCT +=1
        except Exception,e:
            print(nyt_labels[i])
    #error2_writer.writerow((encode_id,encode_date,encoded_user,encoded_str,NCT))
            print e
            pass




finally:
    f.close()
    print(countNCT)



