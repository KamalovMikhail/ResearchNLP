import csv
import glob
from os import listdir
from xml.etree import ElementTree
from os.path import isfile, join

__author__ = 'mikhail'


from bs4 import BeautifulSoup
from BeautifulSoup import BeautifulStoneSoup, Tag, NavigableString

from nltk.tokenize import RegexpTokenizer
import urllib2



f = open('/home/mikhail/Documents/research/sampleWithNCT_ISRCTNArticles2.csv', 'wt')
f1 = open('/home/mikhail/Documents/research/errorISRCTN2Articles.csv', 'wt')
f2 = open('/home/mikhail/Documents/research/errorNCT2Articles.csv', 'wt')

randomized = RegexpTokenizer("randomized")
nonrandomized = RegexpTokenizer("non[-]{0,17}[ ]{0,1}randomized")
sblind = RegexpTokenizer("single[-]{0,17}[ ]{0,1}blind")
dblind = RegexpTokenizer("double[-]{0,17}[ ]{0,1}blind")
open = RegexpTokenizer("open")

countNCT=0
countSCRT=0
try:

        writer = csv.writer(f)
        error_writer = csv.writer(f1)
        error2_writer = csv.writer(f2)
        for i in range(1,406,1):
            answer = []
            document = ElementTree.parse( '/home/mikhail/Documents/research/DATA2015/medline/part-'+`i`+'.xml' )
            print i
            membership = document.getroot()
            users = document.find( 'document')

            for userid in document.findall( 'document'):
                for user in userid.getchildren():
                    if user.tag =='doc_id':
                        sch=0
                        encode_id = user.text.encode('ascii', 'ignore')
                    if user.tag =='date':
                        encode_date = user.text
                    if user.tag == 'title':
                        encoded_user = user.text.encode("utf8")
                    if user.tag == 'body':
                        encoded_str = user.text.encode('ascii', 'ignore')
                        qwe = RegexpTokenizer('ISRCTN[0-9]{8}')
                        qwen = RegexpTokenizer('NCT[0-9]{8}')
                        tokens = qwe.tokenize(encoded_str)
                        tokens2 = qwen.tokenize(encoded_str)
                        NCT=""
                        ISRCTN=""
                        if len(tokens2)>0:
                            NCT = tokens2[0]

                        if len(tokens)>0:
                            ISRCTN=tokens[0]



                        print(NCT)
                        try:
                                page = urllib2.urlopen('http://clinicaltrials.gov/show/'+NCT+'?resultsxml=true')
                                document = ElementTree.parse(page)
                                page_content = page.read()
                                study_design = document.findtext('study_design')
                                primary_study_design= document.findtext('study_type')
                                group  = document.find('intervention/intervention_type')
                                countNCT=countNCT+1

                                ran = randomized.tokenize(study_design)
                                nr = nonrandomized.tokenize(study_design)
                                sb = sblind.tokenize(study_design)
                                db = dblind.tokenize(study_design)
                                op = open.tokenize(study_design)
                                if len(nr) > 0 :
                                    countNCT += 1
                                    answer.append(nr[0].replace(" ","-"))
                                elif len(ran) > 0:
                                    countSCRT += 1
                                    answer.append(ran[0].replace(" ","-"))
                                if len(db) > 0:
                                    countSCRT += 1
                                    answer.append(db[0].replace(" ","-"))
                                elif len(sb) > 0 :
                                    countNCT += 1
                                    answer.append(sb[0].replace(" ","-"))
                                elif len(op) > 0:
                                    answer.append(op[0].replace(" ","-"))

                                if len(answer) == 2 :
                                    writer.writerow((encode_id,encode_date,encoded_user,encoded_str.replace("\'","").replace("\"","").replace("\\","").replace("\/",""),NCT,primary_study_design,":    ".join(answer),group.text))

                        except Exception,e:

                                error2_writer.writerow((encode_id,encode_date,encoded_user,encoded_str,NCT))
                                print e
                                pass
                        if len(ISRCTN)==14:

                                print(ISRCTN)
                                try:
                                    response = urllib2.urlopen('http://www.isrctn.com/'+ISRCTN+'?q='+ISRCTN+'&filters=&sort=&offset=1&totalResults=1&page=1&pageSize=10&searchType=basic-search')
                                    html_doc = response.read()
                                    soup = BeautifulSoup(html_doc)
                                    h3s = soup('h3')
                                    for h in h3s:
                                        if h.text == 'Intervention type':
                                            subtype = h.next_sibling.next_sibling.text.strip()
                                            if h.text == 'Study design':
                                                study_design = h.next_sibling.next_sibling.text.strip()
                                        if h.text == 'Primary study design':
                                            primary_study_design =h.next_sibling.next_sibling.text.strip()

                                    ran = randomized.tokenize(study_design)
                                    nr = nonrandomized.tokenize(study_design)
                                    sb = sblind.tokenize(study_design)
                                    db = dblind.tokenize(study_design)
                                    op = open.tokenize(study_design)
                                    if len(nr) > 0 :
                                        countNCT += 1
                                        answer.append(nr[0].replace(" ","-"))
                                    elif len(ran) > 0:
                                        countSCRT += 1
                                        answer.append(ran[0].replace(" ","-"))
                                    if len(db) > 0:
                                        countSCRT += 1
                                        answer.append(db[0].replace(" ","-"))
                                    elif len(sb) > 0 :
                                        countNCT += 1
                                        answer.append(sb[0].replace(" ","-"))
                                    elif len(op) > 0:
                                        answer.append(op[0].replace(" ","-"))

                                    if len(answer) == 2 :
                                        writer.writerow((encode_id,encode_date,encoded_user,encoded_str.replace("\'","").replace("\"","").replace("\\","").replace("\/",""),ISRCTN,primary_study_design,":    ".join(answer),subtype))
                                    countSCRT = countSCRT+1

                                except Exception,e:
                                    
                                    error_writer.writerow((encode_id,encode_date,encoded_user,encoded_str,ISRCTN))
                                    print e
                                    pass
                        else:
                             qwe = RegexpTokenizer('[ISRCTN]{6}')
                             tokens =qwe.tokenize(encoded_str)
                             qwen = RegexpTokenizer('[NCT]{3}\w+')
                             tokens2 =qwen.tokenize(encoded_str)

                             if len(tokens) > 0:
                                 error_writer.writerow((encode_id,encode_date,encoded_user,encoded_str,tokens[0]))
                             elif len(tokens2) > 0:
                                 error2_writer.writerow((encode_id,encode_date,encoded_user,encoded_str,tokens2[0]))

        print(countSCRT,countNCT)





finally:
    f1.close()
    f.close()
    f2.close()
