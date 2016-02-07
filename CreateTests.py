from nltk import RegexpTokenizer

__author__ = 'mikhail'


import csv
from xml.etree import ElementTree




f = open('/home/mikhail/Documents/research/testSet2000.3.csv', 'wt')
countNCT=0
countSCRT=0
#meta-analysis
meta = RegexpTokenizer('meta-analys[i]{0,1}[e]{0,1}s')
rez = 0

try:
        writer = csv.writer(f)
        for i in range(1,3,1):
            document = ElementTree.parse( '/home/mikhail/Documents/research/DATA2015/medline/part-'+`i`+'.xml' )
            print i

            membership = document.getroot()
            users = document.find('document')

            for userid in document.findall('document'):
                rez = rez + 1
                body = ""
                encoded_user = ""
                encoded_str = ""
                for user in userid.getchildren():
                    if user.tag =='doc_id':
                        sch=0
                        encode_id = user.text.encode('ascii', 'ignore')
                    if user.tag =='date':
                        encode_date = user.text
                    if user.tag == 'title':
                        encoded_user = user.text.encode("utf8")
                    if user.tag == 'body':
                        encoded_str = user.text.encode("utf8")

                    body = encoded_user + " " + encoded_str

                    mbod = meta.tokenize(encoded_user)
                    body = body.replace("\'","").replace("\"","").replace("\\","").replace("\/","")

                if (len(mbod) > 0):
                        #print(body.replace("\'","").replace("\"","").replace("\\","").replace("\/",""),"meta-analysis")
                    writer.writerow(('\"',body,'\"', "meta-analysis",rez))
                else :
                    writer.writerow(('\"',body,'\"',"0",rez))

        print(countSCRT, countNCT)
        print(rez)


finally:
    f.close()

