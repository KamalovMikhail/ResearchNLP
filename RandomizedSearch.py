import csv
import glob
from os import listdir
from xml.etree import ElementTree
from os.path import isfile, join

__author__ = 'mikhail'


from nltk.tokenize import RegexpTokenizer



f1 = open('/home/mikhail/Documents/research/hierarchical_classification/TestBlindRosenfeldShifmanConstraint.csv', 'wt')
f2 = open('/home/mikhail/Documents/research/hierarchical_classification/TestRandomRosenfeldShifmanConstraint.csv', 'wt')

countNCT = 0
countRT = 0
countDB = 0
countSB = 0
countOP = 0

try:


        writerBlind = csv.writer(f1)
        writerRandom = csv.writer(f2)
        for i in range(1,406,1):
            text = ""
            topics = ""
            title = ""
            document = ElementTree.parse( '/home/mikhail/Documents/research/DATA2015/medline/part-'+`i`+'.xml' )
            print i
            membership = document.getroot()
            users = document.find( 'document')

            for userid in document.findall( 'document'):
                answerR = []
                answerB = []
                for user in userid.getchildren():
                    if user.tag =='doc_id':
                        sch=0
                        encode_id = user.text.encode('ascii', 'ignore')
                    if user.tag =='date':
                        st = user.text
                    if user.tag == 'title':
                        title = user.text.encode("utf8")
                    if user.tag == 'body':
                        text = user.text.encode('ascii', 'ignore')
                    if user.tag == 'topics':
                        topics = user.text.encode('ascii', 'ignore')

                text = text + title

                randomized = RegexpTokenizer("[Rr]andomized")
                animals = RegexpTokenizer("Animals")
                nonrandomized = RegexpTokenizer("[Nn]on[-]{0,17}[ ]{0,1}[Rr]andomized")
                sblind = RegexpTokenizer("[Ss]ingle[-]{0,17}[ ]{0,1}[Bb]lind")
                dblind = RegexpTokenizer("[Dd]ouble[-]{0,17}[ ]{0,1}[Bb]lind")
                open = RegexpTokenizer("[Oo]pen[-]{0,17}[ ]{0,1}[Ll]abel")


                ant = animals.tokenize(topics)


                nrtext = nonrandomized.tokenize(text)
                ratext = randomized.tokenize(text)
                sbtext = sblind.tokenize(text)
                dbtext = dblind.tokenize(text)
                obtext = open.tokenize(text)

                nrtitle = nonrandomized.tokenize(title)
                ratitle = randomized.tokenize(title)
                sbtitle = sblind.tokenize(title)
                dbtitle = dblind.tokenize(title)
                obtitle = open.tokenize(title)






                if ( (len(nrtitle) > 0) and (len(nrtext) > 0) ) and (len(ant) == 0) :
                    countNCT+=1
                    print("nonrandomized")
                    answerR.append("Non-Randomized")
                elif ( (len(ratext) > 0) and (len(ratitle) > 0) ) and (len(ant) == 0):
                    countRT+=1
                    print("randomized")
                    answerR.append("Randomized")


                if ( (len(sbtitle) > 0) and (len(sbtext) > 0) ) and (len(ant) == 0):
                    countSB += 1
                    print("singleblind")
                    answerB.append("Single-Blind")
                elif ( (len(dbtitle) > 0) and (len(dbtext) > 0) ) and (len(ant) == 0) :
                    countDB += 1
                    print("doubleblind")
                    answerB.append("Double-Blind")
                elif ( (len(obtitle) > 0) and (len(obtext) > 0) ) and (len(ant) == 0) :
                    countOP += 1
                    print("openlabel")
                    answerB.append("Open-Label")




                if len(answerR) > 0 :
                    writerRandom.writerow((text.replace("\'","").replace("\"","").replace("\\","").replace("\/",""),":".join(answerR)))

                if len(answerB) > 0 :
                    writerBlind.writerow((text.replace("\'","").replace("\"","").replace("\\","").replace("\/",""),":".join(answerB)))




finally:
    print(countRT)
    print(countNCT)
    print(countSB)
    print(countDB)
    print(countOP)
    f1.close()
    f2.close()


