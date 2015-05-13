import urllib2
from xml.etree import ElementTree
from nltk import SnowballStemmer
from nltk.corpus import stopwords

__author__ = 'mikhail'
import csv
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
nyt1 = open('/home/mikhail/Documents/research/NCTSLINK2015.csv')
nyt_data1 = []
nyt_labels1 = []
st = []
st.append(",")
st.append(".")
st.append("(")
st.append(")")
words = []

wordsKeyString = "analysis, millimeter, p=d.ddd, ddddd, relapse, standardis, may, rapid, month, -d.dd, yang, aim, uncertain, time, stent, illness, upper, thi, dos, statistical, cardiovascular, study, given, stroke, myocardial, respective, control, clinical, investigate, syndrome, combination, among, relative, randomized, subject, earlier-art, achiev, significant, however, dd.d, disorder, vitamin, progression, [ci], death, dd-dd, includ, aids-defin, care, active, lung, infarction, common, apixaban, suggest, chronic, tuberculosi, state, scale, exacerbation, can, cd, symptom, snu, rat, years, [d%], case, three-month, culture-bas, one, dd%, known, dai, number, remain, sham, telaprevir, three, outcome, therapy, v, assign, pta, ischemic, recommend, weeks, patient, failure, treatment, maintenance, procedure, versu, contrast-induc, -d.d, undergo, infection"
wordKey = wordsKeyString.split(", ");

file = open('/home/mikhail/Desktop/CRF++-0.58/example/chunking/rez1NCT.data', 'w')
csv_reader1 = csv.reader(nyt1)
stemmer = SnowballStemmer("english")
for line1 in csv_reader1:
    nyt_labels1.append(line1[1])
    drug =[]
    if line1[2]=="Drug":
        try:
            page = urllib2.urlopen('http://clinicaltrials.gov/show/'+line1[0]+'?resultsxml=true')
            document = ElementTree.parse(page)
            for group in document.findall( 'intervention/intervention_name' ):
                print(group.text)
                drug.append(group.text)
            encoded_str = line1[1].decode('ascii', 'ignore')
            result = pos_tag(word_tokenize(encoded_str))
            for s in result:
                ind =0
                n = s[0].encode('ascii', 'ignore')
                for d in drug:
                    id = word_tokenize(d)
                    # iterate over word_list


                    print d
                    for i in id:
                        print i
                        if (i in stopwords.words('english')) | (i in st):
                            print "remove  "+i
                        else:
                            if (i.lower() == stemmer.stem(n.lower())) | (i.lower()==n.lower()):
                                ind=1
                if ind==1:
                    file.writelines(n+" "+s[1]+" "+"B-Dr"+"\n")
                    print(n+" "+s[1]+" "+"B-Dr")
                else:
                    file.writelines(n+" "+s[1]+" "+"O"+"\n")
                    print(n+" "+s[1]+" "+"O")

    #nyt_data1.append(line1[0])
        except Exception,e:
           print e
           pass
file.close()


print nyt_labels1.count("Drug")





"""
for line in nyt_data1:
    a="0"
    words = line.split()
    for word in wordKey:
        for newWord in words:
            if newWord == word.replace(" ",""):
                a= "drug"
            else:
                a="0"


    if(a=="drug"):
        i.append(j)
    j=j+1

print(i.__sizeof__())
for k in i:
    print(nyt_labels1[k])


"""


    #print word.replace(" ","")





