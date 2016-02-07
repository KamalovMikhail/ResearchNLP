from nltk import RegexpTokenizer

__author__ = 'mikhail'

import csv

nyt = open('/home/mikhail/Documents/research/sampleWithNCT_ISRCTNArticles2.csv') # check the structure of this file!
nyt_data = []
nyt_data1 = []
nyt_labels = []


csv_reader = csv.reader(nyt)

for line in csv_reader:
    nyt_data.append(line[3])
    nyt_data1.append(line[2])
    nyt_labels.append(line[6])
nyt.close()
f = open('/home/mikhail/Documents/research/hierarchical_classification/JUSTEBM2015.csv', 'wt')
writer = csv.writer(f)
countNCT = 0
countSCRT = 0
ope = 0


for i in range(0,len(nyt_data),1):
    answer = []
    text = nyt_data1[i]+" "+nyt_data[i]

    randomized = RegexpTokenizer("randomized")
    nonrandomized = RegexpTokenizer("non[-]{0,17}[ ]{0,1}randomized")
    sblind = RegexpTokenizer("single[-]{0,17}[ ]{0,1}blind")
    dblind = RegexpTokenizer("double[-]{0,17}[ ]{0,1}blind")
    open = RegexpTokenizer("open")
    print(nyt_labels[i])
    ran = randomized.tokenize(nyt_labels[i])
    nr = nonrandomized.tokenize(nyt_labels[i])
    sb = sblind.tokenize(nyt_labels[i])
    db = dblind.tokenize(nyt_labels[i])
    op = open.tokenize(nyt_labels[i])

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
        countSCRT += 1
        answer.append(op[0].replace(" ","-"))


    if len(answer) == 2 :
        writer.writerow((text.replace("\'","").replace("\"","").replace("\\","").replace("\/",""),":    ".join(answer)))

print(countNCT)
print(countSCRT)


f.close()
