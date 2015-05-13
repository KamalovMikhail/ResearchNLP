import csv

__author__ = 'mikhail'


nyt_labels =[]
nyt_data =[]
nyt = open('/home/mikhail/Documents/research/sampleNCTArticles.csv')
csv_reader = csv.reader(nyt)


f = open('/home/mikhail/Documents/research/LINKNCT.csv', 'wt')

nyt1 = open('/home/mikhail/Documents/research/NCTScrap.csv')
csv_reader1 = csv.reader(nyt1)

try:

    writer = csv.writer(f)
    for line1 in csv_reader1:

        print line[0]+"    1"
        ind =0
        for line in csv_reader:
            if line1[0]==line[0]:
                ind=1
                l1 = line1[2]
        if ind==1:
            writer.writerow( (line[0],line[4],l1) )
finally:
    f.close()




