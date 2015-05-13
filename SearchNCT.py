import csv
import glob
from os import listdir
from xml.etree import ElementTree
from os.path import isfile, join

__author__ = 'mikhail'


##line = "Japanese encephalitis is associated with high rates of mortality and disabling sequelae. To date, no specific antiviral has proven to be of benefit for this condition. We attempted to determine the efficacy of oral ribavirin treatment for reducing early mortality among children with Japanese encephalitis in Uttar Pradesh, India.Children (age, 6 months to 15 years) who had been hospitalized with acute febrile encephalopathy (a &lt; or =2-week history of fever plus altered sensorium) were tested for the presence of immunoglobulin M antibodies to Japanese encephalitis virus with commercial immunoglobulin M capture enzyme-linked immunosorbent assay. Children with positive results were randomized to receive either ribavirin (10 mg/kg per day in 4 divided doses for 7 days) or placebo syrup through nasogastric tube or by mouth. The primary outcome was early mortality; secondary outcome measures were early (at hospital discharge; normal or nearly normal, independent functioning, dependent, vegetative state, or death) outcome, time to resolution of fever, time to resumption of oral feeding, duration of hospitalization, and late outcome (&gt; or =3 months after hospital discharge). The study was double-blind, and analysis was by intention to treat.A total of 153 patients were enrolled during a 3-year period; 70 patients received ribavirin, and 83 received placebo. There was no statistically significant difference between the 2 groups in the early mortality rate: 19 (27.1%) of 70 ribavirin recipients and 21 (25.3%) of 83 placebo recipients died (odds ratio, 1.10; 95% confidence interval, 0.5-2.4). No statistically significant differences in secondary outcome measures were found.For the dosage schedule used in our study, oral ribavirin has no effect in reducing early mortality associated with Japanese encephalitis.ClinicalTrials.gov identifier: NCT00216268"
#words = line.split()






#for word in words:
 #   if (word[0:3]=="NCT") & (len(word)==11):
  #      print word
line = 'sdfsdfd'+`1`+''
print line


gl = glob.glob('/home/mikhail/Desktop/medline/*.xml')
print gl
nyt = open('/home/mikhail/Documents/research/sampleNCTArticles.csv') # check the structure of this file!
nyt_data = []

csv_reader = csv.reader(nyt)
sch=0
for line in csv_reader:
    nyt_data.append(line[1])
nyt.close()


#onlyfiles = [ f for f in listdir('/home/mikhail/Desktop/medline/') if isfile(join('/home/mikhail/Desktop/medline/',f)) ]
f = open('/home/mikhail/Documents/research/sampleWithoutNCTArticles.csv', 'wt')
try:

        writer = csv.writer(f)
        for i in range(1,406,1):
            document = ElementTree.parse( '/home/mikhail/Documents/research/medline/part-'+`i`+'.xml' )
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
                        #print user.text
                        encoded_str = user.text.encode("utf8")
                    for id in nyt_data:
                        if encode_id==id:
                            sch=1
            if sch==0:
                writer.writerow((encode_id,encode_date,encoded_user,encoded_str,"undefined"))

finally:
    f.close()
