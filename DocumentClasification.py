from gensim.models import ldamodel, TfidfModel
from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer


__author__ = 'mikhail'
import csv
from  gensim import corpora, models, similarities
from gensim.parsing.preprocessing import STOPWORDS
from os import listdir
from os.path import isfile, join

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
nyt = open('/home/mikhail/Documents/research/NCTSLINK2015.csv') # check the structure of this file!
nyt_data = []
nyt_labels = []
csv_reader = csv.reader(nyt)

for line in csv_reader:
    nyt_labels.append((line[2]))
    nyt_data.append(line[1])
nyt.close()

documents = ["The authors report the 3-year results of the Barrow Ruptured Aneurysm Trial (BRAT). The objective of this ongoing randomized trial is to compare the safety and efficacy of microsurgical clip occlusion and endovascular coil embolization for the treatment of acutely ruptured cerebral aneurysms and to compare functional outcomes based on clinical and angiographic data. The 1-year results have been previously reported.Two-hundred thirty-eight patients were assigned to clip occlusion and 233 to coil embolization. There were no anatomical exclusions. Crossovers were allowed based on the treating physician's determination, but primary outcome analysis was based on the initial assignment to treatment modality. Patient outcomes were assessed independently using the modified Rankin Scale (mRS). A poor outcome was defined as an mRS score>2. At 3 years' follow-up 349 patients who had actually undergone treatment were available for evaluation. Of the 170 patients who had been originally assigned to coiling, 64 (38%) crossed over to clipping, whereas 4 (2%) of 179 patients assigned to surgery crossed over to clipping.The risk of a poor outcome in patients assigned to clipping compared with those assigned to coiling (35.8% vs 30%) had decreased from that observed at 1 year and was no longer significant (OR 1.30, 95% CI 0.83-2.04, p=0.25). In addition, the degree of aneurysm obliteration (p=0.0001), rate of aneurysm recurrence (p=0.01), and rate of retreatment (p=0.01) were significantly better in the group treated with clipping compared with the group treated with coiling. When outcomes were analyzed based on aneurysm location (anterior circulation, n=339; posterior circulation, n=69), there was no significant difference in the outcomes of anterior circulation aneurysms between the 2 assigned groups across time points (at discharge, 6 months, 1 year, or 3 years after treatment). The outcomes of posterior circulation aneurysms were significantly better in the coil group than in the clip group after the 1st year of follow-up, and this difference persisted after 3 years of follow-up. However, while aneurysms in the anterior circulation were well matched in their anatomical location between the 2 treatment arms, this was not the case in the posterior circulation where, for example, 18 of 21 posterior inferior cerebellar artery aneurysms were in the clip group.Based on mRS scores at 3 years, the outcomes of all patients assigned to coil embolization showed a favorable 5.8% absolute difference compared with outcomes of those assigned to clip occlusion, although this difference did not reach statistical significance (p=0.25). Patients in the clip group had a significantly higher degree of aneurysm obliteration and a significantly lower rate of recurrence and retreatment. In post hoc analysis examining only anterior circulation aneurysms, no outcome difference between the 2 treatment cohorts was observed at any recorded time point. CLINICAL TRIAL REGISTRATION NO.: NCT01593267 ( ClinicalTrials.gov ).",
             "The study to Evaluate Patient OutComes, Safety, and Tolerability of Fingolimod (EPOC; NCT01216072) aimed to test the hypothesis that therapy change to oral Gilenya (Novartis AG, Stein, Switzerland) (fingolimod) improves patient-reported outcomes compared with standard-of-care disease-modifying therapy (DMT) in patients with relapsing multiple sclerosis; safety and tolerability were also assessed. This communication describes the study rationale and design.EPOC is a phase 4, open-label, multi-center study conducted in the US and Canada of patients with relapsing forms of multiple sclerosis who are candidates for therapy change. Therapy change eligibility was determined by the treating physician (US patients) or required an inadequate response to or poor tolerance for at least 1 MS therapy (Canadian patients). Patients were randomly assigned in a 3:1 ratio to 6 months of treatment with once-daily oral fingolimod 0.5?mg or standard-of-care DMTs. The primary study end-point was the change from baseline in treatment satisfaction as determined by the global satisfaction sub-scale of the Treatment Satisfaction Questionnaire for Medication. Secondary end-points included changes from baseline in perceived effectiveness and side-effects, and measures of activities of daily living, fatigue, depression, and quality-of-life. A 3-month open-label fingolimod extension was available for patients randomly assigned to the DMT group who successfully completed all study visits.Enrollment has been completed with 1053 patients; the patient population is generally older and has a longer duration of disease compared with populations from phase 3 studies of fingolimod.Inclusion criteria selected for patients with a sub-optimal experience with a previous DMT, limiting the collection of data on therapy change in patients who were satisfied with their previous DMT.Results of the EPOC study are anticipated in early 2013 and will inform treatment selection by providing patient-centered data on therapy switch to fingolimod or standard-of-care DMTs. Trial Registration: ClinicalTrials.gov NCT01216072.",
             "To analyze the global microbial composition, using large-scale DNA sequencing of 16 S rRNA genes, in faecal samples from colicky infants given L. reuteri DSM 17938 or placebo.Twenty-nine colicky infants (age 10-60 days) were enrolled and randomly assigned to receive either Lactobacillus reuteri (10(8) cfu) or a placebo once daily for 21 days. Responders were defined as subjects with a decrease of 50% in daily crying time at day 21 compared with the starting point. The microbiota of faecal samples from day 1 and 21 were analyzed using 454 pyrosequencing. The primers: Bakt_341F and Bakt_805R, complemented with 454 adapters and sample specific barcodes were used for PCR amplification of the 16 S rRNA genes. The structure of the data was explored by using permutational multivariate analysis of variance and effects of different variables were visualized with ordination analysis.The infants' faecal microbiota were composed of Proteobacteria, Firmicutes, Actinobacteria and Bacteroidetes as the four main phyla. The composition of the microbiota in infants with colic had very high inter-individual variability with Firmicutes/Bacteroidetes ratios varying from 4000 to 0.025. On an individual basis, the microbiota was, however, relatively stable over time. Treatment with L. reuteri DSM 17938 did not change the global composition of the microbiota, but when comparing responders with non-responders the group responders had an increased relative abundance of the phyla Bacteroidetes and genus Bacteroides at day 21 compared with day 0. Furthermore, the phyla composition of the infants at day 21 could be divided into three enterotype groups, dominated by Firmicutes, Bacteroidetes, and Actinobacteria, respectively.L. reuteri DSM 17938 did not affect the global composition of the microbiota. However, the increase of Bacteroidetes in the responder infants indicated that a decrease in colicky symptoms was linked to changes of the microbiota.ClinicalTrials.gov NCT00893711.",
            "There is considerable interest in dairy products from low-input systems, such as mountain-pasture grazing cows, because these products are believed to be healthier than products from high-input conventional systems. This may be due to a higher content of bioactive components, such as phytanic acid, a PPAR-agonist derived from chlorophyll. However, the effects of such products on human health have been poorly investigated.To compare the effect of milk-fat from mountain-pasture grazing cows (G) and conventionally fed cows (C) on risk markers of the metabolic syndrome.In a double-blind, randomized, 12-week, parallel intervention study, 38 healthy subjects replaced part of their habitual dietary fat intake with 39 g fat from test butter made from milk from mountain-pasture grazing cows or from cows fed conventional winter fodder. Glucose-tolerance and circulating risk markers were analysed before and after the intervention.No differences in blood lipids, lipoproteins, hsCRP, insulin, glucose or glucose-tolerance were observed. Interestingly, strong correlations between phytanic acid at baseline and total (P<0.0001) and LDL cholesterol (P=0.0001) were observed.Lack of effects on blood lipids and inflammation indicates that dairy products from mountain-pasture grazing cows are not healthier than products from high-input conventional systems. Considering the strong correlation between LDL cholesterol and phytanic acid at baseline, it may be suggested that phytanic acid increases total and LDL cholesterol.ClinicalTrials.gov, NCT01343589.",
             "One third of all cancer patients will develop bone metastases and the vertebral column is involved in approximately 70% of these patients. Conventional radiotherapy with of 1-10 fractions and total doses of 8-30?Gy is the current standard for painful vertebral metastases; however, the median pain response is short with 3-6 months and local tumor control is limited with these rather low irradiation doses. Recent advances in radiotherapy technology - intensity modulated radiotherapy for generation of highly conformal dose distributions and image-guidance for precise treatment delivery - have made dose-escalated radiosurgery of spinal metastases possible and early results of pain and local tumor control are promising. The current study will investigate efficacy and safety of radiosurgery for painful vertebral metastases and three characteristics will distinguish this study. 1) A prognostic score for overall survival will be used for selection of patients with longer life expectancy to allow for analysis of long-term efficacy and safety. 2) Fractionated radiosurgery will be performed with the number of treatment fractions adjusted to either good (10 fractions) or intermediate (5 fractions) life expectancy. Fractionation will allow inclusion of tumors immediately abutting the spinal cord due to higher biological effective doses at the tumor - spinal cord interface compared to single fraction treatment. 3) Dose intensification will be performed in the involved parts of the vertebrae only, while uninvolved parts are treated with conventional doses using the simultaneous integrated boost concept.It is the study hypothesis that hypo-fractionated image-guided radiosurgery significantly improves pain relief compared to historic data of conventionally fractionated radiotherapy. Primary endpoint is pain response 3 months after radiosurgery, which is defined as pain reduction of =?2 points at the treated vertebral site on the 0 to 10 Visual Analogue Scale. 60 patients will be included into this two-centre phase II trial.Results of this study will refine the methods of patient selection, target volume definition, treatment planning and delivery as well as quality assurance for radiosurgery. It is the intention of this study to form the basis for a future randomized controlled trial comparing conventional radiotherapy with fractionated radiosurgery for palliation of painful vertebral metastases.ClinicalTrials.gov Identifier: NCT01594892.",
             "We aim to evaluate the effectiveness of the Good School Toolkit, developed by Raising Voices, in preventing violence against children attending school and in improving child mental health and educational outcomes.We are conducting a two-arm cluster randomised controlled trial with parallel assignment in Luwero District, Uganda. We will also conduct a qualitative study, a process evaluation and an economic evaluation. A total of 42 schools, representative of Luwero District, Uganda, were allocated to receive the Toolkit plus implementation support, or were allocated to a wait-list control condition. Our main analysis will involve a cross-sectional comparison of the prevalence of past-week violence from school staff as reported by children in intervention and control primary schools at follow-up.At least 60 children per school and all school staff members will be interviewed at follow-up. Data collection involves a combination of mobile phone-based, interviewer-completed questionnaires and paper-and-pen educational tests. Survey instruments include the ISPCAN Child Abuse Screening Tools to assess experiences of violence; the Strengths and Difficulties Questionnaire to measure symptoms of common childhood mental disorders; and word recognition, reading comprehension, spelling, arithmetic and sustained attention tests adapted from an intervention trial in Kenya.To our knowledge, this is the first study to rigorously investigate the effects of any intervention to prevent violence from school staff to children in primary school in a low-income setting. We hope the results will be informative across the African region and in other settings.clinicaltrials.gov NCT01678846.",
             "Artemisinin-based combination therapy is very effective in clearing asexual stages of malaria and reduces gametocytemia, but may not affect mature gametocytes. Primaquine is the only commercially available drug that eliminates mature gametocytes.We conducted a 2-arm, open-label, randomized, controlled trial to evaluate the efficacy of single-dose primaquine (0.75 mg/kg) following treatment with dihydroartemisinin-piperaquine (DHP) on Plasmodium falciparum gametocytemia, in Indonesia. Patients aged =5 years with uncomplicated falciparum malaria, normal glucose-6-phosphate dehydrogenase enzyme levels, and hemoglobin levels =8 g/dL were assigned by computerized-generating sequence to a standard 3-day course of DHP alone (n = 178) or DHP combined with a single dose of primaquine on day 3 (n = 171). Patients were seen on days 1, 2, 3, and 7 and then weekly for 42 days to assess the presence of gametocytes and asexual parasites by microscopy. Survival analysis was stratified by the presence of gametocytes on day 3.DHP prevented development of gametocytes in 277 patients without gametocytes on day 3. In the gametocytemic patients (n = 72), primaquine was associated with faster gametocyte clearance (hazard ratio = 2.42 [95% confidence interval, 1.39-4.19], P = .002) and reduced gametocyte densities (P = .018). The day 42 cure rate of asexual stages in the DHP + primaquine and DHP-only arms were: polymerase chain reaction (PCR) unadjusted, 98.7% vs 99.4%, respectively; PCR adjusted, 100% for both. Primaquine was well tolerated.Addition of single-dose 0.75 mg/kg primaquine shortens the infectivity period of DHP-treated patients and should be considered in low-transmission regions that aim to control and ultimately eliminate falciparum malaria. Clinical Trials Registration. NCT01392014."]

stoplist = set('for a of the and to in'.split())

texts = [[word for word in document.lower().split() if word not in STOPWORDS]
         for document in nyt_data]


from collections import defaultdict


frequency = defaultdict(int)
for text in texts:
    for token in text:
        #print token
        frequency[token] += 1

from sets import Set
f1 = Set()




texts = [[token for token in text if frequency[token] > 1]
         for text in texts]

for text in texts:
    for token in text:
        f1.add(token)

#for f in f1:
    #print(f)



dictionary = corpora.Dictionary(texts)




dictionary.save('/home/mikhail/Documents/research/deerwester.dict')


corpus = [dictionary.doc2bow(text) for text in texts]

corpora.MmCorpus.serialize('/home/mikhail/Documents/deerwester.mm', corpus)

#lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=9)

#lsi.print_topics(num_topics=9, num_words=5)



#lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=9)

#lda.print_topics(num_topics=9, num_words=12)

tfidf = TfidfModel(corpus)

pca = TruncatedSVD(n_components=2)
X_reduced = pca.fit_transform(tfidf)

print(X_reduced)