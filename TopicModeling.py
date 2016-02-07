__author__ = 'mikhail'

from nltk.tokenize import RegexpTokenizer

from gensim.parsing.preprocessing import STOPWORDS

from nltk.stem.porter import PorterStemmer

from gensim import corpora, models

doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health."

# compile sample documents into a list
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

tokenizer = RegexpTokenizer(r'\w+')


raw = doc_a.lower()
tokens = tokenizer.tokenize(raw)

#print(tokens)


stopped_tokens = [i for i in tokens if not i in STOPWORDS]



# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

print(stemmed_tokens)





dictionary = corpora.Dictionary(stemmed_tokens)

corpus = [dictionary.doc2bow(text) for text in stemmed_tokens]
print(corpus[0])