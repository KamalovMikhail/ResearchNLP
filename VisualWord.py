__author__ = 'mikhail'
import numpy as np
import matplotlib.pyplot as plt
import re
from operator import itemgetter

def tokenize(text):
    tokenizer = re.compile('\\W*')
    return tokenizer.split(text.lower())


def word_count(text):
    words = tokenize(text)
    word_freq = dict([(word, words.count(word)) for word \
    in set(words)])
    return word_freq


def top_words(text,n=50):
    wordfreq = word_count(text)
    topwords = sorted(wordfreq.iteritems(), key = itemgetter(1),reverse=True)[:n]
    return topwords

def plot_freq_tag(text):
    tfw = top_words(text, n=10)
    words = [tfw[i][0] for i in range(len(tfw))]
    x = range(len(tfw))
    np = len(tfw)
    y = []
    for item in range(np):
        y = y + [tfw[item][1]]
    fig = plt.figure()
    ax = fig.add_subplot(111,xlabel="Word Rank",ylabel="Word Freqquncy")
    ax.set_title('Top 50 words')
    ax.plot(x, y, 'go-',ls='dotted')
    plt.xticks(range(0, len(words) + 1, 1))
    plt.yticks(range(0, max(y) + 15, 10))
    for i, label in enumerate(words):
        plt.text (x[i], y[i], label ,rotation=45)
    plt.show()
text = open('/home/mikhail/Documents/research/FileSet/NCT00000470_Procedure.txt','r').read()
plot_freq_tag(text)