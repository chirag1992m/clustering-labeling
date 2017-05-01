'''
Course: Search Engine Architecture
Project: Cluster-Labeling

File: loadData.py
Job: Load the required dataset from internet and generate 
their TF-IDF indexes
'''
from urllib.request import urlopen
import tarfile, os, shutil, pickle
import string, nltk
from math import log
from nltk.tokenize import RegexpTokenizer

def get_stop_words():
    with open('data_supplements/stop_words_250.txt', 'r') as f:
        return [x.strip() for x in f.readlines()]

tokenizer = RegexpTokenizer(r'[A-Za-z][A-Za-z]+')
stop_words = nltk.corpus.stopwords.words('english') + get_stop_words()

def all_files(path):
	fl = []

	for f in os.listdir(path):
		df = os.path.join(path, f)
		if os.path.isdir(df):
			fl.extend(all_files(df))
		else:
			fl.append(df)

	return fl

def text_vectorize(textdata):
    vec = [i.strip() for i in tokenizer.tokenize(textdata.lower())]
    vec = [i for i in vec if i and i not in stop_words]
    return vec

def gen_indexes_20_news(files):
    doc_count = 0
    TF, IDF = {}, {}
    doc_labeling, all_text = [], []

    for filepath in files:
        doc_labeling.append(os.path.basename(os.path.dirname(filepath)))
        with open(filepath, 'rb') as f:
            textdata = f.read().decode('latin-1')
            tv = text_vectorize(textdata)
            all_text.append(''.join(textdata))

            total_len = 1
            for word in tv:
                if word not in TF:
                    TF[word] = {}
                if doc_count not in TF[word]:
                    TF[word][doc_count] = 0
                TF[word][doc_count] += 1/total_len
        doc_count += 1

    for word in TF.keys():
        count = len(TF[word].keys())
        IDF[word] = log((doc_count / float(count)))

    return {'tf': TF, 'idf': IDF, 'doc_count': doc_count}, doc_labeling, all_text

def download_20_newsgroup():
    link = 'http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz'

    if not os.path.exists('20news-bydate.tar.gz'):
        with urlopen(link) as cont:
            open('20news-bydate.tar.gz', 'wb').write(cont.read())
        tarfile.open('20news-bydate.tar.gz', "r:gz").extractall(path='20news/')

def clean_20_newsgroup():
    if os.path.exists('20news/'):
        os.remove('20news-bydate.tar.gz')

    if os.path.exists('20news/'):
        shutil.rmtree('20news/')

def load_20_newsgroup(options):
    print("Loading 20-newsgroup dataset...")
    download_20_newsgroup()
    files = all_files('20news')
    indexes, labels, all_text = gen_indexes_20_news(files)

    if options.save_intermediate:
        pickle.dump(labels, open('intermediate_results/doc_labels.pkl', 'wb'))
        pickle.dump(indexes, open('intermediate_results/indexes.pkl', 'wb'))
        pickle.dump(all_text, open('intermediate_results/all_text.pkl', 'wb'))
        pickle.dump(set(labels), open('intermediate_results/unique_labels.pkl', 'wb'))

    if options.clean_data:
        clean_20_newsgroup()

    return indexes, labels, all_text
