'''
Course: Search Engine Architecture
Project: Cluster-Labeling

File: loadData.py
Job: Load the required dataset from internet and generate 
their TF-IDF indexes
'''
from urllib.request import urlopen
import tarfile, os, shutil, pickle
from scipy import sparse
import string, nltk
from math import log

TF = {}
IDF = {}
doc_count = 0
doc_labeling = []


punctuation = string.punctuation
stop_words = nltk.corpus.stopwords.words('english')

link = 'http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz'

with open('downloaded_data', 'wb') as f:
	f.write(urlopen(link).read())


if os.path.exists('temp_data/'):
	shutil.rmtree('temp_data/')

tarfile.open('downloaded_data', "r:gz").extractall(path='temp_data/')

os.remove('downloaded_data')

def all_files(path):
	fl = []

	for f in os.listdir(path):
		df = os.path.join(path, f)
		if os.path.isdir(df):
			fl.extend(all_files(df))
		else:
			fl.append(df)

	return fl

def text_vector(textdata):
	vec = nltk.word_tokenize(textdata)
	vec = [i.strip("".join(punctuation)) for i in vec]
	vec = [i for i in vec if i not in stop_words]
	return vec


def add_file(filepath, label, idx):
	doc_labeling.append(label)
	with open(filepath, 'rb') as f:
		textdata = "".join(str(f.read()).split())
		tv = text_vector(textdata)

		for word in tv:
			if word not in TF:
				TF[word] = {}
			if idx not in TF[word]:
				TF[word][idx] = 0
			TF[word][idx] += 1

def gen_indexes(files):
	global doc_count
	for f in files:
		label = os.path.basename(os.path.dirname(f))
		add_file(f, label, doc_count)
		doc_count += 1

	for word in TF.keys():
		count = len(TF[word].keys())
		IDF[word] = log((doc_count / float(count)))

files = all_files('temp_data')

gen_indexes(files)

pickle.dump({'tf': TF, 'idf': IDF, 'labels': doc_labeling, 'doc_count': doc_count}, open('indexes.pkl', 'wb'))

if os.path.exists('temp_data/'):
	shutil.rmtree('temp_data/')
