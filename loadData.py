'''
Course: Search Engine Architecture
Project: Cluster-Labeling

File: loadData.py
Job: Load the required dataset from internet and generate 
their TF-IDF indexes
'''
from urllib.request import urlopen
import tarfile, os, shutil, pickle, re
from scipy import sparse
import string, nltk
from math import log
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer

TF = {}
IDF = {}
doc_count = 0
doc_labeling = []
all_text = []

def get_stop_words():
    with open('stop_words_250.txt', 'r') as f:
        return [x.strip() for x in f.readlines()]

punctuation = "".join(string.punctuation)
tokenizer = RegexpTokenizer(r'[A-Za-z]\w+')
stop_words = nltk.corpus.stopwords.words('english') + get_stop_words()
UGLY_TEXT_MAP = dict([(ord(char), None) for char in '[]{}'] + [(ord(char), ' ') for char in '|=*\\#'])

link = 'http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz'

# with open('downloaded_data', 'wb') as f:
# 	f.write(urlopen(link).read())


# if os.path.exists('temp_data/'):
# 	shutil.rmtree('temp_data/')

# tarfile.open('downloaded_data', "r:gz").extractall(path='temp_data/')

# os.remove('downloaded_data')

def all_files(path):
	fl = []

	for f in os.listdir(path):
		df = os.path.join(path, f)
		if os.path.isdir(df):
			fl.extend(all_files(df))
		else:
			fl.append(df)

	return fl


def clean_text(text):
	text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.S)
	text = re.sub(r'<ref>.*?</ref>', '', text, flags=re.S)
	text = re.sub(r'\[\[File:.*?\|.*?\|.*?\|(.*?)\]\]', r'\1', text, flags=re.S)
	text = BeautifulSoup(text, 'lxml').get_text()
	text = text.translate(UGLY_TEXT_MAP)
	text = text.replace("'''", '"').replace("''", '"')
	text = text.strip()
	return text

def text_vector(textdata):
	# textdata = clean_text(textdata)
	# vec = [word.lower() for word in nltk.word_tokenize(textdata)]
	# vec = [i.strip(punctuation) for i in vec]
	vec = tokenizer.tokenize(textdata.lower())
	vec = [i for i in vec if i and i not in stop_words]
	# print(vec)
	return vec


def add_file(filepath, idx):
	doc_labeling.append(filepath)
	with open(filepath, 'rb') as f:
		textdata = f.read().decode('latin-1')
		tv = text_vector(textdata)
		all_text.append(tv)

		total_len = 1#float(len(tv)) #Normalizing text length
		for word in tv:
			if word not in TF:
				TF[word] = {}
			if idx not in TF[word]:
				TF[word][idx] = 0
			TF[word][idx] += 1/total_len

def gen_indexes(files):
	global doc_count
	for f in files:
		add_file(f, doc_count)
		doc_count += 1

	for word in TF.keys():
		count = len(TF[word].keys())
		IDF[word] = log((doc_count / float(count)))

files = all_files('temp_data')

gen_indexes(files)

pickle.dump(doc_labeling, open('doc_labels.pkl', 'wb'))
pickle.dump({'tf': TF, 'idf': IDF, 'doc_count': doc_count}, open('indexes.pkl', 'wb'))
pickle.dump(all_text, open('all_text.pkl', 'wb'))

# if os.path.exists('temp_data/'):
# 	shutil.rmtree('temp_data/')
