#Test script
#To be deleted in the end
from urllib.request import urlopen
import tarfile, os, shutil, pickle, re
from scipy import sparse
import string, nltk
from math import log
from bs4 import BeautifulSoup
import sklearn
from sklearn import cluster
from collections import Counter

all_text = []
doc_count = 0

punctuation = string.punctuation
# stop_words = nltk.corpus.stopwords.words('english')

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

UGLY_TEXT_MAP = dict([(ord(char), None) for char in '[]{}'] + [(ord(char), ' ') for char in '|=*\\#'])
STOPWORDS = nltk.corpus.stopwords.words('english')

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
	textdata = clean_text(textdata)
	term_list = [word.lower() for word in  nltk.word_tokenize(textdata) if word.lower() not in STOPWORDS]
	return term_list


def add_file(filepath, idx):
	with open(filepath, 'rb') as f:
		textdata = f.read().decode('latin-1')
		tv = clean_text(textdata)

		all_text.append(tv)

def gen_indexes(files):
	global doc_count
	for f in files:
		add_file(f, doc_count)
		doc_count += 1

files = all_files('temp_data')

gen_indexes(files)

# if os.path.exists('temp_data/'):
# 	shutil.rmtree('temp_data/')


tfidf = sklearn.feature_extraction.text.TfidfVectorizer(stop_words='english')
# pickle.dump(all_text, open('all_text.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
X = tfidf.fit_transform(all_text)

print(X.shape)

kmeans = cluster.KMeans(n_clusters=6, verbose=0).fit(X)
print(Counter(kmeans.labels_))
