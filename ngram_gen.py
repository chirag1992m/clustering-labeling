import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
from nltk.corpus import brown
import pickle

def dump_ngrams(token, filename):
	unigrams = Counter([" ".join(x) for x in ngrams(token,1)])
	bigrams = Counter([" ".join(x) for x in ngrams(token,2)])
	trigrams = Counter([" ".join(x) for x in ngrams(token,3)])
	fourgrams = Counter([" ".join(x) for x in ngrams(token,4)])
	fivegrams = Counter([" ".join(x) for x in ngrams(token,5)])
	count_1 = sum(unigrams.values())
	count_2 = sum(bigrams.values())
	count_3 = sum(trigrams.values())
	count_4 = sum(fourgrams.values())
	count_5 = sum(fivegrams.values())

	pickle.dump({1: unigrams, '1_count': count_1,
	             2: bigrams, '2_count': count_2, 3: trigrams, '3_count': count_3,
	            4: fourgrams, '4_count': count_4, 5: fivegrams, '5_count': count_5},
	            open(filename, 'wb'))

def generate_brown_ngrams():
	tokens = brown.words()
	dump_ngrams(tokens, 'data_supplements/ngrams_brown.pkl')