import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
from nltk.corpus import brown
import pickle

# token = nltk.word_tokenize(text)
token = brown.words()
unigrams = Counter([" ".join(x) for x in ngrams(token,1)])
bigrams = Counter([" ".join(x) for x in ngrams(token,2)])
trigrams = Counter([" ".join(x) for x in ngrams(token,3)])
fourgrams = Counter([" ".join(x) for x in ngrams(token,4)])
fivegrams = Counter([" ".join(x) for x in ngrams(token,5)])

pickle.dump({1: unigrams, 2: bigrams, 3: trigrams, 4: fourgrams, 5: fivegrams}, open('ngrams_brown.pkl', 'wb'))
