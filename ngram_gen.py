import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
from nltk.corpus import brown
import pickle

import mwxml, string

# UGLY_TEXT_MAP = dict([(ord(char), ' ') for char in string.punctuation + "“”‘’–—"])
#
# num_pages = sum(1 for _ in mwxml.Dump.from_file("info_ret.xml.gz").pages)
#
# dump = mwxml.Dump.from_file("info_ret.xml.gz")
# chunk_size = num_pages
#
# output = None
# for doc_id, page in enumerate(dump.pages):
#     if doc_id % chunk_size == 0:
#         output = open(str(int(doc_id / chunk_size)) + ".in", "w")
#     if page.namespace == 14:
#         continue
#     title = page.title
#     for revision in page:
#         text = revision.text
#     text = ' '.join(text.translate(UGLY_TEXT_MAP).replace('\n', ' ').lower().split())
#     output.write("%s\n" % text)
# output.close()
#
# token = nltk.word_tokenize(text)
token = brown.words()
# token = []
# token_list = pickle.load(open("all_text.pkl", "rb"))
# for tok in token_list:
#     token.extend(tok)

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
            open('ngrams_brown.pkl', 'wb'))
