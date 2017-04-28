import wikipedia
import pickle
import inventory

def calc_score(idx):
    return (inventory.MAX_DOCS - idx)/(inventory.MAX_DOCS + 1)

def extractLabels(input_terms):
    output_terms = []
    for term in input_terms:
        print(term)
        doc_list = wikipedia.search(term)
        if doc_list == []:
            continue
        for idx, doc in enumerate(doc_list[:inventory.MAX_DOCS]):
            try:
                doc_page = wikipedia.page(doc)
                title = doc_page.title
                categories = doc_page.categories
                labels = [(title, calc_score(idx))]
                labels.extend([(cat, calc_score(idx)) for cat in categories if len(cat.split()) < 4])
                output_terms.append(labels)
            # check this error
            except wikipedia.exceptions.DisambiguationError:
                pass
            except wikipedia.exceptions.PageError:
                pass
    return output_terms

imp_terms = pickle.load(open('important_words.pkl', 'rb'))

for i in range(inventory.NUM_CLUSTERS):
    top_imp_terms = [x[0] for x in imp_terms[i][:inventory.NUM_TOP_WORDS]]
    print(top_imp_terms)
    pickle.dump(extractLabels(top_imp_terms), open('labels_words_{}.pkl'.format(i), 'wb'))

'''
TODO: Only the first title of the term was taken into consideration.
All titles used currently.
All categories of the page are added.
'''
