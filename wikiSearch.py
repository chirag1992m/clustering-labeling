import wikipedia
import pickle

def calc_score(options, idx):
    return (options.num_wiki_results - idx)/(options.num_wiki_results + 1)

def extractCandidateLabels(options, input_terms):
    output_terms = []
    for term in input_terms:
        # print(term)
        doc_list = wikipedia.search(term)
        if doc_list == []:
            continue
        for idx, doc in enumerate(doc_list[:options.num_wiki_results]):
            try:
                doc_page = wikipedia.page(doc)
                title = doc_page.title
                categories = doc_page.categories
                if len(title.split()) < 4:
                    labels = [(title, calc_score(options, idx))]
                else:
                    labels = []
                labels.extend([(cat, calc_score(options, idx)) for cat in categories if len(cat.split()) < 4])
                if len(labels) > 0:
                    output_terms.append(labels)
            # check this error
            except wikipedia.exceptions.DisambiguationError:
                pass
            except wikipedia.exceptions.PageError:
                pass
    return output_terms
