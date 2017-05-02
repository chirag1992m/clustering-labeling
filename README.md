# Cluster Labeling
### Enhancing Cluster Labeling using Wikipedia
### Search Engine Architecture (NYU CS Spring-17) Project
#### [Report](), [Slides]()

## Datasets
* [20-newsgroup](http://people.csail.mit.edu/jrennie/20Newsgroups/)

## Language
Python >= 3.5

## Required Packages
* wikipedia
* nltk (+ brown corpora)
* gensim
* [GoogleNews Trained Word2Vec model](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)

To install the packages, run:
~~~
$ pip install wikipedia
$ pip install gensim
~~~

To install the ntlk brown corpora:
~~~
$ python

>>> import nltk
>>> nltk.download('brown')
>>> exit()
~~~

Download the [`GoogleNews Trained Word2Vec model`](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing), extract and put it in the root directory of the project which will be used by the evaluation mechanism.

## Pipeline Overview
1. Data download and Pre-process
2. TF-IDF vetorization 
3. Clustering (KMeans, Gaussian Mixture Model, Birch Clustering, Agglomerative Clustering)
4. Impoprtant Term Extraction (JSD, Naive term weighing)
5. Search Wikipedia for the important terms
6. Judge Candidate labels(Important terms + Wikipedia Labels) using (MI, SP) 
7. Evaluation using GoogleNews Word2Vec Similarity metric

## How to run
You can extract cluster labels based on different combinations of clustering methods, Important Term Extraction, using wikipedia or not, and using the specific judging mechanism. Respective option needs to be provided while running `main.py`. The whole list of options available are:
~~~
usage: main.py [-h] [-d {20newsgroup}] [-c {kmeans,gmm,birch,ac}]
               [-i {JSD,naive}] [-nw] [-j {PMI,SP}] [-nc NUM_CLUSTERS]
               [-ni NUM_IMPORTANT_WORDS] [-K TOP_K]
               [--num_wiki_results NUM_WIKI_RESULTS] [-clean] [-s]
               [-o OUT_DIRECTORY] [-io INTERMEDIATE_OUT_DIRECTORY]

optional arguments:
  -h, --help            show this help message and exit
  -d {20newsgroup}, --dataset {20newsgroup}
                        Dataset to use.
  -c {kmeans,gmm,birch,ac}, --clustering {kmeans,gmm,birch,ac}
                        Clustering algorithm to be used
  -i {JSD,naive}, --important_terms {JSD,naive}
                        Algorithm for choosing the cluster representative
                        terms
  -nw, --no_wiki_search
                        Label clusters without the searching over wikipedia
  -j {PMI,SP}, --judge {PMI,SP}
                        The judging mechanism to be used
  -nc NUM_CLUSTERS, --num_clusters NUM_CLUSTERS
                        Number of clusters to be made from dataset
  -ni NUM_IMPORTANT_WORDS, --num_important_words NUM_IMPORTANT_WORDS
                        Number of important words to be extracted
  -K TOP_K, --top_K TOP_K
                        K of the Top-K results
  --num_wiki_results NUM_WIKI_RESULTS
                        Number of top wiki results to be used
  -clean, --clean_data  Delete downloaded data after usage
  -s, --save_intermediate
                        Save any intermediate results during the whole process
  -o OUT_DIRECTORY, --out_directory OUT_DIRECTORY
                        Directory where the final names will be displayed
  -io INTERMEDIATE_OUT_DIRECTORY, --intermediate_out_directory INTERMEDIATE_OUT_DIRECTORY
                        Directory where all the intermediate files will be
                        saved
~~~

To evaluate the labeling performance, run `evaluation.py` with the __output directory of main.py__ as the only argument. For eg:
~~~
$ python main.py -c kmeans -o 'eval_kmeans'

$ python evaluation.py 'eval_kmeans'
~~~