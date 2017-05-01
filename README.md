# clustering-labeling
Search Engine Architecture (NYU CS Spring-17) Project
**Enhancing Cluster Labeling using Wikipedia**

## Datasets
* [20-newsgroup](http://people.csail.mit.edu/jrennie/20Newsgroups/)

## Required Packages
* wikipedia
* nltk (+ brown corpora)
* gensim

To install the packages, run:
~~~
pip install wikipedia
pip install gensim
~~~

## Pipeline Overview
1. Data download and Pre-process
2. TF-IDF vetorization 
3. Clustering (KMeans)
4. Impoprtant Term Extraction (JSD, Naive term weighting)
5. Search Wikipedia for the important terms
6. Judge Candidate labels(Important terms + Wikipedia Labels) using (MI, SP) 
7. Evaluation using Match@K
