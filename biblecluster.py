import numpy as np
import pandas as pd
from nltk import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import  CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import pairwise_distances as pair_dist
from sklearn.cluster import DBSCAN,KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.decomposition import PCA
from glob import glob
from bs4 import BeautifulSoup as bsoup
from time import time
import re, itertools, pickle, itertools, scipy


class Clusterer:

    def __init__(self,k=10):
        self.vect=CountVectorizer()
        self.tfidf=TfidfTransformer()
        self.cluster=KMeans(n_clusters=k)
        self.stats=dict()

    def _load_stops(self):
        with open('stops.json','rb') as f:
            self.stats=json.load(f)

    def fit(self):
        start=time()
        self.sparse = self.vect.fit_transform(self.corpus)
        self.sparse = self.tfidf.fit_transform(self.sparse)
        self.cluster.fit(self.sparse)
        self.stats['time']=time()-start

    def calc_stats(self):
        self.stats['n_labels']=len(set(self.cluster.labels_)) - (1 if -1 in self.cluster.labels_ else 0)
        self.stats['silhouette'] =  metrics.silhouette_score(self.sparse, self.cluster.labels_,
                                      metric='euclidean',
                                      sample_size=len(self.corpus))
        return self.stats

    def save_corpus(self,path):
        with open(path,'wb') as fp:
            pickle.dump((self.corpus, self.books),fp)

    def load_corpus(self,path):
        with open(path,'rb') as fp:
            self.corpus, self.books=pickle.load(fp)

    def _read_text(self,path):
        try:
            page = bsoup(open(path,'rb').read(),'html.parser')
            text = [div.text for div in page.find('div',class_='main').find_all('div')[1:-2]]
            return re.sub(' [0-9]+\\xa0','',''.join(text))
        except:
            pass

    def write_text(self):
        paths=np.array(glob('eng-web_html/*.htm'))
        texts = np.array(list(map(self._read_text,paths)))
        self.corpus = texts[texts != np.array(None)]
        self.books = np.array([re.sub('(.*/|[0-9]+.htm)','',x) for x in paths[texts != np.array(None)]])

    def write_results(self):
        self._group_passages()
        sentences = self._key_sentences()
        for i,sent in enumerate(sentences):
            with open(f'results/cluster_{i}.json','w') as fp:
                json.dump(sent.tolist(),fp ,sort_keys=True,indent=4)

    def _group_passages(self):
        self.sections=list()
        for i in set(self.cluster.labels_):
            self.sections.append(self.corpus[self.cluster.labels_==i])

    def _key_sentences(self,k=5):
        key_phrases=list()
        for sect in self.sections:
            sentences = np.array(list(itertools.chain.from_iterable([sent_tokenize(passage) for passage in sect])))
            ratings = self.pair_sim(sentences).sum(0)
            indices = list(ratings.sort_values().index[:k])
            key_phrases.append(sentences[indices]) 
        return key_phrases

    def pair_sim(self,texts):
        vectors = self.vect.transform(texts)
        vectors = self.tfidf.transform(vectors)
        return pd.DataFrame(pair_dist(vectors,vectors,'euclidean'))


if __name__ == "__main__":
    C=Clusterer()
    c.load_corpus('bible')
    c.fit()
    c.calc_stats()
    c.analyze()
