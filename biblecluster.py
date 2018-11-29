import numpy as np
import pandas as pd
from nltk import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import  CountVectorizer, TfidfTransformer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import pairwise_distances as pair_dist
from sklearn.cluster import DBSCAN,KMeans
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from glob import glob
from bs4 import BeautifulSoup as bsoup
from time import time
import matplotlib.pyplot as plt
import re, itertools, pickle, itertools, scipy, json

class Clusterer:

    def __init__(self,k=8,is_tfidf=True):
        self.is_tfidf=is_tfidf
        self.vect=CountVectorizer(stop_words=ENGLISH_STOP_WORDS.union(self._load_stops()))
        self.tfidf=TfidfTransformer()
        self.cluster=KMeans(n_clusters=k)
        self.stats=dict()

    def _load_stops(self):
        with open('stops.json','rb') as f:
            self.stops=json.load(f)
        return self.stops

    def fit(self):
        start=time()
        self.sparse = self.vect.fit_transform(self.corpus)
        if self.is_tfidf:
            self.sparse = self.tfidf.fit_transform(self.sparse)
        self.cluster.fit(self.sparse)
        self.stats['time']=time()-start

    def transform_sentences(self):
        sentences = list(itertools.chain.from_iterable([sent_tokenize(passage) for passage in self.corpus]))
        self.sentences = np.array([s for s in sentences if len(s.split(' '))>5])
        self.sent_sparse = self.vect.transform(self.sentences)
        if self.is_tfidf:
            self.sent_sparse = self.tfidf.transform(self.sent_sparse)

    def calc_stats(self):
        self.stats['n_labels']=len(set(self.cluster.labels_)) - (1 if -1 in self.cluster.labels_ else 0)
        self.stats['silhouette'] =  metrics.silhouette_score(self.sparse, self.cluster.labels_,
                                      metric='euclidean',
                                      sample_size=len(self.corpus))
        self.stats['mean_centroid_dist'] = pd.DataFrame(pair_dist(self.cluster.cluster_centers_,self.cluster.cluster_centers_)).mean(0).to_dict()
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

    def analyze(self):
        self.calc_stats()
        self._group_passages()
        self.key = self._key_sentences()
        self.links = self._linking_sentences().to_dict()
        self.topics = self.get_topics()

    def write_results(self):
        for i in range(len(self.sections)):
            result={
            'key_sentences':self.key[i].tolist(),
            'linking_sentences': {k:v.tolist() for k,v in self.links.items() if str(i) in k},
            'topics': self.topics[i]}
            with open(f'results/cluster_{i}.json','w') as fp:
                json.dump(result,fp ,sort_keys=True,indent=4)
        with open(f'results/stats.json','w') as fp:
            json.dump(self.stats,fp ,sort_keys=True,indent=4)

    def _group_passages(self):
        self.sections=list()
        self.transform_sentences()
        df = pd.DataFrame(self.cluster.transform(self.sent_sparse))
        for i in set(self.cluster.labels_):
            self.sections.append(self.sentences[df.idxmin(1)==i])

    def _key_sentences(self,k=5):
        key_phrases=list()
        for sect in self.sections:
            ratings = self.pair_sim(sect).sum(0)
            indices = list(ratings.sort_values().index[:k])
            key_phrases.append(sect[indices]) 
        return key_phrases

    def _linking_sentences(self,k=5):
        df = pd.DataFrame(self.cluster.transform(self.sent_sparse))
        df = df.apply(lambda x: x.sort_values().head(2),1).fillna(0)
        results = pd.DataFrame()
        results['sums'] = df.sum(1)
        results['connections'] = df.apply(lambda x: ''.join(str(np.array(df.columns[x.values>0]))), 1)
        results['sentences'] = self.sentences
        return results.groupby('connections').apply(lambda y: y.sentences[:3].values)

    def pair_sim(self,texts):
        vectors = self.vect.transform(texts)
        if self.is_tfidf:
            vectors = self.tfidf.transform(vectors)
        return pd.DataFrame(pair_dist(vectors,vectors,'euclidean'))

    def get_topics(self,n=5):
        feature_names = self.vect.get_feature_names()
        topics=list()
        for topic_idx, topic in enumerate(self.cluster.cluster_centers_):
            topics.append([feature_names[i] for i in topic.argsort()[:-n - 1:-1]])
        return topics

    def plot(self):
        data = pd.DataFrame(TruncatedSVD(5).fit_transform(self.sparse))
        colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "blue"]
        a=0
        b=1
        df = pd.DataFrame(self.cluster.transform(self.sent_sparse))
        for i in set(self.cluster.labels_):
            x=data[df.idxmin(1)==i]
            plt.scatter(x[a],x[b],c=colors[i])
        plt.show()

if __name__ == "__main__":
    print('\rInstiating...',end='')
    c=Clusterer(6,False)
    print('\rLoading Corpus...',end='')
    c.load_corpus('bible')
    print('\rFitting Models...',end='')
    c.fit()
    print('\rAnalyzing...',end='')
    c.analyze()
    print('\rPrinting Results...')
    c.write_results()
