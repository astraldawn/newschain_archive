"""
Class for performing similarity query
"""
__author__ = "Mark Lee"

import Queue
import sys

from gensim import *

import analyse


class SimMatrix(object):
    def __init__(self, data, refresh=False, clusters=True):
        """Does the processing on the data before performing a similarity
        query. This entails a conversion of the data into a LSI space. It has
        added ability to save and load indexes / dictionaries, so that if the
        same set of data is used (refresh=False), the code is sped up.

        Args:
            data (list): Contains the cluster data, each item in the list is a
                         single cluster. It is also possible for the data to
                         be articles, with each item in the list being a
                         single article. The program **defaults to clusters**
            refresh (bool): Set to true to rebuild all the indexes and save
                            them to the disk (this is necessary when loading
                            new data). Otherwise, for reasons of speed,
                            the indexes will be loaded from the disk
            clusters (bool): Decides whether to pull data from the cluster DB or
                             the article DB
        """
        self.data = data
        self.n_items = len(data)
        self.stop_list = analyse.get_stop_list()
        load_success = False

        self.dictionary = []
        # corpus = []
        self.lsi = []
        self.index = []
        self.bigram = []
        self.trigram = []

        if refresh is False:
            try:
                load_dir = "tmp_articles/"
                if clusters is True:
                    load_dir = "tmp_clusters/"
                # corpus = corpora.MmCorpus("/tmp/corpus.mm")
                self.dictionary = corpora.Dictionary.load(load_dir +
                                                          "dictionary.dict")
                self.lsi = models.LsiModel.load(load_dir + "model.lsi")
                self.index = similarities.MatrixSimilarity.load(
                    load_dir + "nyt.index")
                self.bigram = models.Phrases.load(load_dir + "bigram")
                self.trigram = models.Phrases.load(load_dir + "trigram")
                print "Loaded from files"
                load_success = True
            except:
                print "Did not load from files"
                load_success = False

        if not load_success or refresh is True:
            print "Rebuilding indexes"
            # texts = [[word for word in document.lower().split() if word not in
            #           self.stop_list] for document in
            #          [row[0] for row in self.data]]
            texts = [[word for word in document.lower().split() if word not in
                      self.stop_list] for document in
                     [row[0] + row[4] for row in self.data]]

            #  Bigrams
            self.bigram = models.Phrases(min_count=2, threshold=1)
            for item in texts:
                self.bigram.add_vocab([item])
            print self.bigram
            texts = self.bigram[texts]
            self.bigram.save("tmp/bigram")

            # Trigram
            self.trigram = models.Phrases(min_count=2, threshold=1)
            for item in texts:
                self.trigram.add_vocab([item])
            print self.trigram
            texts = self.trigram[texts]
            self.trigram.save("tmp/trigram")

            self.dictionary = corpora.Dictionary(texts)
            self.dictionary.save("tmp/dictionary.dict")
            corpus = [self.dictionary.doc2bow(text) for text in texts]
            # corpora.MmCorpus.serialize("/tmp/corpus.mm", corpus)
            tfidf = models.TfidfModel(corpus)
            corpus_tfidf = tfidf[corpus]
            self.lsi = models.LsiModel(corpus_tfidf, id2word=self.dictionary,
                                       num_topics=200)
            self.lsi.save("tmp/model.lsi")
            corpus_lsi = self.lsi[corpus]
            self.index = similarities.MatrixSimilarity(corpus_lsi)
            self.index.save("tmp/nyt.index")

    def query(self, article_no=0):
        """DEPRECATED

        This is a helper method for :func:`dijkstra<dijkstra>`. It converts the
        contents of the given article into the same LSI space as the data **(
        must be articles)** and finds the top 25 most similar articles by
        cosine similarity. It then returns these articles in res.

        Args:
            article_no (int): The article which the query is to be performed on

        Return:
            res (list): The ids of the top 25 most similar articles to
            article_no
        """
        query = [word for word in self.data[article_no][0].split() if word
                 not in self.stop_list]
        query = self.trigram[self.bigram[query]]
        # print query, self.data[article_no][7], self.data[article_no][2]
        vec_bow = self.dictionary.doc2bow(query)

        vec_lsi = self.lsi[vec_bow]
        sims = self.index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        cnt = 0
        # print sims
        res = []
        for (k, v) in sims:
            if cnt != 0:
                # print k, v, self.data[k][7], self.data[k][2], self.data[k][1]
                res.append((k, v))
                # print self.data[k][0]
            cnt += 1
            if cnt >= 25:
                break

        return res

    def dijkstra(self, article_no):
        """DEPRECATED

        **This method chains articles together.**

        Runs Djikstra's algorithm to find a path in the graph with the lowest
        average distance between a starting article and other articles in the
        corpus. When an article is examined, the 25 nearest nodes by
        similarity are added into the queue and explored (using
        :func:`query<query>`). The process ends when there are no new articles
        to explore. It then prints the best chain of articles.

        Args:
            article_no (int): The article to start dijkstra's algorithm from
        """
        print "----- Running Dijkstra on article " + str(article_no) + " -----"
        # print self.data[article_no][0], self.data[article_no][7], self.data[
        #      article_no][2]
        distance, prev, pathlen = {}, {}, {}
        pq = Queue.PriorityQueue()
        visited = set()

        for i in range(0, self.n_items):
            if i not in distance.keys():
                distance[i] = sys.float_info.max
                prev[i] = None
                pathlen[i] = 0

        pq.put((0, article_no, 0))
        distance[article_no] = 0
        pathlen[article_no] = 0

        while not pq.empty():
            c_dist, c_article, c_pathlen = pq.get()
            if c_article not in visited:
                visited.add(c_article)
            # print c_dist, c_article

            for (n_article, n_dist) in self.query(c_article):
                if n_article < c_article:
                    continue
                # print n_article, n_dist
                n_dist = 1 / n_dist
                alt = distance[c_article] + n_dist
                if alt < distance[n_article] and n_article not in visited:
                    prev[n_article] = c_article
                    distance[n_article] = alt
                    pathlen[n_article] = pathlen[c_article] + 1
                    pq.put((alt, n_article, pathlen[n_article]))

        res = []
        for i in range(0, self.n_items):
            if distance[i] is not sys.float_info.max and pathlen[i] is not 0:
                res.append((i, distance[i], pathlen[i], distance[i] /
                            pathlen[i]))

        res = sorted(res, key=lambda item: item[3])

        for item in res[:10]:
            print item

        item = res[0][0]
        if res[0][2] > 4:
            while (prev[item] is not None):
                # print item
                print item, self.data[item][0], self.data[item][7], self.data[
                    item][2]
                item = prev[item]
            print item, self.data[item][0], self.data[item][7], self.data[item][
                2]

        print "----- End Dijkstra -----\n"

    def addition(self, clusters):
        """DEPRECATED

        Attempts to find similar clusters to a given list of initial
        clusters in a greedy fashion.

        Algorithm
            - Take an initial list of N clusters
            - For each of the clusters in the list, generate a list of
              similar clusters
            - With N lists of clusters, identify the cluster which is the most
              similar to all N initial clusters by searching for the cluster
              with the highest combined similarity score (summed over the N
              lists)
            - If that cluster is not in the initial list, add it to the list
              and repeat, otherwise the algorithm terminates

        Return:
            cluster_set (list): The ids of the clusters in the chain
        """
        print "----- Greedy addition article " + str(clusters) + " -----"
        cluster_set = clusters
        new_addition = True
        score = 0
        while new_addition is True:
            freq_tmp = {}
            for cluster in cluster_set:
                for (k, v) in self.query(cluster):
                    if k not in freq_tmp:
                        freq_tmp[k] = v
                    else:
                        freq_tmp[k] += v
            freq_tmp = sorted(freq_tmp.items(), key=lambda item: -item[1])
            candidate = freq_tmp[0][0]

            if candidate in cluster_set:
                new_addition = False
            else:
                score += freq_tmp[0][1]
                cluster_set.append(candidate)

        print len(cluster_set), score, score / len(cluster_set)

        # for cluster in sorted(cluster_set):
        #     article_list = self.data[cluster][4].strip("[]").split(", ")
        #     for article_id in article_list:
        #         res = utils.load_nyt_by_article_id(article_id)
        #         print res[0][0], res[0][4], res[0][2]
        #     print
        print "----- End greedy addition -----\n"
        return cluster_set

    def keyword_query(self, keyword, n_cluster=5):
        """Selects the top N most similar clusters to the keyword provided
        and returns their ids in sorted order.

        Args:
            keyword (str): The desired keywords for the query, space separated
            n_cluster (int): The number of most similar clusters to return

        Return
            res (list): The ids of the top N most similar clusters in sorted
            order
        """
        query = [word for word in keyword.split() if word
                 not in self.stop_list]
        query = self.trigram[self.bigram[query]]
        # print query, self.data[article_no][7], self.data[article_no][2]
        vec_bow = self.dictionary.doc2bow(query)

        vec_lsi = self.lsi[vec_bow]
        sims = self.index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        # for (k,v) in sims[:20]:
        #     print k,v, self.data[k]
        # for k in sorted([k for (k, v) in sims[:20]]):
        #     print self.data[k]
        res = sorted([k for (k, v) in sims[:n_cluster]])
        return res


if __name__ == "__main__":
    pass
