"""
Classes to perform clustering on individual articles.
"""
__author__ = 'Mark Lee'

from collections import Counter
import re
import Queue
import itertools

from gensim import *
from sklearn.cluster import KMeans
from sklearn.neighbors import DistanceMetric
import numpy as np

import snap_cluster_lib

VERBOSE = False


def get_stop_list():
    """DEPRECATED

    Stopword list generator

    Returns:
        list: list of stopwords
    """
    stop_list \
        = set("a about above after again against all am an and "
              "any "
              "are aren't as at be because been before being below "
              "between both but by can't cannot could couldn't day did "
              "didn't do does doesn't doing don't down during each "
              "few for from further had hasn't has have haven't "
              "having he he'd he'll he's her here here's hers "
              "herself him himself his how how's i i'd i'll i'm "
              "i've if in into is isn't it it's its itself let's me "
              "more most mustn't my  myself no nor not of off on "
              "once only or other ought our ours ourselves out over "
              "own same shan't she she'd she'll she's should "
              "shouldn't so some such than that that's the their "
              "theirs them themselves then there there's these they "
              "they'd they'll they've they're this those through to "
              "too under until up very was wasn't we we'd we'll "
              "we're we've were weren't what what's when when's "
              "where where's which while who who's whom why why's "
              "with won't would wouldn't you you'd you'll you're "
              "you've your yours yourself yourselves read quick iht image"
              .split())

    return stop_list


class ExtractFeatures(object):
    """Extracts features from data extracted from SQL DB in a specific format.

    DB Format
        - name text
        - section text
        - date datetime
        - wordcnt int
        - summary text
        - id int

    Usage
        - load_data() with the desired data in appropriate format
        - pre_processing()
        - transform_tfidf()
        - model_lsi()
    """

    def __init__(self):
        """Declares variables for later use
        """
        self.data = []
        self.documents = []
        self.dictionary = []
        self.corpus = []
        self.corpus_transformed = []
        self.corpus_lsi = []

        self.stop_list = get_stop_list()

    def load_data(self, data):
        """ Loads data in the format specified above.

        Loaded data located in self.data.
        """
        self.data = data

    def word_freq(self):
        """DEPRECATED

        Extracts the 75 most common words in the data and adds them to the
        stopword list.

        Returns:
            list: A modified stop word list
        """
        word_list = []
        for row in self.data:
            word_list += row[0].lower().split()
        word_counter = Counter(word_list)
        for (k, v) in word_counter.most_common(75):
            self.stop_list.add(k)
        print sorted(self.stop_list)

    def pre_processing_line(self):
        """Helper function for feature extraction in snap_cluster_lib.
        """
        self.documents = [str(row) for row in self.data]
        self.parsing()

    def pre_processing(self):
        """Combines the titles with the summaries of the articles for further
        processing. Combined articles and summaries located in self.documents.

        Calling this function will result in a call to the parsing function.
        """
        self.documents = []
        for row in self.data:
            row[0] = row[0] + " " + row[4]
            row[4] = ""
            row[0] = re.sub(r'[^\w\s]', '', row[0])
            self.documents.append(row[0])
        self.parsing()

    def parsing(self):
        """Performs standard text processing on the contents of self.documents.

        Steps:
            - Removal of html tags, punctuation, multiple whitespaces,
              numbers and stopwords
            - Addition of bigrams and trigrams to the dictionary (
              self.dictionary)
            - Conversion into a bag of words corpus (self.corpus)
        """
        texts = []
        for text in self.documents:
            texts.append(parsing.preprocess_string(
                str.lower(str(text)),
                filters=[parsing.strip_tags,
                         parsing.strip_punctuation,
                         parsing.strip_multiple_whitespaces,
                         parsing.strip_numeric,
                         parsing.remove_stopwords]
            )
            )

        #  Bigrams
        bigram = models.Phrases(min_count=2, threshold=1)
        for item in texts:
            bigram.add_vocab([item])
        texts = bigram[texts]

        # Trigram
        trigram = models.Phrases(min_count=2, threshold=1)
        for item in texts:
            trigram.add_vocab([item])
        texts = trigram[texts]

        self.dictionary = corpora.Dictionary(texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]

        if VERBOSE:
            print bigram
            print trigram
            print self.dictionary


    def transform_corpus(self):
        """Transforms bag of words corpus into a LogEntropy weighted corpus,
        reducing the weight of unimportant terms. The transformed corpus is
        located in self.corpus_transformed.
        """
        tfidf = models.LogEntropyModel(self.corpus)
        # tfidf = models.TfidfModel(self.corpus)
        self.corpus_transformed = tfidf[self.corpus]

    def model_lsi(self, n_topics=200):
        """Performs dimensionality reduction on the LogEntropy weighted
        corpus using latent semantic indexing (LSI). The corpus,
        now converted into LSI space, is located in self.corpus_lsi.

        Args:
            n_topics (int): the number of topics in the LSI space
        """
        lsi = models.LsiModel(self.corpus_transformed, id2word=self.dictionary,
                              num_topics=n_topics)
        self.corpus_lsi = lsi[self.corpus_transformed]

        if VERBOSE:
            print "Executed LSI"
            print "Topics: ", n_topics
            lsi.print_topics(n_topics)

    def print_data(self):
        """Helper function that prints the contents of self.data
        """
        for row in self.data:
            print row


class ComputedCluster(object):
    """Stores a computed cluster and its associated date in an easy format for
    subsequent processing.
    """
    def __init__(self, data):
        """Breaks down raw data into different fields for subsequent
        processing into SQL DB. Also initialises metrics for the cluster.

        Args:
            data (list): Articles in raw format that are included in the cluster
        """
        self.data = data
        self.size = len(data)
        self.content = " ".join([x[0] for x in data])
        self.articles_id = [x[5] for x in data]
        self.articles_date = [x[2] for x in data]
        self.articles_length = [x[3] for x in data]
        self.diameter = 0
        self.centroid = 0
        self.radius = 0
        self.closest_article_id = self.articles_id[0]
        self.closest_article_distance = 0
        self.closest_article_content = []

    def compute_metrics(self, corpus, article_pos):
        """Computes metrics for the given cluster. Metrics computed are:
        diameter, radius, centroid, closest article to centroid, the distance
        of the closest article to the centroid.

        Args:
            corpus: A corpus in LSI space
            article_pos (dict): Maps the article id to the actual
                                positions of the article in the corpus
        """
        dist_corpus = [corpus[article_pos[x]] for x in self.articles_id]

        # Centroid calculation
        self.centroid = np.average(dist_corpus, axis=0)

        # Diameter calculation
        dist = DistanceMetric.get_metric('euclidean')
        dist_pair = dist.pairwise(dist_corpus)
        self.diameter = max(list(itertools.chain.from_iterable(dist_pair)))

        # Radius calculation
        dist_corpus.append(self.centroid)
        dist_pair = dist.pairwise(dist_corpus)
        centroid_dist = [x for x in dist_pair[-1] if x > 0]
        if len(centroid_dist) > 0:
            self.radius = max(centroid_dist)

            # Closest article computation
            closest_article = self.articles_id[0]
            min_dist = self.radius
            tmp_content = []

            for k, id in enumerate(self.articles_id):
                if centroid_dist[k] < min_dist:
                    closest_article = id
                    min_dist = centroid_dist[k]
                    tmp_content = self.data[k]

            self.closest_article_id = closest_article
            self.closest_article_distance = min_dist
            self.closest_article_content = tmp_content

    def display(self):
        """Helper function to display the contents of a computed cluster
        """
        print "----- Computed cluster -----"
        for row in self.data:
            print row

        print
        print "Diameter: ", self.diameter
        print "Radius: ", self.radius
        print "Closest article: ", self.closest_article_content
        print "Closest article id: ", self.closest_article_id
        print "Closest article distance: ", self.closest_article_distance
        print


class BisectingKmeans(object):
    """Performs clustering using Bisecting Kmeans.

    Algorithm
        - Begin with all items in a single cluster
        - Use normal kmeans to split the cluster into 2 clusters
        - Repeat the previous step until some termination condition is reached.
          For this project, we terminate when there is no cluster which
          contains more than 5 items.
    """
    def __init__(self, data):
        """Initialises the object, key parameter to modify is
        max_cluster_size, to determine when the algorithm terminates. We
        insert the full data into the queue initially, and it functions as
        the first cluster.

        Args:
            data (list): list of articles, 1 article per row
        """
        self.max_cluster_size = 6

        self.computed_clusters = []
        self.articles_lost = 0
        self.cluster_queue = Queue.PriorityQueue()
        self.cluster_queue.put((-len(data), QueueCluster(data)))
        self.first_cluster = False
        self.original_corpus = []
        self.original_article_pos = {}
        self.max_centroid_dist = 0

    def compute(self):
        """Performs Bisecting Kmeans on the given data.

        Parameters for Kmeans are as follows (detailed explanation at scikit
        learn website)

        - n_clusters: 2 (split into 2)
        - max_iter: 500
        - n_init: 50
        """
        if VERBOSE:
            print "----- Computing cluster -----"
        # We need to fix the random state
        self.cluster_object = KMeans(n_clusters=2, max_iter=500,
                                     n_init=50, n_jobs=-2, random_state=1)

        # Run out of clusters to process
        while not self.cluster_queue.empty():

            cur_item = self.cluster_queue.get()
            cur_cluster = cur_item[1]

            if self.first_cluster is False:
                self.first_cluster = True
                self.original_corpus = cur_cluster.corpus
                original_article = [x[5] for x in cur_cluster.data]
                for i, id in enumerate(original_article):
                    self.original_article_pos[id] = i

            if len(cur_cluster.data) < self.max_cluster_size:
                self.computed_clusters.append(ComputedCluster(
                    cur_cluster.data))
                continue

            # This code is to fix issues that occur with raw input data
            try:
                # Compute a cluster
                self.predict_results = self.cluster_object.fit_predict(
                    cur_cluster.corpus)
                self.labels = self.cluster_object.labels_
            except ValueError:
                # Computation fails only for this reason
                logging.debug("VALUE ERROR")

                # Show the documents which do not conform
                doc_len = [len(x) for x in cur_cluster.corpus]
                logging.debug(str(doc_len))

                # Helper to find the offending documents
                mode = Counter(doc_len).most_common(1)
                mode = mode[0][0]
                logging.debug("MODE %s", str(mode))

                # Create a new cluster by removing the offending documents
                tmp_data = []
                for i in range(0, len(cur_cluster.data)):
                    if doc_len[i] != mode:
                        logging.debug(str(cur_cluster.data[i]))
                        self.articles_lost += 1
                    else:
                        tmp_data.append(cur_cluster.data[i])

                logging.debug("ARTICLES LOST %s", self.articles_lost)

                # Make a prediction on the new cluster
                self.cluster_queue.put(
                    (-len(tmp_data), QueueCluster(tmp_data)))
                continue

            # print self.predict_results

            for i in range(0, 2):
                if VERBOSE:
                    print "----- Label: " + str(i) + " -----"
                    print "labels: " + str(len(self.labels)) + str(len(
                    cur_cluster.data))

                # Create a new cluster for a label
                tmp_data = []
                for j in range(0, len(self.labels)):
                    if self.labels[j] == i:
                        tmp_data.append(cur_cluster.data[j])

                if VERBOSE:
                    print len(tmp_data)

                if len(tmp_data) != 0:
                    self.cluster_queue.put(
                        (-len(tmp_data), QueueCluster(tmp_data)))

    def find_computed_cluster_metrics(self):
        """Initialises cluster metric computation over every cluster that is
        found by the given clustering algorithm.
        """
        for cluster in self.computed_clusters:
            cluster.compute_metrics(self.original_corpus,
                                    self.original_article_pos)

        centroid_locs = [x.centroid for x in self.computed_clusters]
        dist = DistanceMetric.get_metric('euclidean')
        dist_pair = dist.pairwise(centroid_locs)
        self.max_centroid_dist = max(list(itertools.chain.from_iterable(
            dist_pair)))


class QueueCluster(object):
    """
    Object to store a cluster while it is in the queue. This supports the
    bisecting kmeans clustering. In that algorithm, a cluster in the queue
    can either be processed further (broken down into 2 smaller clusters) or
    stored.
    """
    def __init__(self, data):
        """Performs feature extraction / dimensionality reduction on the raw
        article data that makes up this cluster for further processing later.
        The data is converted into LSI space which is necessary for Kmeans
        clustering.

        Args:
            data (list): Articles in raw format
        """
        feat_extract = ExtractFeatures()
        feat_extract.load_data(data)
        feat_extract.pre_processing()
        feat_extract.transform_corpus()
        feat_extract.model_lsi()

        self.data = data
        self.original_corpus = feat_extract.corpus_lsi
        self.size = len(self.data)
        self.corpus = [[b for (a, b) in doc] for doc in self.original_corpus]
        self.corpus = np.asarray(self.corpus)


class BigClamArticle(BisectingKmeans):
    """Performs clustering using overlapping community detection
    """
    def __init__(self, data, coms):
        """Refer to superclass for details

        Args:
            data (list): Refer to superclass
            coms (int): Number of communities to find
        """
        super(BigClamArticle, self).__init__(data)
        self.coms = coms

    def compute(self):
        """Prepares data for community detection algorithm, the detected
        communities are then saved to bigclam_cluster. Each detected
        community represents a single cluster and is stored as a
        ComputedCluster.
        """
        cur_item = self.cluster_queue.get()
        cur_cluster = cur_item[1]

        self.original_corpus = cur_cluster.corpus
        original_article = [x[5] for x in cur_cluster.data]
        for i, id in enumerate(original_article):
            self.original_article_pos[id] = i

        if VERBOSE:
            print cur_cluster.data

        bigclam_cluster = snap_cluster_lib.BigClamArticleCluster(
            cur_cluster.data, self.coms)

        for item in bigclam_cluster.computed:
            self.computed_clusters.append(ComputedCluster(item))


if __name__ == '__main__':
    pass
