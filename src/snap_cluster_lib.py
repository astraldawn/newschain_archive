"""
Class to perform overlapping community detection using bigclam.
"""
__author__ = "Mark Lee"

from snap import PUNGraph, TIntIntVV, TAGMFast
from gensim import parsing

import analyse
import utils
import sys


class BigClamCluster(object):
    def __init__(self):
        """Initialise the BigClamCluster object. Creates objects for feature
        extraction and the bipartite graph.
        """
        self.edges = {}
        self.FeatExtract = analyse.ExtractFeatures()
        self.graph = PUNGraph.New()

    def create_graph(self, cutoff=0.1):
        """Creates the bipartite graph necessary for overlapping community
        detection. It assumes that the necessary edges between the words and
        the clusters have already been generated.
        """
        print "---- CREATING GRAPH -----"
        # Remove low weight edges
        max_freq = 0
        tmp = {}

        # This does nothing for the article / cluster chaining
        for (_, freq) in self.edges.iteritems():
            if freq > max_freq:
                max_freq = freq
        print "MAX FREQ "
        print max_freq, max_freq * cutoff
        for (k, v) in self.edges.iteritems():
            if v < max_freq * cutoff:
                print "Item removed"
                print k, v, self.FeatExtract.dictionary.__getitem__(k[0]), \
                    self.FeatExtract.dictionary.__getitem__(k[1])
            else:
                tmp[k] = v

        edges = tmp
        for ((u, v), _) in edges.iteritems():
            try:
                self.graph.AddNode(u)
            except:
                pass

            try:
                self.graph.AddNode(v)
            except:
                pass

            self.graph.AddEdge(u, v)

        print "Nodes %d, Edges %d" % (self.graph.GetNodes(),
                                      self.graph.GetEdges())

        # Print all the edges
        # for ((u,v),freq) in edges.iteritems():
        #     print self.FeatExtract.dictionary.__getitem__(u), \
        #         self.FeatExtract.dictionary.__getitem__(v), freq

    def find_community(self, num_threads=4, min_com=10, max_com=100,
                       div_com=5, step_alpha=0.3, step_beta=0.3, opt_com=-1,
                       threshold=0.001):
        """Finds communities using bigclam. Each of these communities will
        represent a chain of articles. The communities are stored in
        self.EstCmtyVV.

        Args:
            num_threads: number of threads (parallel)
            min_com: minimum number of communities
            max_com: maximum number of communities
            div_com: how many trials for number of communities
            step_alpha: alpha for backtracking line search
            step_beta: beta for backtracking line search
            opt_com: number of communities to detect
        """
        print "----- Finding community -----"
        self.EstCmtyVV = TIntIntVV()
        print self.EstCmtyVV
        RAGM = TAGMFast(self.graph, 10, 10)
        if opt_com is -1:
            opt_com = RAGM.FindComsByCV(num_threads, max_com, min_com,
                                        div_com, "", step_alpha, step_beta)
        RAGM.NeighborComInit(opt_com)
        print "----- Coms initialised -----"
        try:
            if self.graph.GetEdges() < 5000:
                print "----- Single core grad ascent -----"
                RAGM.MLEGradAscent(threshold, 1000 * self.graph.GetNodes(),
                                   "", step_alpha, step_beta)
            else:
                print "----- Parallel grad ascent -----"
                RAGM.MLEGradAscentParallel(threshold, 1000, num_threads, 200,
                                           "",
                                           step_alpha, step_beta)
        except:
            print "Error"
        print "----- Grad ascent success -----"
        RAGM.GetCmtyVV(self.EstCmtyVV)

"""
# Deprecated class that performs word clustering.

class BigClamWordCluster(BigClamCluster):
    def __init__(self, init_data):
        super(BigClamWordCluster, self).__init__()
        self.init_data = init_data
        self.FeatExtract.load_data(self.init_data)
        self.FeatExtract.pre_processing()
        self.FeatExtract.transform_corpus()
        self.data = self.FeatExtract.data
        self.computed = []
        self.create_edges()

    def create_edges(self):
        # Construct the graph
        for doc in self.FeatExtract.corpus_transformed:
            for (u, _) in doc:
                for (v, _) in doc:
                    if u == v:
                        continue
                    if u > v:
                        (u, v) = (v, u)
                    if (u, v) in self.edges.keys():
                        self.edges[(u, v)] += 1
                    else:
                        self.edges[(u, v)] = 1

        self.create_graph()

    def print_community(self):
        print
        # print EstCmtyVV
        # print len(EstCmtyVV)

        # Create the computed clusters here
        for comm in self.EstCmtyVV:
            words = [self.FeatExtract.dictionary.__getitem__(word) for word in
                     comm]
            self.computed.append(ComputedWordCluster(words))
            # for row in self.data:
            #     print row[0]
"""


class BigClamLineCluster(BigClamCluster):
    def __init__(self, data):
        super(BigClamLineCluster, self).__init__()
        self.load(data)

    def load(self, data):
        """Loads the data into the object for later use. It performs feature
        extraction to build the dictionary and transforms the corpus into
        log-entropy vectors. This is so that we can find the top 50 words for
        each cluster.

        Args
            data (list): A list of clusters, where each item is a cluster
        """
        self.clusters = data
        self.FeatExtract.load_data(self.clusters)
        self.FeatExtract.pre_processing_line()
        self.FeatExtract.transform_corpus()
        self.data = self.FeatExtract.data
        self.computed = []
        self.cluster_const = 500000
        self.create_edges()

    def create_edges(self):
        """Builds the edges of the bipartite graph using the top 50 tf-idf words
        from each cluster. Once the edges are built, the graph can be created.
        """
        for doc, u in zip(self.FeatExtract.corpus_transformed,
                          range(0, len(self.FeatExtract.corpus_transformed))):

            # Top 50 tf-idf
            doc = sorted(doc, key=lambda item: -item[1])

            u += self.cluster_const
            for (v, _) in doc[:50]:
                if (u, v) in self.edges.keys():
                    self.edges[(u, v)] += 1
                else:
                    self.edges[(u, v)] = 1

        self.create_graph()

    def print_community(self):
        """Prints all the communities produced by overlapping community
        detection.
        """
        # print
        # print RAGM.Likelihood

        # EstCmtyVV contains the community
        # Each item in EstCmtyVV contains node ID
        # Each node represents a word in the community
        print
        # print EstCmtyVV
        # print len(EstCmtyVV)

        # Create the computed clusters here
        for comm in self.EstCmtyVV:
            tmp = []
            for item in comm:
                if item < self.cluster_const:
                    tmp.append(self.FeatExtract.dictionary.__getitem__(item))
                else:
                    tmp.append(item)
                    # tmp.append(str(self.data[item - self.cluster_const]))
            print tmp
            # for row in self.data:
            #     print row[0]


class BigClamChainCluster(BigClamLineCluster):
    def __init__(self, data):
        super(BigClamChainCluster, self).__init__(data)

    def load(self, data):
        """Loads the data into the object for later use. It performs feature
        extraction to build the dictionary and transforms the corpus into
        log-entropy vectors. This is so that we can find the top 50 words for
        each cluster.

        Args
            data (list): A list of clusters, where each item is a cluster
        """
        print "----- Chain cluster load -----"
        self.clusters = data
        self.loaded_data = [row[0] for row in self.clusters]
        self.FeatExtract.load_data(self.loaded_data)
        self.FeatExtract.pre_processing_line()
        self.FeatExtract.transform_corpus()
        self.data = self.FeatExtract.data
        self.computed = []
        self.cluster_const = 500000
        self.create_edges()

    def print_community(self):
        """Prints all the communities produced by overlapping community
        detection.
        """
        # print
        # print RAGM.Likelihood

        # EstCmtyVV contains the community
        # Each item in EstCmtyVV contains node ID
        # Each node represents a word in the community
        print
        # print EstCmtyVV
        # print len(EstCmtyVV)

        words = []
        cluster_id_list = []
        # Create the computed clusters here
        for comm in self.EstCmtyVV:
            tmp_c = []
            tmp_w = []
            for item in comm:
                if item < self.cluster_const:
                    tmp_w.append(self.FeatExtract.dictionary.__getitem__(item))
                    pass
                else:
                    tmp_c.append(item)
                    # tmp.append(str(self.data[item - self.cluster_const]))
            words.append(tmp_w)
            cluster_id_list.append(tmp_c)
            # for row in self.data:
            #     print row[0]
        return words, cluster_id_list

    def print_cluster(self, cluster_id_list):
        """Prints the clusters in the given cluster list and performs
        coherence calculation.

        Args:
            cluster_id_list (list): A list of clusters that make up a
                                    coherent chain.
        """
        cluster_content = []
        print "----- Cluster -----"
        for cid in sorted(cluster_id_list):
            cur_cluster = self.clusters[cid - self.cluster_const]
            print cur_cluster[7]
            print cur_cluster[8]
            print utils.load_nyt_by_article_id(str(cur_cluster[9]))
            print "Distance: ", cur_cluster[10]
            article_list = cur_cluster[4].strip("[]").split(", ")
            tmp = ""
            for article_id in article_list:
                res = utils.load_nyt_by_article_id(article_id)
                print res[0][0] + " # " + res[0][4] + " # " + res[0][2]
                tmp += res[0][0] + " " + res[0][4] + " "
            cluster_content.append(parsing.preprocess_string(
                str.lower(str(tmp)),
                filters=[parsing.strip_tags,
                         parsing.strip_punctuation,
                         parsing.strip_multiple_whitespaces,
                         parsing.strip_numeric,
                         parsing.remove_stopwords]
            )
            )
            print

        # Coherence calculation
        cluster_content = [list(set(x)) for x in cluster_content]

        coherence = sys.maxint
        for i in range(0, len(cluster_content) - 1):
            cnt = 0
            tmp = []
            for word in cluster_content[i]:
                if word in cluster_content[i + 1]:
                    tmp.append(word)
                    cnt += 1
            print tmp
            coherence = min(coherence, cnt)

        print "Coherence: " + str(coherence)


class BigClamArticleCluster(BigClamLineCluster):
    def __init__(self, data, coms=40):
        """Modified BigClamLineCluster which is used to cluster articles
        together.

        Args:
            data (list): A list where each item is a single article
            coms (int): The number of communities to be found using
                        overlapping community detection
        """
        super(BigClamArticleCluster, self).__init__(data)
        self.load(data)
        self.find_community(opt_com=coms)
        self.print_community()

    def print_community(self):
        """Prints all the communities produced by overlapping community
        detection.
        """
        print "----- Article cluster print -----"
        for comm in self.EstCmtyVV:
            tmp = []
            for item in comm:
                if item < self.cluster_const:
                    pass
                    # tmp.append(self.FeatExtract.dictionary.__getitem__(item))
                else:
                    tmp.append(self.data[item - self.cluster_const])
                    # tmp.append(str(self.data[item - self.cluster_const]))
            self.computed.append(tmp)
            # for row in self.data:
            #     print row[0]


"""
# Deprecated class

class ComputedWordCluster(object):
    def __init__(self, words):
        self.words = " ".join(words)
        self.articles = []
        self.date = []

    def __str__(self):
        return self.words
"""


if __name__ == "__main__":
    pass
