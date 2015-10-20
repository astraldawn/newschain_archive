__author__ = 'intern'

# Data format
# title / section / subsection / time / word count / summary / author

import datetime
from timeit import default_timer as timer
import sqlite3 as lite
import json

import utils
import analyse

if __name__ == '__main__':
    # utils.log_to_file(__file__)
    start_timer = timer()
    print "Feature extraction + data processing to get top articles"

    # Query for week
    # feat_extract.load_data(utils.load_nyt("World", "2014-01-01",
    #                                       "2016-01",
    #                                       "Vladimir Putin Russia Ebola"))

    # feat_extract.load_data(utils.load_nyt(start_date="2014-12-19",
    #                                       end_date="2015-01-01"))

    start_date = datetime.date(2013, 4, 8)
    start_date = datetime.date(2015, 7, 5)
    end_date = datetime.date(2015, 7, 5)
    # end_date = datetime.date(2015, 1, 4)
    date_inc = 14
    increment = datetime.timedelta(days=+date_inc)
    articles_lost = 0
    output_file = "top_articles/dump_to_file_" + utils.get_time()
    f = open(output_file, "a")
    all_clusters = []

    while start_date <= end_date:
        print "Week from: " + start_date.isoformat() + \
              " to " + (start_date + datetime.timedelta(days=date_inc - 1
                                                        )).isoformat()

        start = start_date.isoformat()
        end = (start_date + datetime.timedelta(days=date_inc - 1)).isoformat()

        with utils.stdout_redirect(f):
            feat_extract = analyse.ExtractFeatures()
            feat_extract.load_data(utils.load_nyt(start_date=start,
                                                  end_date=end,
                                                  keywords="China"))
            feat_extract.pre_processing()
            feat_extract.transform_corpus()
            feat_extract.model_lsi()

            cluster = analyse.Cluster(feat_extract.data,
                                      feat_extract.corpus_lsi, 0)
            cluster.init_agglo()
            cluster.compute_cluster()

        print "Number of articles for week: " + str(len(cluster.data) -
                                                    cluster.articles_lost)
        print "Number of clusters: " + str(len(cluster.computed_clusters))
        print "Number of articles from clusters: " \
              + str(sum([x.size for x in cluster.computed_clusters]))
        print
        articles_lost += cluster.articles_lost
        start_date += datetime.timedelta(days=7)
        all_clusters += [x for x in cluster.computed_clusters]

    print "Total articles lost: " + str(articles_lost)

    # Code to store the stuff into a database
    con = lite.connect("../data/NewYT_clustered.db")

    with con:
        cur = con.cursor()
        cur.execute("DROP TABLE IF EXISTS clusters")
        cur.execute(
            "CREATE TABLE clusters("
            "content TEXT,"
            "n_articles INT,"
            "first_article_date DATETIME,"
            "last_article_date DATETIME,"
            "articles_id TEXT,"
            "articles_date TEXT,"
            "articles_length TEXT,"
            "id INT PRIMARY KEY"
            ")"
        )

        cur_id = 0
        for cluster in all_clusters:
            insert_tuple = (
                cluster.content,
                cluster.size,
                cluster.articles_date[0],
                cluster.articles_date[-1],
                json.dumps(cluster.articles_id),
                json.dumps(cluster.articles_date),
                json.dumps(cluster.articles_length),
                cur_id
            )
            cur_id += 1
            cur.execute("INSERT INTO clusters VALUES (?,?,?,?,?,?,?,?)",
                        insert_tuple)

    end_timer = timer()
    print "Time taken: " + str(end_timer - start_timer)

    # feat_extract.load_data(utils.load_nyt(start_date="2014-12-19",
    #                                       end_date="2014-12-26"))
    # feat_extract.word_freq()

    '''
    Use agglomerative clustering
    The clusters have nicer size (smaller)
    '''

    # cluster = analyse.Cluster(feat_extract.get_data(),
    #                           feat_extract.get_corpus_lsi(),0)
    # cluster.init_agglo(n_cluster_percent=10)
    # cluster.compute_cluster()
    # cluster.dump_to_file()
    # cluster.init_kmeans(n_cluster_percent=10)
    # cluster.test()
