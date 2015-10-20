__author__ = 'Mark Lee'

import datetime
from timeit import default_timer as timer

import analyse
import utils

if __name__ == '__main__':
    # utils.log_to_file(__file__)
    start_timer = timer()
    print "Feature extraction + data processing to get top articles"

    # Start date
    start_date = datetime.date(2013, 4, 8)

    # End date
    end_date = datetime.date(2015, 7, 5)

    # Step size
    step_size = 400

    articles_lost = 0
    all_clusters = []

    data = utils.load_nyt(start_date=start_date.isoformat(),
                          end_date=end_date.isoformat(),
                          keywords="china")

    output_file = "top_articles_new/dump_to_file_" + utils.get_time()
    f = open(output_file, "a")

    for i in range(0, len(data), step_size):
        # with utils.stdout_redirect(f):

        # cluster = analyse.BisectingKmeans(data[i:i+step_size])
        cluster = analyse.BigClamArticle(data[i:i + step_size], coms=130)

        cluster.compute()
        cluster.find_computed_cluster_metrics()

        # for x in cluster.computed_clusters:
        #     x.display()

        print
        print "Centroid max distance: ", cluster.max_centroid_dist
        print "Number of articles for step: " + str(len(data[i:i + step_size]))
        print "Number of clusters: " + str(len(cluster.computed_clusters))
        print "Number of articles from clusters: " \
              + str(sum([x.size for x in cluster.computed_clusters]))
        print "Cluster size: " + str(
            [x.size for x in cluster.computed_clusters])
        print

        articles_lost += cluster.articles_lost
        all_clusters += [x for x in cluster.computed_clusters]
        # break

    print "Total articles lost: " + str(articles_lost)

    # Code to store the stuff into a database
    utils.create_nyt_cluster_database("NewYT_clustered.db", all_clusters)

    end_timer = timer()
    print "Time taken: " + str(end_timer - start_timer)
