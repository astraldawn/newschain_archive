__author__ = "intern"

from snap import *
import utils
import snap_cluster_lib
from timeit import default_timer as timer
import analyse
step_size = 20
steps = 5
opt_com = 10

if __name__ == "__main__":
    start_timer = timer()
    data = utils.load_nyt(start_date="2013-07-01", end_date="2016-01-01",
                          keywords="Israel")
    # Form the word clusters
    tmp = []
    clusters = []
    for article in data[:step_size * steps]:
        tmp.append(article)
        if len(tmp) == step_size:
            cluster = snap_cluster_lib.BigClamWordCluster(tmp)
            cluster.find_community(opt_com=opt_com)
            cluster.print_community()
            for c in cluster.computed:
                clusters.append(c)
            tmp = []

    # Cluster the word clusters and words
    print [str(x) for x in clusters]
    line_cluster = snap_cluster_lib.BigClamLineCluster(clusters)
    line_cluster.find_community(opt_com=opt_com)
    line_cluster.print_community()


    end_timer = timer()
    print "Time taken: " + str(end_timer - start_timer)
    print "COMPLETE"
    pass
