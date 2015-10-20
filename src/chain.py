__author__ = "Mark Lee"

from timeit import default_timer as timer
import datetime

import utils
import sim_lib
import snap_cluster_lib

if __name__ == "__main__":
    # utils.log_to_file(__file__)
    start_timer = timer()
    print "Loading from SQLite DB"

    start_date = datetime.date(2012, 7, 1).isoformat()
    end_date = datetime.date(2016, 1, 1).isoformat()
    data = utils.load_nyt_clusters("2013-01-01",
                                   "2016-01-01",
                                   db_name="NewYT_clustered_china_bisect_final.db")

    similarity = sim_lib.SimMatrix(
        data,
        # utils.load_nyt(start_date=start_date,end_date=end_date),
        refresh=False,
        clusters=True  # Adjust this if the data range is changed
    )

    # for i in range(0,1):
    # similarity.dijkstra(4)

    # similarity.addition(10)
    # for i in range(0,10):
    #     similarity.addition([i])

    cluster_id = similarity.keyword_query("south china sea",
                                          n_cluster=30)
    print cluster_id
    clusters = [similarity.data[x] for x in cluster_id]
    bigclamcluster = snap_cluster_lib.BigClamChainCluster(clusters)
    bigclamcluster.find_community(opt_com=6, threshold=0.001)
    words, cluster_id_list = bigclamcluster.print_community()

    for i in range(0, len(cluster_id_list)):
        bigclamcluster.print_cluster(cluster_id_list[i])

    end_timer = timer()
    print "Time taken: " + str(end_timer - start_timer)
