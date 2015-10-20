.. News Chain documentation master file, created by
   sphinx-quickstart on Tue Aug 25 11:11:00 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

News chain project documentation
================================
Please read the following information before proceeding


.. _contents:

Contents
=========
	- :ref:`intro`
	- :ref:`clean_data`
	- :ref:`create_cluster`
	- :ref:`create_chain`
	- :ref:`indexes`

.. _intro:

Introduction
=============
Requirements
	- Language: Python 2.7.6
	- Additional libraries: gensim, numpy, scikit-learn, snap-py, sqlite3

Structure of the project
	- /data: Contains all of the data files used by the program, the raw .csv files and the SQL databases
	- /docs: Contains all the documentation for the project	
	- /src: Contains source files as well as other folders that hold save indexes and dictionaries

**The programs in this project were written for use with a specific data format. It is possible to run this project with different sets of data. However, this would involve rewriting functions that deal with the saving / loading of the raw data.**

Module information

.. toctree::
   :maxdepth: 2

   analyse
   sim_lib
   snap_cluster_lib
   utils

Return to :ref:`top<contents>`

.. _clean_data:

Cleaning of data
===================
#. Crawl raw data from the Internet and save it in the following format (.csv)
#. Raw article format
	- Title (str): Title of the article
	- Section (str): Section in which the article was published
	- Region (str): Region which the events described in the article occured in
	- Time (datetime): Time of publication of the article
	- Word count (int): Number of words in the article
	- Summary (str): A short summary of the article
#. The program minimally requires title, section, time and summary. The region and word count fields can be filled
   with " " and 0 respectively.
#. The details of how the program builds the database can be found in :func:`utils.FileList.sqlite_build_nyt_full`
#. The database is then stored in **../data/NewYT_all.db**. This can be modified by changing the con variable in :func:`utils.FileList.sqlite_build_nyt_full` 

**Example program**: ../src/data_clean.py

Return to :ref:`top<contents>`

.. _create_cluster:

Creating article clusters
==========================
From this point onwards, we will discuss how to run the programs in the context of finding relevant information about events that occured in the South China Sea.

#. Load the cleaned data from **../data/NewYT_all.db** using :func:`utils.load_nyt`. We load all the data in the range, from 8 Apr 2013 to 5 July 2015. We set the keyword field to "China", while this restricts the search space significantly, it still allows for some breadth in the discussion.
#. Set the step size to 400. This can be adjusted to increase or decrease the number of clusters generated per step.
#. Cluster the data for each step using one of the provided clustering method, either :class:`bisecting kmeans<analyse.BisectingKmeans>` or :class:`overlapping community detection<analyse.BigClamArticle>`. Assuming we use the former, we then call :func:`compute<analyse.BisectingKmeans.compute>`. This function performs clustering and has parameters set to ensure that the cluster size is not too large. The details of this can be found in the documentation for :doc:`analyse<analyse>`.
#. After the clusters have been computed, call :func:`find_computed_cluster_metrics<analyse.BisectingKmeans.find_computed_cluster_metrics>` to compute metrics for each cluster. The metrics that are computed include: radius, diameter and the closest article to the centroid of the cluster.

**Example program**: ../src/top_articles_new.py

Return to :ref:`top<contents>`

.. _create_chain:

Cluster chaining
=================
#. Load all the cluster data from **../data/NewYT_clustered_china_bisect_final.db** into the variable data using :func:`utils.load_nyt_clusters`. As we are using the clusters generated in the previous step, all the clusters will contain articles with content pertaining to China.
#. Prepare the data for similarity query by initialising a new :class:`SimMatrix<sim_lib.SimMatrix>` object and loading the data into it. It is important that for the initial run, **refresh = True**. The indexes are saved in ../src/tmp. **To load cluster data, the folder must be renamed to tmp_clusters.**
#. Perform a similarity query using :func:`keyword_query<sim_lib.SimMatrix.keyword_query>`. The keywords that we use for the query are "south china sea" and we also set the number of similar clusters to be returned to be 30 (n_cluster = 30). The results are stored in cluster_id.
#. From cluster_id, we then extract the data for each of the clusters and load it into a new :class:`BigClamChainCluster<snap_cluster_lib.BigClamChainCluster>`.
#. Overlapping community detection is then performed by calling :func:`find_community<snap_cluster_lib.BigClamCluster.find_community>`. It is possible to set the number of desired communities by adjusting the opt_com variable.
#. The chains found via overlapping community detection can then be viewed by calling :func:`print_community<snap_cluster_lib.BigClamChainCluster.print_community>`.

**Example program**: ../src/chain.py

Return to :ref:`top<contents>`

.. _indexes:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Return to :ref:`top<contents>`
