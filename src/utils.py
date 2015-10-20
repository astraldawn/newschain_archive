"""
Utility library

Contains functions for
    - Loading of articles
    - Loading of clusters
    - Creating the database of articles
    - Initial cleaning of the raw csv data from crawler
"""
__author__ = 'Mark Lee'

from os import listdir
from os.path import isfile, join
import csv
import time
import logging
import sqlite3 as lite
import sys
import datetime
import json
from contextlib import contextmanager


class FileList(object):
    """Object that hold lists of files and creates a database from them.
    """

    def __init__(self, folder_path):
        """Builds a list of files (file_list) and csv files (csv_list) from a
        given folder path.

        Args:
            folder_path (str): Path of the folder to build file_list and
                               csv_list from
        """
        self.file_list = sorted([f for f in listdir(folder_path) if isfile(join(
            folder_path,
            f))])
        self.csv_list = [f for f in self.file_list if '.csv' in f]
        pass

    def display_all(self):
        """Prints file_list.
        """
        print self.file_list

    def display_csv(self):
        """Prints csv_list.
        """
        print self.csv_list

    def get_csv(self):
        """
        Returns
            list: A list of csv files
        """
        return self.csv_list

    def find_csv(self, key):
        """Used to load csv files for a specific database. For example,
        find_csv("NYT") will return every csv file with NYT in its file name.

        Args:
            key (str): A keyword which every file's name must contain

        Returns:
            list: A list of csv files where every file's name contains the key
        """
        return [f for f in self.csv_list if key in f]

    def sqlite_build_nyt_full(self):
        """Construct database for NYT.

        Format
            - name (text) PRIMARY KEY
            - section (text)
            - date (datetime)
            - wordcnt (int)
            - summary (text)
            - id (int)

        name, rather than id is the primary key for this database to remove
        articles with duplicate names. All the text fields involve a conversion
        to unicode, with any errors being ignored. This does lead to some
        weird characters in the processed text, although this should not have
        an impact on the results.
        """
        logging.info('COMBINE NYT FULL')

        # Modify this to change the database output
        con = lite.connect("../data/NewYT_all.db")

        cnt_world = 0
        cnt_us = 0

        with con:
            cur = con.cursor()

            cur.execute("DROP TABLE IF EXISTS articles")
            cur.execute(
                "CREATE TABLE articles("
                "name TEXT PRIMARY KEY,"
                "section TEXT,"
                "date DATETIME,"
                "wordcnt INT,"
                "summary TEXT,"
                "id INT"
                ")"
            )

            csv_search = self.find_csv("NYT")
            current_id = 0
            for f in csv_search:
                buf = load_csv("../data/" + f)

                for row in reversed(buf):
                    row_zero_conv = unicode(row[0], errors='ignore')
                    row_one_conv = unicode(row[1], errors='ignore')
                    row_five_conv = unicode(row[5], errors='ignore')
                    insert_tuple = (
                        row_zero_conv,
                        row_one_conv,
                        row[3],
                        row[4],
                        row_five_conv,
                        current_id
                    )

                    if "IHT" in row_zero_conv:
                        print "Removing IHT"
                        continue
                    # print insert_tuple
                    try:
                        cur.execute(
                            "INSERT INTO articles VALUES (?,?,?,?,?,?)",
                            insert_tuple)
                        if row[1] == "World":
                            cnt_world += 1
                        elif row[1] == "U.S.":
                            cnt_us += 1
                        current_id += 1
                    except lite.IntegrityError:
                        pass

        logging.info('WORLD CNT %s', str(cnt_world))
        logging.info('US CNT %s', str(cnt_us))


def sqlite_test():
    """Tests if the database has been loaded correctly.
    """
    con = lite.connect("../data/NewYT_all.db")
    with con:
        cur = con.cursor()
        # cur.execute("SELECT * FROM articles")
        # res = cur.fetchall()
        # for row in res:
        #     print "DB call: " + str(row)
        cur.execute("SELECT COUNT(*) FROM articles WHERE section='World'")
        res = cur.fetchall()
        print "COUNT: " + str(res)
        cur.execute("SELECT * FROM articles LIMIT 10")
        res = cur.fetchall()
        print str(res)


@contextmanager
def stdout_redirect(stream):
    """ Redirects stdout to the given stream.
    """
    old_stdout = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout


def load_csv(file_name):
    """Loads the CSV file with the given filename.

    Args:
        file_name (str): Name of a csv file

    Returns
        list: Each item is a single row in the provided CSV file
    """
    infile = open(file_name, "rb")
    reader = csv.reader(infile)
    buf = []
    for row in reader:
        buf.append(row)
    infile.close()
    return buf


def load_nyt(section_name="World", start_date="2014-01-01",
             end_date="2015-01-01",
             keywords=""):
    """Loads data from SQLite DB

    SQLite DB requires the following format
        - name (text)
        - section (text)
        - date (datetime)
        - wordcnt (int)
        - summary (text)
        - id (int)

    Args
        section_name (str): Name of a section, either "World" or "US".
                            May also be blank to load from both.
        start_date (date): Articles loaded will be published after this date
        end_date (date): Articles loaded will be published before this date
        keywords (str): Keywords, each separated by a single space to
                        restrict the articles loaded. e.g. "china israel"

    Returns:
        list: Each item in the list is an article loaded from the SQLite DB
    """
    print "CONNECTING TO NYT DB"
    res = []
    con = lite.connect("../data/NewYT_all.db")
    keyword_list = keywords.split()
    cnt = 0
    with con:
        cur = con.cursor()
        cur.execute(
            "SELECT * FROM articles WHERE section = '" + section_name +
            "' AND date >= '" + start_date +
            "' AND date <= '" + end_date + "'")
        # # cur.execute(
        # #      "SELECT * FROM articles WHERE section = '" + section_name + "'"
        # # )
        # cur.execute("SELECT * FROM articles");
        while True:
            row = cur.fetchone()
            if row is None:
                break
            info = row[0] + row[4]
            if len(info) < 20:
                continue
            flag = 0
            for word in keyword_list:
                if word in info:
                    flag = 1
            if flag or keywords == "":
                cnt += 1
                res.append(list(row))

    print str(cnt) + " articles loaded"
    logging.info("DATA SOURCE: NYT DB - %s articles loaded", str(cnt))
    logging.info("SECTION: %s", section_name)
    logging.info("START DATE: %s", str(start_date))
    logging.info("END DATE: %s", str(end_date))
    logging.info("KEYWORDS: %s", keywords)
    return res


def load_nyt_by_article_id(article_id):
    """Loads an article from SQLite DB

    SQLite DB requires the following format
        - name text,
        - section text,
        - date datetime,
        - wordcnt int,
        - summary text,
        - id int

    Args:
        article_id (int): The ID of an article

    Returns:
        list: A list with a single item, the article if it exists in the
        database. Otherwise a blank list is returned
    """
    con = lite.connect("../data/NewYT_all.db")
    with con:
        cur = con.cursor()
        cur.execute(
            "SELECT * FROM articles WHERE id = '" + article_id + "'")
        res = cur.fetchall()
        return res


def load_nyt_clusters(start_date=None, end_date=None,
                      db_name="NewYT_clustered.db"):
    """Loads clusters from a specified database.

    Args:
        start_date (datetime): The earliest published article in the cluster
                               must be published on or after this date
        end_date (datetime): The earliest published article in the cluster
                             must be published on or before this date
        db_name (str): The name of the database of clusters to load from

    Returns
        list: A list containing the clusters that were selected by the query.
              Each row in the list is a single cluster
    """
    print "Connecting to NYT clusters"
    res = []
    con = lite.connect("../data/" + db_name)
    if start_date is None or end_date is None:
        start_date = datetime.date(2011, 12, 30).isoformat()
        end_date = datetime.date(2016, 1, 1).isoformat()
    with con:
        cur = con.cursor()
        cur.execute("SELECT * FROM clusters WHERE first_article_date >='" +
                    start_date +
                    "' AND first_article_date <= '" + end_date + "'"
                    )
        while True:
            row = cur.fetchone()
            if row is None:
                break
            res.append(list(row))
    return res


def get_time():
    """Gets current system time

    Returns:
        str: Current system time
    """
    return time.strftime("%Y%m%d_%H:%M:%S")


def log_to_file(program_name):
    """Prepares configuration for logging for a specific program.

    Args:
        program_name (str): Name of a program (usually a .py file)
    """
    outfile = get_time() + "_" + program_name + ".log"
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                        level=logging.DEBUG, filename=outfile)


def create_nyt_cluster_database(database_name, all_clusters):
    """Creates a database to store computed clusters for subsequent chaining.

    Args:
        database_name (str): Name of the database to store computed clusters to
        all_cluster (list): A list of computed clusters
    """
    con = lite.connect("../data/" + database_name)

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
            "diameter INT,"
            "radius INT,"
            "nearest_article_id INT,"
            "nearest_article_dist INT,"
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
                cluster.diameter,
                cluster.radius,
                cluster.closest_article_id,
                cluster.closest_article_distance,
                cur_id
            )
            cur_id += 1
            cur.execute("INSERT INTO clusters VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        insert_tuple)


if __name__ == '__main__':
    pass
