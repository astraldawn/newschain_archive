__author__ = 'Mark Lee'

import utils

if __name__ == '__main__':
    print "Data cleaning"
    # utils.log_to_file(__file__)
    file_list = utils.FileList("../data")
    # file_list.display_all()
    # file_list.display_csv()
    # file_list.csv_combine_nyt_full()
    # file_list.csv_small_nyt()
    # file_list.csv_combine_guardian()

    # Do not call this function unless updating DB
    file_list.sqlite_build_nyt_full()

    # Test function for the DB
    # sqlite_test()  # Test function for sql

    # Loading from the DB
    data = utils.load_nyt("World", "2014-01", "2015-07",
                          "Vladimir Putin Russia Ebola")
