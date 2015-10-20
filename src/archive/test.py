__author__ = 'intern'

from testclass import *


def main():
    """

    :rtype : object
    """
    print "Hello world"
    print "blah blah"

    newnum = Num(5, 6)
    print newnum.get_a()
    print newnum.get_b()
    print newnum  # __str___
    newnum2 = BetterNum(5, 6)
    print newnum2.get_a()
    print newnum2.get_b()
    print newnum2.get_c()
    print newnum2


if __name__ == "__main__":
    main()
