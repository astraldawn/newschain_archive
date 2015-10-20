__author__ = 'intern'


class Num(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def get_a(self):
        return self.a

    def get_b(self):
        return self.b

    def __str__(self):
        return "%s a, %s b" % (self.a, self.b)


class BetterNum(Num):
    def __init__(self, a, b):
        Num.__init__(self, a, b)

    def get_c(self):
        return self.a + self.b

    def __str__(self):
        return "betternum"
