from collections import OrderedDict


class Metric(OrderedDict):

    def __gt__(self, other):
        return self[self.cmp_key] > other[self.cmp_key]

    def __eq__(self, other):
        return self[self.cmp_key] == other[self.cmp_key]

    def __lt__(self, other):
        return self[self.cmp_key] < other[self.cmp_key]

    def __ge__(self, other):
        return self[self.cmp_key] >= other[self.cmp_key]

    def __le__(self, other):
        return self[self.cmp_key] <= other[self.cmp_key]

    def set_cmp_key(self, key):
        self.cmp_key = key


class Evaluator(object):
    def __init__(self):
        pass

    def eval(self, res_file):
        """
        This should return a dict with keys of metric names, values of metric values.

        Arguments:
            res_file (str): file that holds classification results
        """
        raise NotImplementedError

    @staticmethod
    def add_subparser(self, name, subparsers):
        raise NotImplementedError

    @staticmethod
    def from_args(cls, args):
        raise NotImplementedError
