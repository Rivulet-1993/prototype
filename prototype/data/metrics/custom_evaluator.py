import json
from .evaluator import Evaluator, Metric


class CustomMetric(Metric):
    def __init__(self, metric_dict={}):
        pass


class CustomEvaluator(Evaluator):
    def __init__(self):
        super(CustomEvaluator, self).__init__()

    def load_res(self, res_file):
        res_dict = {}
        with open(res_file) as f:
            lines = f.readlines()
        for line in lines:
            info = json.loads(line)
            for key in info.keys():
                res_dict[key].append(info[key])

        return res_dict

    def calculate_recall_tpr(self):
        pass

    def calculate_fpr_from_threshold(self):
        pass

    def eval(self, res_file):
        pass
