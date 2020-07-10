import json
import yaml
import torch
import numpy as np
from .evaluator import Evaluator, Metric


class ClsMetric(Metric):
    def __init__(self, top1=0., top5=0.):
        self.top1 = top1
        self.top5 = top5

    def __str__(self):
        return f'top1={self.top1} top5={self.top5}'

    def __eq__(self, other):
        return self.top1 == other.top1

    def __ne__(self, other):
        return self.top1 != other.top1

    def __gt__(self, other):
        return self.top1 > other.top1

    def __lt__(self, other):
        return self.top1 < other.top1

    def __ge__(self, other):
        return self.top1 >= other.top1

    def __le__(self, other):
        return self.top1 <= other.top1


class ImageNetEvaluator(Evaluator):
    def __init__(self):
        super(ImageNetEvaluator, self).__init__()
        self.topk = (1, 5)

    def load_res(self, res_file):
        """
        Load results from file.
        """
        res_dict = {}
        with open(res_file) as f:
            lines = f.readlines()
        for line in lines:
            info = json.loads(line)
            for key in info.keys():
                res_dict[key].append(info[key])

        return res_dict

    def eval(self, res_file):
        res_dict = self.load_res(res_file)
        pred = torch.from_numpy(np.array(res_dict['prediction']))
        label = torch.from_numpy(np.array(res_dict['label']))
        filename = res_dict['filename']
        pred = pred.t()
        correct = pred.eq(label.view(1, -1).expand_as(pred))
        res = []
        for k in self.topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / len(filename)))
        metric = ClsMetric(res[0].item(), res[1].item())

        return metric

    @staticmethod
    def add_subparser(name, subparsers):
        subparser = subparsers.add_parser(
            name, help='subcommand for ImageNet of Top-1/5 accuracy metric')
        subparser.add_argument('--config', dest='config', required=True,
                               help='settings of classification in yaml format')
        subparser.add_argument('--res_file', required=True,
                               action='append', help='results file of classification')

        return subparser

    @classmethod
    def from_args(cls, args):
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        kwargs = config['data']['evaluator']['kwargs']

        return cls(**kwargs)
