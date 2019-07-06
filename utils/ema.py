import torch
from collections import OrderedDict
from utils.misc import get_logger

class EMA(object):
    def __init__(self, model, decay, copy_init=False, use_double=False, inner_T=1):
        self.ema_state_dict = OrderedDict()
        self.logger = get_logger(__name__)
        self.logger.info('EMA: decay={}, copy_init={}, use_double={}, inner_T={}'.format(decay, copy_init, use_double, inner_T))
        self.use_double = use_double
        self.inner_T = inner_T
        self.decay = decay
        if self.inner_T > 1:
            self.decay = self.decay ** self.inner_T
            self.logger.info('EMA: effective decay={}'.format(self.decay))
        state_dict = model.state_dict()
        if copy_init:
            if self.use_double:
                for k, v in state_dict.items():
                    self.ema_state_dict[k] = v.data.clone().double()
            else:
                for k, v in state_dict.items():
                    self.ema_state_dict[k] = v.data.clone().float()
        else:
            if self.use_double:
                for k, v in state_dict.items():
                    self.ema_state_dict[k] = torch.zeros_like(v).double()
            else:
                for k, v in state_dict.items():
                    self.ema_state_dict[k] = torch.zeros_like(v).float()

    def step(self, model, curr_step=None):
        if curr_step is None:
            decay = self.decay
        else:
            decay = min(self.decay, (1+curr_step)/(10+curr_step))

        if curr_step % self.inner_T != 0:
            return 

        state_dict = model.state_dict()
        if self.use_double:
            for k, v in state_dict.items():
                self.ema_state_dict[k].mul_(self.decay).add_(1-self.decay, v.double())
        else:
            for k, v in state_dict.items():
                self.ema_state_dict[k].mul_(self.decay).add_(1-self.decay, v.float())

    def load_ema(self, model):
        for k, v in model.state_dict().items():
            tmp = v.data.clone()
            v.data.copy_(self.ema_state_dict[k].data)
            self.ema_state_dict[k].data.copy_(tmp)
            
    def recover(self, model):
        state_dict = model.state_dict()
        for k, v in self.ema_state_dict.items():
            tmp = v.data.clone()
            v.data.copy_(state_dict[k].data)
            state_dict[k].data.copy_(tmp)

    def state_dict(self):
        obj = {}
        for k, v in self.__dict__.items():
            if k not in ['logger']:
                obj[k] = v
        return obj

    def load_state_dict(self, state):
        for k, v in state.items():
            self.__dict__[k] = v

        self.logger = get_logger(__name__)
        self.logger.info('loading ema, keys={}'.format(self.__dict__.keys()))