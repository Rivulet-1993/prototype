from torch.optim.optimizer import required
from torch.optim import SGD, RMSprop
from linklink.fp16 import FP16_Optimizer


class FP16SGD(FP16_Optimizer):

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, loss_scale='dynamic', verbose=False):

        optimizer = SGD(params, lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)

        dynamic_loss_scale = False
        static_loss_scale = 1.0
        if loss_scale == 'dynamic':
            dynamic_loss_scale = True
        else:
            static_loss_scale = float(loss_scale)

        super(FP16SGD, self).__init__(optimizer,
                                      static_loss_scale=static_loss_scale,
                                      dynamic_loss_scale=dynamic_loss_scale,
                                      verbose=verbose)


class FP16RMSprop(FP16_Optimizer):

    def __init__(self, params, lr=required, alpha=0.99, eps=1e-08,
                 weight_decay=0, momentum=0, centered=False, loss_scale='dynamic', verbose=False):

        optimizer = RMSprop(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay,
                            momentum=momentum, centered=centered)

        dynamic_loss_scale = False
        static_loss_scale = 1.0
        if loss_scale == 'dynamic':
            dynamic_loss_scale = True
        else:
            static_loss_scale = float(loss_scale)

        super(FP16RMSprop, self).__init__(optimizer,
                                          static_loss_scale=static_loss_scale,
                                          dynamic_loss_scale=dynamic_loss_scale,
                                          verbose=verbose)
