prototype.optimizer
===============================

.. toctree::
   :maxdepth: 2

SGD
----------

Consistent with PyTorch official implementation
`SGD <https://pytorch.org/docs/stable/optim.html#torch.optim.SGD>`_.

RMSprop
----------

Consistent with PyTorch official implementation
`RMSprop <https://pytorch.org/docs/stable/optim.html#torch.optim.RMSprop>`_.

Adam
----------

Consistent with PyTorch official implementation
`Adam <https://pytorch.org/docs/stable/optim.html#torch.optim.Adam>`_.

LARS
----------

.. autoclass:: prototype.optimizer.lars.LARS
    :members: __init__, step

FP16SGD
--------------------

.. autoclass:: prototype.optimizer.fp16_optim.FP16SGD
    :members: __init__

FP16RMSprop
--------------------

.. autoclass:: prototype.optimizer.fp16_optim.FP16RMSprop
    :members: __init__
