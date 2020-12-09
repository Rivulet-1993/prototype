快速上手/Get started
===============================

.. toctree::
   :maxdepth: 2

Training
---------
1. Create/Enter directory of one experiment

.. code-block:: bash

    cd /model_zoo_exp/resnet_experiments/resnet50_batch1k_epoch100_coslr_nesterov_wd0.0001

2. Check ``config.yaml`` and ``run.sh`` under your workspace

.. code-block:: bash

    PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 \
    srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 \
    python -u -m prototype.solver.cls_solver --config config.yaml

3. Start training your model

.. code-block:: bash

    sh run.sh Test 16

Testing
---------

1. Check ``config.yaml``, ``run.sh`` and ``checkpoints`` under your workspace
2. Modify your ``config.yaml`` and ``run.sh``

add information of checkpoints into ``config.yaml``

.. code-block:: yaml

    saver:
        print_freq: 10
        val_freq: 100
        save_many: False
        pretrain:
            path: checkpoints/ckpt.pth.tar

add ``--evaluate`` into ``run.sh``

.. code-block:: bash

    PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 \
    srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 \
    python -u -m prototype.solver.cls_solver --config config.yaml --evaluate

3. Start testing your model

.. code-block:: bash

    sh run.sh Test 8

Recovering and Finetuning
--------------------------

1. Check **config.yaml**, **run.sh** and **ckpt** under your workspace

2. If recovering, add information of checkpoints into ``config.yaml``

.. code-block:: yaml

    saver:
        print_freq: 10
        val_freq: 100
        save_many: False
        pretrain:
            path: checkpoints/ckpt.pth.tar

If finetuning, add information and pop modules in ``config.yaml``

.. code-block:: yaml

    saver:
        print_freq: 10
        val_freq: 100
        save_many: False
        pretrain:
            path: checkpoints/ckpt.pth.tar
            ignore:
                key:
                    - optimizer
                    - last_iter
                model:
                    - module.fc.weight
                    - module.fc.bias

.. code-block:: bash

    PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 \
    srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 \
    python -u -m prototype.solver.cls_solver --config config.yaml

3. Start recovering/finetuning your model from checkpoints

.. code-block:: bash

    sh run.sh Test 16

ToCaffe and ToKestrel
---------------------------

1. Check ``config.yaml``, ``run.sh`` and ``ckpt`` under your workspace

2. Add ``to_kestrel`` configurations into ``config.yaml``

.. code-block:: yaml

    to_kestrel:
        model_name: KM_Classifier_Project_Test
        version: '1.0.0'
        add_softmax: True
        pixel_means: [123.675, 116.28, 103.53]
        pixel_stds: [58.395, 57.12, 57.375]
        is_rgb: True
        save_all_label: True
        type: 'UNKNOWN'
        class_label:
            label_name:
                calculator: 'bypass'
                start: 0
                end: 5
                labels: ['apple', 'banana', 'orange', 'dog', 'cat', 'bird']

.. code-block:: bash

    PYTHONPATH=$PYTHONPATH:../../ GLOG_vmodule=MemcachedClient=-1 \
    srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 \
    python -u -m prototype.tools.convert --config config.yaml --recover=checkpoints/ckpt.pth.tar

3. Start converting your model from checkpoints to Caffe and Kestrel

.. code-block:: bash

    sh run.sh Test 1
