配置说明/Configurations
==========================

.. toctree::
   :maxdepth: 2

Model
---------

模型配置，具体 ``kwargs`` 参数可以参考模型结构定义

.. code-block:: yaml

    model:    # architecture details
        type: resnet50    # model name
        kwargs:
            num_classes: 1000    # number of classes
            bn:
                use_sync_bn: False    # whether to use syncbn
                kwargs: {}    # kwargs of bn

Distribution Training
---------------------------

分布式训练配置，对于分支较多的网络结构建议设置为 ``True``，避免通信卡住

.. code-block:: yaml

    dist:    # distributed communication
        sync: False    # if "True", synchronize gradients after forward; if "False", synchronize gradient during forward

Optimizer
---------------------------

优化器配置，包括FP32优化器和FP16优化器

FP32

.. code-block:: yaml

    optimizer:
        type: SGD
        kwargs:
            nesterov: True
            momentum: 0.9
            weight_decay: 0.0001

FP16，大部分情况下将BN或FC层注册为FP32，进行混合精度训练

.. code-block:: yaml

    optimizer:
        type: FusedFP16SGD
        fp16_normal_bn: True
        kwargs:
            nesterov: True
            momentum: 0.9
            weight_decay: 0.0001

Learing Rate Scheduler
---------------------------

学习率调整配置，包括StepLR，CosineLR，StepDecay等

.. code-block:: yaml

    lr_scheduler:
        type: Cosine
        kwargs:
            base_lr: 0.1    # initial leaning rate
            warmup_lr: 0.4    # learning rate after warming up
            warmup_steps: 2500    # iterations of warmup
            min_lr: 0.0    # minimal learning rate for cosine lr
            max_iter: 250000    # total iterations of training

EMA
---------------------------

滑动平均参数配置，缓解训练不稳定

.. code-block:: yaml

    ema:
        enable: True
        kwargs:
            decay: 0.999

LMS
---------------------------

利用CPU进行大模型支持配置

.. code-block:: yaml

    lms:
        enable: True
        kwargs:
            limit: 12    # the soft limit in G-bytes on GPU memory allocated for tensors

Data
---------------------------

数据配置

``ImageNet`` 数据格式，存储为 ``txt`` 格式，此数据格式仅在训练ImageNet时使用，不推荐用户使用此数据格式训练自定义任务，其中每一行的示例如下：

.. code-block:: yaml

    n01440764/n01440764_10026.JPEG 0

``Custom`` 数据格式，存储为 ``jsonl`` 格式，此数据格式为通用的数据格式，其中每一行的示例如下：

.. code-block:: yaml

    {"filename": n01440764/n01440764_10026.JPEG, "label": 0, "label_name": "dog"}

在使用 ``ImageNet`` 数据格式进行训练时，配置文件为

.. code-block:: yaml

    data:                     # data details
        type: imagenet        # choices = {'imagenet', 'custom'}
        read_from: mc         # choices = {'mc', 'fs', 'fake', 'osg'}
        use_dali: False       # whether to use NVIDIA dali dataloader
        batch_size: 64        # batch size in one GPU
        num_workers: 4        # number of subprocesses for data loading
        pin_memory: True      # whether to copy Tensors into CUDA pinned memory
        input_size: 224       # training image size
        test_resize: 256      # testing resize image size

        train:                            # training data details
            root_dir: /mnt/lustre/share/images/train/
            meta_file: /mnt/lustre/share/images/meta/train.txt
            image_reader:                 # image decoding type
                type: pil
            sampler:                      # sampler details
                type: distributed_iteration  # distributed iteration-based sampler
            transforms:                   # torchvision transforms, flexible
                # type: STANDARD
                - type: random_resized_crop
                kwargs:
                    size: 224
                - type: random_horizontal_flip
                - type: color_jitter
                kwargs:
                    brightness: 0.2
                    contrast: 0.2
                    saturation: 0.2
                    hue: 0.1
                - type: to_tensor
                - type: normalize
                kwargs:
                    mean: [0.485, 0.456, 0.406]
                    std: [0.229, 0.224, 0.225]

        test:                             # testing data details
            root_dir: /mnt/lustre/share/images/val/
            meta_file: /mnt/lustre/share/images/meta/val.txt
            image_reader:
                type: pil
            sampler:                      # sampler details
                type: distributed         # non-repeated sampling
            transforms:                   # torchvision transforms, flexible
                # type: ONECROP
                - type: resize
                kwargs:
                    size: [256, 256]
                - type: center_crop
                kwargs:
                    size: [224, 224]
                - type: to_tensor
                - type: normalize
                kwargs:
                    mean: [0.485, 0.456, 0.406]
                    std: [0.229, 0.224, 0.225]

            evaluator:                    # evaluation metric
                type: imagenet            # choices = {'imagenet', 'custom'}
                kwargs:
                    topk: [1, 5]          # compute topk accuracies

在使用 ``ImageNet`` 数据格式，并使用 ``NVIDIA DALI`` 进行训练时，配置文件为

.. code-block:: yaml

    data:                     # data details
        type: imagenet        # choices = {'imagenet', 'custom'}
        read_from: mc         # choices = {'mc', 'fs', 'fake', 'osg'}
        use_dali: True        # whether to use NVIDIA dali dataloader
        batch_size: 64        # batch size in one GPU
        num_workers: 4        # number of subprocesses for data loading
        pin_memory: True      # whether to copy Tensors into CUDA pinned memory
        input_size: 224       # training image size
        test_resize: 256      # testing resize image size

        train:                            # training data details
            root_dir: /mnt/lustre/share/images/train/
            meta_file: /mnt/lustre/share/images/meta/train.txt
            image_reader:                 # image decoding type
                type: pil
            sampler:                      # sampler details
                type: distributed_iteration  # distributed iteration-based sampler
            transforms:                   # NVIDIA dali transforms, fixed
                type: STANDARD            # random_resized_crop -> random_horizontal_flip -> color_jitter -> to_tensor -> normalize

        test:                             # testing data details
            root_dir: /mnt/lustre/share/images/val/
            meta_file: /mnt/lustre/share/images/meta/val.txt
            image_reader:
                type: pil
            sampler:                      # sampler details
                type: distributed         # non-repeated sampling
            transforms:                   # NVIDIA dali transforms, fixed
                type: ONECROP             # resize -> center_crop -> to_tensor -> normalize
            evaluator:                    # evaluation metric
                type: imagenet            # choices = {'imagenet', 'custom'}
                kwargs:
                    topk: [1, 5]          # compute topk accuracies

在使用 ``Custom`` 数据格式进行训练时，配置文件为

.. code-block:: yaml

    data:                     # data details
        type: custom          # choices = {'imagenet', 'custom'}
        read_from: mc         # choices = {'mc', 'fs', 'fake', 'osg'}
        use_dali: False       # whether to use NVIDIA dali dataloader
        batch_size: 64        # batch size in one GPU
        num_workers: 4        # number of subprocesses for data loading
        pin_memory: True      # whether to copy Tensors into CUDA pinned memory
        input_size: 224       # training image size
        test_resize: 256      # testing resize image size

        train:                            # training data details
            root_dir: /mnt/lustrenew/yuankun/rail-prototype/railc4_data/
            meta_file: /mnt/lustrenew/yuankun/rail-prototype/railc4_data/luoshuan_6_class_train_0618_up10_add_400_700.jsonl
            image_reader:                 # image decoding type
                type: pil
            sampler:                      # sampler details
                type: distributed_iteration  # distributed iteration-based sampler
            transforms:                   # torchvision transforms, flexible
                - type: to_grayscale
                kwargs:
                    num_output_channels: 3
                - type: adjust_gamma
                kwargs:
                    gamma: 1
                - type: resize
                kwargs:
                    size: [224, 224]
                - type: color_jitter
                kwargs:
                    brightness: 0.3
                    contrast: 0.1
                    saturation: 0.1
                - type: random_orientation_rotation
                kwargs:
                    angles: [0, 90, 180, 270]
                - type: random_horizontal_flip
                - type: ramdom_vertical_flip
                - type: to_tensor
                - type: normalize
                kwargs:
                    mean: [0.485, 0.456, 0.406]
                    std: [0.229, 0.224, 0.225]

        test:                             # testing data details
            root_dir: /mnt/lustrenew/yuankun/rail-prototype/railc4_data/
            meta_file: /mnt/lustrenew/yuankun/rail-prototype/railc4_data/luoshuan_6_class_test_0615.jsonl
            image_reader:
                # type: pil
                type: kestrel             # decoding use kestrel type, the same with kestrel SDK
                ues_gpu: False            # if kestrel_caffe, True; if kestrel_ppl/nnie, False; if True, num_workers should be 0
            sampler:                      # sampler details
                type: distributed         # non-repeated sampling
            transforms:                   # torchvision transforms, flexible
                - type: to_grayscale
                kwargs:
                    num_output_channels: 3
                - type: adjust_gamma
                kwargs:
                    gamma: 1
                - type: resize
                kwargs:
                    size: [224, 224]
                - type: to_tensor
                - type: normalize
                kwargs:
                    mean: [124.16, 116.736, 103.936]
                    std: [58.624, 57.344, 57.6]
                    #   mean: [0.485, 0.456, 0.406]
                    #   std: [0.229, 0.224, 0.225]
            evaluator:
                type: custom
                kwargs:
                    key_metric: fp@0.95recall
                    defect_classes: [3, 4]
                    recall_thres: [1.0, 0.95, 0.90, 0.8, 0.7]
                    tpr_thres: [0.99, 0.98, 0.97, 0.95, 0.9]

Saver
---------------------------

存储读取配置

.. code-block:: yaml

    saver:
        print_freq: 10    # frequence of printing logger
        val_freq: 1000    # frequence of evaluating during training
        save_many: False    # whether to save checkpoints after every evaluation
        # pretrain:
        #     path: checkpoints/ckpt.pth.tar
        #     ignore:    # ignore keys in checkpoints
        #         key:    # if training from scratch, pop 'optimzier' and 'last_iter'
        #             - optimizer    # if resuming from ckpt, DO NOT pop them
        #             - last_iter
        #         model:    # ignore modules in model
        #             - module.fc.weight    # if training with different number of classes, pop the keys
        #             - module.fc.bias    # of last fully-connected layers
