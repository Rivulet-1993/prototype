安装说明/Installation
==========================

.. toctree::
   :maxdepth: 2


1. 查询tags然后查看历史版本，推荐使用master分支

.. code-block:: bash

   git clone http://gitlab.bj.sensetime.com/spring-ce/element/prototype.git prototype
   cd prototype

2. 激活环境：激活集群PyTorch和Linklink环境

.. code-block:: bash

   source r0.3.2

3. 安装依赖项

.. code-block:: bash

   pip install --user -r requirements.txt
