PYTHONPATH=$PYTHONPATH:../../ GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 \
python -u -m spring.wrapper --config config.yaml --test-sci \
 #--load-path=$work_path/ckpt.pth.tar \
 #--recover
#LD_LIBRARY_PATH=/mnt/lustre/zhouyucong/tmp/Prototype/third_party/lib64:/mnt/lustre/zhouyucong/tmp/Prototype/third_party/lib64/ceph:$LD_LIBRARY_PATH \
