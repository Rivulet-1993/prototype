PYTHONPATH=$PYTHONPATH:../../ GLOG_vmodule=MemcachedClient=-1 \
LD_LIBRARY_PATH=/mnt/lustre/zhouyucong/tmp/Prototype/third_party/lib64:/mnt/lustre/zhouyucong/tmp/Prototype/third_party/lib64/ceph:$LD_LIBRARY_PATH \
srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 \
python -u -m prototype.solver.cls_solver --config config.yaml \
 #--recover=checkpoints/ckpt_1000.pth.tar \
