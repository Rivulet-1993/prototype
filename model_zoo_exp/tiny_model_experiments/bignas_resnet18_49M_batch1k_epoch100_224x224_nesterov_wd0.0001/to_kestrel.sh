PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$3 --ntasks-per-node=8 --cpus-per-task=5 \
python -u -m prototype.tools.convert --config config.yaml \
 --recover=checkpoints/ckpt.pth.tar
