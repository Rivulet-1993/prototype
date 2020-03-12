PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --job-name=mb-share \
python -u -m prototype.solver.cls_solver --config config.yaml \
# --recover=checkpoints/ckpt.pth.tar \
