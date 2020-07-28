PYTHONPATH=$PYTHONPATH:../../ GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 -n16 --gres=gpu:8 -x SH-IDC1-10-5-37-[21,22] --ntasks-per-node=8 --cpus-per-task=5 --job-name=mixup \
python -u -m prototype.solver.cls_solver --config config.yaml \
 --recover=checkpoints/ckpt.pth.tar \
