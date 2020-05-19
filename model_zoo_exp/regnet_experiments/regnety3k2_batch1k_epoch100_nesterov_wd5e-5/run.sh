PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 -n16 -x SH-IDC1-10-5-36-156 -x SH-IDC1-10-5-37-[20-21,34] --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 \
python -u -m prototype.solver.cls_solver --config config.yaml \
 #--recover=checkpoints/ckpt_1000.pth.tar \
