PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:1 --ntasks-per-node=8 --cpus-per-task=5 \
python -u -m prototype.solver.multiclass_cls_solver --config config.yaml  # --evaluate
