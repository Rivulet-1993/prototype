PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 -n16 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 \
python -u -m prototype.solver.cls_solver --config config.yaml # --evaluate
