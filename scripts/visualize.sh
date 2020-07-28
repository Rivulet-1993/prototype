PYTHONPATH=$PYTHONPATH:../../ GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 \
python -u -m prototype.tools.inference --config config.yaml  --recover=./checkpoints/ckpt.pth.tar --cam True -o=val_img -i image_dir
