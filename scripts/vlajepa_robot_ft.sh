export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
export TMPDIR=/home/dataset-local/tmp
export FFMPEG_THREADS=1
export OMP_NUM_THREADS=1

export WANDB_MODE=disabled

# 24 GB 卡上必开，避免显存碎片化导致中途 OOM（见 doc/02_training.md §8.2）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 用绝对路径，避免落在没装 accelerate 的 conda base 环境里（见 doc/02_training.md §2.1）
/opt/conda/envs/VLA_JEPA/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/vlajepa_robot_ft.yaml
