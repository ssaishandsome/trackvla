#!/usr/bin/env bash
set -e

# 按数据集串行生成视觉 cache：先 STT，再 DT，再 AT。
# 每个数据集内部同时使用 GPU 0/1/2 三张卡分片处理。
# 默认跳过已经存在的 *_vfine.pt 和 *_vcoarse.pt；如果要全部重算，
# 在 run_one_dataset 里的 python 命令末尾加 --overwrite 即可。

# 杀死进程
# pkill -f "precache_frames.py"

cd /home/ssa/code/EVT/OpenTrackVLA/OpenTrackVLA

unset LD_LIBRARY_PATH # 清空动态链接库，不让去其他地方寻找，使用conda环境中的lib即可
export LD_LIBRARY_PATH=/miniconda3/envs/trackvla/lib:$LD_LIBRARY_PATH
source /home/ssa/anaconda3/etc/profile.d/conda.sh
conda activate trackvla

PYTHON=python
BATCH_SIZE=8
IMAGE_SIZE=384
NUM_SHARDS=2

DINO_MODEL=/home/ssa/code/EVT/OpenTrackVLA/pretrain/dinov3-vits16-pretrain-lvd1689m
SIGLIP_MODEL=/home/ssa/code/EVT/OpenTrackVLA/pretrain/siglip-so400m-patch14-384

run_one_dataset() {
  NAME=$1
  DATA_ROOT=$2
  CACHE_ROOT=$3

  echo "===== Precache ${NAME}: GPU 0/1 together ====="

  CUDA_VISIBLE_DEVICES=0 $PYTHON precache_frames.py \
    --data_root $DATA_ROOT \
    --cache_root $CACHE_ROOT \
    --batch_size $BATCH_SIZE \
    --image_size $IMAGE_SIZE \
    --dino_model_path $DINO_MODEL \
    --siglip_model_path $SIGLIP_MODEL \
    --num_shards $NUM_SHARDS \
    --shard_id 0 &
  PID0=$!

  CUDA_VISIBLE_DEVICES=1 $PYTHON precache_frames.py \
    --data_root $DATA_ROOT \
    --cache_root $CACHE_ROOT \
    --batch_size $BATCH_SIZE \
    --image_size $IMAGE_SIZE \
    --dino_model_path $DINO_MODEL \
    --siglip_model_path $SIGLIP_MODEL \
    --num_shards $NUM_SHARDS \
    --shard_id 1 &
  PID1=$!

  # CUDA_VISIBLE_DEVICES=2 $PYTHON precache_frames.py \
  #   --data_root $DATA_ROOT \
  #   --cache_root $CACHE_ROOT \
  #   --batch_size $BATCH_SIZE \
  #   --image_size $IMAGE_SIZE \
  #   --dino_model_path $DINO_MODEL \
  #   --siglip_model_path $SIGLIP_MODEL \
  #   --num_shards $NUM_SHARDS \
  #   --shard_id 2 &
  # PID2=$!

  wait $PID0
  wait $PID1
  # wait $PID2

  echo "===== ${NAME} done ====="
}

run_one_dataset STT \
  /mnt/Data/EVTBenchmark/collected_data/stt_train_oracle \
  /mnt/Data/EVTBenchmark/collected_data/stt_vision_cache

run_one_dataset DT \
  /mnt/Data/EVTBenchmark/collected_data/dt_train_oracle \
  /mnt/Data/EVTBenchmark/collected_data/dt_vision_cache

run_one_dataset AT \
  /mnt/Data/EVTBenchmark/collected_data/at_train_oracle \
  /mnt/Data/EVTBenchmark/collected_data/at_vision_cache

echo "All visual caches are done."
