CHUNKS=30 # 将评估任务分成30个块，每个进程处理一个块（有些场景文件里面的episode非常多，最多有153个）
NUM_PARALLEL=1
SAVE_PATH="sim_data/eval/dt"
HF_MODEL_DIR=/home/ssa/code/EVT/OpenTrackVLA/OpenTrackVLA/hf_downloads/omlab__opentrackvla-qwen06b
DINOV3_MODEL_PATH=/home/ssa/code/EVT/OpenTrackVLA/pretrain/dinov3-vits16-pretrain-lvd1689m
SIGLIP_MODEL_PATH=/home/ssa/code/EVT/OpenTrackVLA/pretrain/siglip-so400m-patch14-384

if [ -n "${HF_MODEL_DIR:-}" ]; then
    export HF_MODEL_DIR
    echo "[eval] Using HuggingFace planner weights from ${HF_MODEL_DIR}"
fi

export DINOV3_MODEL_PATH
export SIGLIP_MODEL_PATH

IDX=0
while [ $IDX -lt $CHUNKS ]; do
    for ((i = 0; i < NUM_PARALLEL && IDX < CHUNKS; i++)); do
        echo "Launching job IDX=$IDX on GPU=$((IDX % NUM_PARALLEL))"
        #CUDA_VISIBLE_DEVICES=$((i)) SAVE_VIDEO=1 PYTHONPATH="habitat-lab" python run_eval.py \
        CUDA_VISIBLE_DEVICES=1 SAVE_VIDEO=1 PYTHONPATH="habitat-lab" python run_eval.py \
            --split-num $CHUNKS \
            --split-id $IDX \
            --exp-config 'habitat-lab/habitat/config/benchmark/nav/track/track_infer_dt.yaml' \
            --run-type 'eval' \
            --save-path $SAVE_PATH &
        ((IDX++))
    done
    wait
done
