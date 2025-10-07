#!/usr/bin/env bash
set -e  # exit on error
set -u  # error on undefined variable



# to run difix on your data ...
#     1- you should have your data.json file structured like expected by Difix as in https://github.com/nv-tlabs/Difix3D?tab=readme-ov-file#difix-single-step-diffusion-for-3d-artifact-removal    
#     2- put path for data.hson below and run
     
#     (note) we add  --dynamo_backend=inductor in running command like we did below to : utilize pytorch 
#    compilar backend (inductor) to get highly optimized kernel for GPU and CPU
#     3- you can find results in w&b link that you will see in the output
#     4- raw output is in '/outputs' folder



# -------------- Configurable parameters --------------
# DATA_JSON="/home/osama/datasets/diffix_dataset_sample/nersemble_dataset.json"
DATA_JSON="/home/osama/datasets/diffix_dataset_sample/ava_dataset.json"
OUTPUT_DIR="./outputs/difix/test"
MAX_STEPS=1
RESOLUTION=512
LR=2e-5
BATCH_SIZE=1
DATALOADER_WORKERS=8
CHECKPOINT_STEPS=100
EVAL_FREQ=1
VIZ_FREQ=1
PROMPT="remove degradation"
TRACKER="wandb"
PROJECT_NAME="difix"
RUN_NAME="validate on AVA_dataset "
TIMESTEP=199
MODEL_NAME="difix_nersemble"
MODEL_PATH="/home/osama/Difix3D/outputs/difix/train/checkpoints/model_1101.pkl"



export CUDA_VISIBLE_DEVICES=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUM_NODES=1
export NUM_GPUS=1
export NCCL_TIMEOUT=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=false

# -------------- Run training --------------
echo "Starting Diffix training..."
accelerate launch   --dynamo_backend=inductor  --mixed_precision=bf16 \
    src/train_difix.py \
    --output_dir "${OUTPUT_DIR}" \
    --dataset_path "${DATA_JSON}" \
    --max_train_steps "${MAX_STEPS}" \
    --resolution "${RESOLUTION}" \
    --learning_rate "${LR}" \
    --train_batch_size "${BATCH_SIZE}" \
    --dataloader_num_workers "${DATALOADER_WORKERS}" \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps "${CHECKPOINT_STEPS}" \
    --eval_freq "${EVAL_FREQ}" \
    --viz_freq "${VIZ_FREQ}" \
    --lambda_lpips 1.0 \
    --lambda_l2 1.0 \
    --lambda_gram 1.0 \
    --gram_loss_warmup_steps 2000 \
    --report_to "${TRACKER}" \
    --tracker_project_name "${PROJECT_NAME}" \
    --tracker_run_name "${RUN_NAME}" \
    --timestep "${TIMESTEP}" \
    --gradient_checkpointing \
    --gradient_accumulation_steps 4 \
    --num_training_epochs 4 \
    --revision "${MODEL_NAME}" \
    --resume  "${MODEL_PATH}" \

    # --variant="${MODEL_NAME}" \
    # --pretrained_model_name_or_path  "${MODEL_PATH}" \


echo "Training finished."
