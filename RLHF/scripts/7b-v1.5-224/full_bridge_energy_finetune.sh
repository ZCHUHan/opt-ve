#!/bin/bash

set -e
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_DIR="$HOME/monkey-verifier/data_dir"
export MODEL_DIR="$HOME/monkey-verifier/model_dir"
export LLAVA_SRC_DIR="$HOME/monkey-verifier/llava_setup/LLaVA"
export PYTHONPATH="$LLAVA_SRC_DIR:$PWD:$PYTHONPATH"
# GPUS_PER_NODE should match the number of GPUs in CUDA_VISIBLE_DEVICES
export GPUS_PER_NODE=8
export OMP_NUM_THREADS=1

# MODEL CONFIG
VISION_TOWER=openai/clip-vit-large-patch14-336
LM_MODEL_NAME=LLaVA-RLHF-7b-v1.5-224/sft_model/

# DATA CONFIG
PREFERENCE_DATA=action_preference_bridge/action_preference_bridge.json

# SAVE CONFIG
MODEL_NAME=Monkey-Verifier-7B-Energy

# TEACHER CONFIG (for knowledge distillation)
TEACHER_DIR=Monkey-Verifier-teacher
# Optional: precomputed teacher scores for offline KD (set after running cache_teacher_scores.py)
# TEACHER_SCORE_PATH=$MODEL_DIR/Monkey-Verifier-7B-Energy/teacher_scores.pt

# WANDB CONFIG
export WANDB_PROJECT="bridge-energy"
export WANDB_NAME="$MODEL_NAME-$(date +%Y%m%d_%H%M%S)"
export WANDB_ENTITY="zhang-c-de42-institue-of-science-tokyo"  # Replace with your wandb username or organization

# TRAINING CONFIG
NUM_EPOCHS=1
LEARNING_RATE=2e-5
BATCH_SIZE=2
GRAD_ACCUMULATION=4

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPUS_PER_NODE \
    finetune_lora_energy.py \
    --do_train \
    --do_eval \
    --seed 42 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --model_name_or_path $MODEL_DIR/$LM_MODEL_NAME \
    --image_folder $DATA_DIR/bridge_data_v2/images \
    --vision_tower $VISION_TOWER \
    --learning_rate $LEARNING_RATE \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_mm_mlp_adapter True \
    --model_max_length 2048 \
    --query_len 1280 \
    --response_len 768 \
    --dataset_path $DATA_DIR/$PREFERENCE_DATA \
    --eval_dataset_path $DATA_DIR/$PREFERENCE_DATA \
    --dataset_name "none" \
    --eval_dataset_name "none" \
    --eval_size 1024 \
    --rm_loss_type "energy_kd_score" \
    --teacher_dir "$MODEL_DIR/$TEACHER_DIR" \
    --lambda_kd 1.0 \
    --lambda_local 1.0 \
    --lambda_score 0.1 \
    --teacher_sigma 0.005 \
    --student_sigma_init 0.005 \
    --student_sigma_final 0.05 \
    --sigma_anneal_steps 2000 \
    --noise_std 0.05 \
    --beta_dist 1.0 \
    --score_sigma 0.05 \
    --lambda_gp 0.1 \
    --gp_cap 1.0 \
    --action_placeholder_token "<ACT>" \
    --action_dim 7 \
    --action_token_start 31744 \
    --action_token_end 31999 \
    --reward_output_activation "softplus" \
    --energy_expand_pairwise_samples True \
    --bits 16 \
    --lora_r 512 \
    --lora_alpha 128 \
    --lora_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --output_dir "$MODEL_DIR/$MODEL_NAME" \
    --num_train_epochs $NUM_EPOCHS \
    --group_by_length False \
    --evaluation_strategy "steps" \
    --eval_steps 150 \
    --save_strategy "steps" \
    --save_steps 150 \
    --save_total_limit 150 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 5 \
    --report_to "wandb" \
    --ddp_backend "nccl" \
    --bf16 True \
    --ddp_find_unused_parameters False \
    --resume_from_training True \
    --reward_prompt_file "./prompts/robot_reward_prompt.txt" \
    --image_aspect_ratio 'pad' \
    --run_name "$WANDB_NAME" \
    ${TEACHER_SCORE_PATH:+--teacher_score_path "$TEACHER_SCORE_PATH"}
