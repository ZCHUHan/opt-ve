export DATA_DIR="$HOME/monkey-verifier/data_dir"
export MODEL_DIR="$HOME/monkey-verifier/model_dir"
export LLAVA_SRC_DIR="$HOME/monkey-verifier/llava_setup/LLaVA"
export PYTHONPATH="$LLAVA_SRC_DIR:$PWD:$PYTHONPATH"


python -m scripts.cache_teacher_scores --teacher_ckpt "$MODEL_DIR/Monkey-Verifier-teacher/lora_adapter" --model_name_or_path "$MODEL_DIR/Monkey-Verifier-teacher/llava-v1.5-7b/sft_model" --dataset_path "$DATA_DIR/action_preference_bridge/action_preference_bridge_mini_50k.json" --image_folder "$DATA_DIR/bridge_data_v2/images" --output "$MODEL_DIR/Monkey-Verifier-7B-Energy/teacher_scores.pt" --batch_size 128 --num_workers 8 --device cuda --fp bf16 --energy_expand_pairwise_samples --bits 16 --image_aspect_ratio pad



cd /Users/mio/Documents/code/opt-ve/RLHF

export DATA_DIR="$HOME/monkey-verifier/data_dir"
export MODEL_DIR="$HOME/monkey-verifier/model_dir"
export TEACHER_SCORE_PATH="$MODEL_DIR/Monkey-Verifier-7B-Energy/teacher_scores.pt"

bash scripts/7b-v1.5-224/full_bridge_energy_finetune.sh
核心参数（最重要）：

TEACHER_SCORE_PATH 必须指向你刚生成的 .pt，这样才走 offline KD（脚本末尾会自动拼 --teacher_score_path）。
--rm_loss_type energy_kd_score（脚本里已有）。
这组必须和推理一致：action_placeholder_token、action_dim、action_token_start/end、reward_output_activation（脚本里已有）。
energy_expand_pairwise_samples 要和你缓存 teacher score 时保持一致，否则会长度不匹配报错。
训练输出里现在会自动生成两个文件（在 output_dir，即 Monkey-Verifier-7B-Energy 下）：

inference_consistency_config.json
training_args_snapshot.json
其中 inference_consistency_config.json 已包含你关心的一致性字段：

action_placeholder_token, action_placeholder_id
diff_action_bins, diff_action_min/max, diff_action_token_ids, diff_action_sigma
diff_score_mode（从 rm_loss_type 自动推断，energy 模型会写成 "energy"）
reward_output_activation

推理侧如何用训练保存的 config

import json
from differentiable_verifier import DifferentiableRobotRewardModel

cfg_path = "/path/to/Monkey-Verifier-7B-Energy/inference_consistency_config.json"
cfg = json.load(open(cfg_path, "r"))   # 现在支持 dict
verifier = DifferentiableRobotRewardModel(cfg=cfg)
注意

如果报 Unknown action_placeholder_token=<ACT>，说明你当前加载的 tokenizer 里没有这个 token；需要保证推理时 tokenizer 与训练配置一致（这一步现在会显式报错，不会 silent fallback）。