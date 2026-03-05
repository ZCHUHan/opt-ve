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

如果报 Unknown action_placeholder_token=placeholder，说明你当前加载的 tokenizer 与训练不一致；legacy 配置要求 placeholder 对应固定 id=12983。




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




--------------

可以，下面给你一套可直接复现的完整流程（`offline KD` 版）。

**1. 环境与路径**
```bash
export ROOT="$HOME/monkey-verifier"
export DATA_DIR="$ROOT/data_dir"
export MODEL_DIR="$ROOT/model_dir"
cd "$ROOT/RLHF"

export LLAVA_SRC_DIR="$ROOT/llava_setup/LLaVA"
export PYTHONPATH="$LLAVA_SRC_DIR:$PWD:$PYTHONPATH"

export RAW_DATA="$DATA_DIR/action_preference_bridge/action_preference_bridge_mini_50k.json"
export CONT_DATA="$DATA_DIR/action_preference_bridge/action_preference_bridge_mini_50k_continuous.json"
export IMG_DIR="$DATA_DIR/bridge_data_v2/images"
export TEACHER_CKPT="$MODEL_DIR/Monkey-Verifier-teacher"
export BASE_SFT="$MODEL_DIR/Monkey-Verifier-teacher/llava-v1.5-7b/sft_model"
export SCORES_OUT="$MODEL_DIR/Monkey-Verifier-7B-Energy/teacher_scores.pt"
```

**2. 可选但推荐：先把数据转到 continuous 域（严格同构）**
```bash
python scripts/preprocess_action_dataset.py \
  --input "$RAW_DATA" \
  --output "$CONT_DATA" \
  --mode token_to_continuous \
  --action_token_start 31744 \
  --action_token_end 31999 \
  --action_value_min -1.0 \
  --action_value_max 1.0 \
  --action_dim 7 \
  --mapping_style openvla_digitize \
  --keep_original \
  --write_sidecar
```

**3. 缓存 teacher 分数（生成 `teacher_scores.pt`）**
```bash
python -m scripts.cache_teacher_scores \
  --teacher_ckpt "$TEACHER_CKPT" \
  --model_name_or_path "$BASE_SFT" \
  --dataset_path "$CONT_DATA" \
  --image_folder "$IMG_DIR" \
  --output "$SCORES_OUT" \
  --batch_size 128 \
  --num_workers 8 \
  --device cuda \
  --fp bf16 \
  --bits 16 \
  --energy_expand_pairwise_samples \
  --action_placeholder_token "<ACT>" \
  --action_dim 7 \
  --action_token_start 31744 \
  --action_token_end 31999 \
  --action_value_min -1.0 \
  --action_value_max 1.0 \
  --image_aspect_ratio pad \
  --reward_prompt_file "./prompts/robot_reward_prompt.txt"
```
显存不够就把 `--batch_size` 降到 `32/16`，或把 `--bits` 改 `4`。

**4. 开训前做长度一致性检查（非常重要）**
```bash
python - <<'PY'
import json, torch
data_path = "$CONT_DATA"
score_path = "$SCORES_OUT"
data = json.load(open(data_path))
scores = torch.load(score_path, map_location="cpu")
if isinstance(scores, dict) and "scores" in scores:
    scores = scores["scores"]
first = data[0]
pairwise = all(k in first for k in ("output_1","output_2","nrmse_0","nrmse_1")) and ("action" not in first or "rmse" not in first)
expand_pairwise = True
expected = len(data)*2 if (pairwise and expand_pairwise) else len(data)
print("dataset_len_raw =", len(data))
print("expected_after_expand =", expected)
print("teacher_scores_len =", len(scores))
assert len(scores) == expected
print("OK")
PY
```

**5. 训练 student（offline KD）**
```bash
export TEACHER_SCORE_PATH="$SCORES_OUT"
export PREPROCESS_ACTION_DATA=true
export ACTION_VALUE_MIN=-1.0
export ACTION_VALUE_MAX=1.0

bash scripts/7b-v1.5-224/full_bridge_energy_finetune.sh
```
`TEACHER_SCORE_PATH` 一旦传入，训练会优先用离线 `teacher_score`，不走 online teacher forward。

**6. 训练完成后你要拿这两个文件做推理一致性**
```bash
ls "$MODEL_DIR/Monkey-Verifier-7B-Energy/inference_consistency_config.json"
ls "$MODEL_DIR/Monkey-Verifier-7B-Energy/training_args_snapshot.json"
```

**7. 推理/eval 时统一加这个参数（你刚改好的三个入口都支持）**
```bash
--diff_consistency_config "$MODEL_DIR/Monkey-Verifier-7B-Energy/inference_consistency_config.json"
```
适用于 `run_simpler_eval.py` / `run_libero_eval.py` / `run_bridgev2_eval.py`。




现在只要你传 `TEACHER_SCORE_PATH`，它会在 `torchrun` 前：
- 计算 `corr(teacher_score, rmse)`
- 若 `< 0` 自动翻转符号
- `strict` 模式下若相关性太弱直接报错停训（避免 silent 风险）

---

单独运行的一条命令（你要的）：

```bash
python /Users/mio/Documents/code/opt-ve/RLHF/scripts/align_teacher_scores_sign.py \
  --dataset_path "$DATA_DIR/action_preference_bridge/action_preference_bridge_mini_50k_continuous.json" \
  --teacher_scores "$MODEL_DIR/Monkey-Verifier-7B-Energy/teacher_scores.pt" \
  --output "$MODEL_DIR/Monkey-Verifier-7B-Energy/teacher_scores.energy_aligned.pt" \
  --energy_expand_pairwise_samples \
  --min_abs_corr 0.02 \
  --strict
```

---

融入后的训练用法（推荐）：

```bash
export TEACHER_SCORE_PATH="$MODEL_DIR/Monkey-Verifier-7B-Energy/teacher_scores.pt"
export AUTO_ALIGN_TEACHER_SCORE_SIGN=true
export TEACHER_SCORE_SIGN_STRICT=true
export TEACHER_SCORE_MIN_ABS_CORR=0.02

```

---
python -m scripts.cache_teacher_scores --teacher_ckpt "$TEACHER_CKPT" --model_name_or_path "$BASE_SFT" --dataset_path "$CONT_DATA" --image_folder "$IMG_DIR" --output "$SCORES_OUT" --batch_size 128 --num_workers 8 --device cuda --fp bf16 --bits 16 --energy_expand_pairwise_samples --action_placeholder_token "<ACT>" --action_dim 7 --action_token_start 31744 --action_token_end 31999 --action_value_min -1.0 --action_value_max 1.0 --image_aspect_ratio pad --reward_prompt_file "./prompts/robot_reward_prompt.txt"
