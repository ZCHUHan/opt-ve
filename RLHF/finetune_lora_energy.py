# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

import os
from dataclasses import dataclass, field
from typing import Optional, List, Literal
import logging

import torch
import transformers
import argparse
from transformers import set_seed

from transformers import AutoTokenizer

from lora_utils import (
    SavePeftModelCallback,
    print_trainable_parameters,
    get_last_checkpoint,
)
from data_utils.data_utils_energy import make_energy_reward_modeling_data_module
from models.reward_model import (
    RewardConfig,
    RewardModel,
    RewardModelTrainer as Trainer,
    compute_energy_modeling_metrics,
)

from llava import conversation as conversation_lib
from llava.model import *
from llava.constants import (
    IMAGE_TOKEN_INDEX,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.set_default_dtype(torch.bfloat16)

logger = logging.getLogger(__name__)


class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-12b")
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )
    # from LLaVA
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1
    )  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    dataset_name: str = field(default=None, metadata={"help": "Dataset name"})
    eval_dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    eval_dataset_name: str = field(default="alpaca_human_preference")
    eval_size: int = field(
        default=500,
        metadata={
            "help": "Number of examples to split out from training to use for evaluation."
        },
    )
    # From LLaVA
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    reward_prompt_file: Optional[str] = field(default=None)
    image_to_caption_file: Optional[str] = field(default=None)
    action_dim: int = field(default=7)
    action_placeholder_token: str = field(default="placeholder")
    energy_expand_pairwise_samples: bool = field(default=True)
    # Optional: path to pre-computed teacher scores for offline KD.
    teacher_score_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a .pt tensor of pre-computed teacher scores for offline KD."
        },
    )


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    # in TrainingArguments
    rm_loss_type: str = field(default="energy_kd_score")
    lambda_kd: float = field(default=1.0)
    lambda_local: float = field(default=1.0)
    lambda_score: float = field(default=0.1)
    noise_std: float = field(default=0.05)
    beta_dist: float = field(default=1.0)
    teacher_sigma: float = field(default=0.005)        # hard-approx
    student_sigma_init: float = field(default=0.005)   # start near hard
    student_sigma_final: float = field(default=0.05)   # your inference sigma
    sigma_anneal_steps: int = field(default=2000)
    score_sigma: float = field(default=0.05)
    teacher_dir: Optional[str] = field(default=None)   # frozen teacher ckpt (optional)
    
    cache_dir: Optional[str] = field(default=None)
    # From LLaVA
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    # From AlpacaFarm
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be left padded to this length always during training."
        },
    )
    query_len: int = field(default=None, metadata={"help": "Length of the query."})
    response_len: int = field(
        default=None, metadata={"help": "Length of the response."}
    )
    label_names: List[str] = field(
        default_factory=lambda: ["rmse"],
        metadata={
            "help": "Names of the labels in the dataset. "
            "This is needed to get transformers.Trainer to not throw those tensors away before `compute_loss`."
            "By default, the trainer throws away columns it doesn't recognize when creating the "
            "`train_dataloader` (see `_remove_unused_columns`). "
        },
    )
    padding: Literal["max_length", "longest"] = field(
        default="longest",
        metadata={
            "help": "Padding strategy. If 'max_length', pads to `model_max_length` always; this might lead to some "
            "redundant compute. If 'longest', pads to the longest sequence in the batch, capped by `model_max_length`."
        },
    )
    # From QLoRA
    full_finetune: bool = field(
        default=False, metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    lora_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Which modules to use LoRA on. If None, will use all linear layers."
        },
    )
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    report_to: str = field(
        default="none",
        metadata={"help": "To use wandb or something else for reporting."},
    )
    resume_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the checkpoint to resume."},
    )
    output_dir: str = field(
        default="./output", metadata={"help": "The output dir for logs and checkpoints"}
    )
    optim: str = field(
        default="paged_adamw_32bit", metadata={"help": "The optimizer to be used"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={
            "help": "The training batch size per GPU. Increase for better speed."
        },
    )
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={
            "help": "How many gradients to accumulate before to perform an optimizer step"
        },
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "The L2 weight decay rate of AdamW"}
    )  # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": "The learnign rate"})
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Removed unused columns. Needed to make this codebase work."},
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={
            "help": "Gradient clipping max norm. This is tuned and works well for all models tested."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing. You want to use this."},
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "To train or not to train, that is the question?"},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Fraction of steps to do a warmup for"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "The frequency of update steps after which to log the loss"},
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "When to save checkpoints"}
    )
    save_steps: int = field(default=250, metadata={"help": "How often to save a model"})
    save_total_limit: int = field(
        default=40,
        metadata={
            "help": "How many checkpoints to save before the oldest is overwritten"
        },
    )
    resume_from_training: bool = field(
        default=False, metadata={"help": "Resume from training"}
    )
    rm_loss_type: str = field(default="energy")
    lambda_gp: float = field(default=0.1)
    gp_cap: float = field(default=1.0)
    action_placeholder_id: Optional[int] = field(default=None)
    action_token_offset: int = field(default=1000)
    action_token_start: int = field(default=31744)
    action_token_end: int = field(default=31999)
    action_value_min: Optional[float] = field(default=None)
    action_value_max: Optional[float] = field(default=None)
    reward_output_activation: str = field(default="softplus")


def rank0_print(*args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)


def _score_mode_from_loss_type(rm_loss_type: str) -> str:
    if rm_loss_type in ("energy", "energy_kd_score"):
        return "energy"
    return "reward"


def save_inference_consistency_config(
    output_dir: str,
    args: argparse.Namespace,
    data_args: DataArguments,
    training_args: TrainingArguments,
    action_placeholder_id: int,
):
    action_token_start = int(training_args.action_token_start)
    action_token_end = int(training_args.action_token_end)
    action_token_offset = int(getattr(training_args, "action_token_offset", 1000))
    model_action_token_start = action_token_start - action_token_offset
    model_action_token_end = action_token_end - action_token_offset
    diff_action_bins = action_token_end - action_token_start + 1
    if diff_action_bins <= 0:
        raise ValueError(
            f"Invalid action token range: [{action_token_start}, {action_token_end}]"
        )

    action_min = (
        float(training_args.action_value_min)
        if training_args.action_value_min is not None
        else -1.0
    )
    action_max = (
        float(training_args.action_value_max)
        if training_args.action_value_max is not None
        else 1.0
    )
    if action_max <= action_min:
        raise ValueError(
            f"Invalid action value range: min={action_min}, max={action_max}"
        )

    score_mode = _score_mode_from_loss_type(str(training_args.rm_loss_type))
    consistency_cfg = {
        # Source/runtime metadata.
        "model_name_or_path": str(args.model_name_or_path),
        "rm_loss_type": str(training_args.rm_loss_type),
        "reward_output_activation": str(training_args.reward_output_activation),
        "diff_reward_activation": str(training_args.reward_output_activation),
        "diff_score_mode": score_mode,
        # Placeholder consistency (A1).
        "action_placeholder_token": str(data_args.action_placeholder_token),
        "action_placeholder_id": int(action_placeholder_id),
        # Action token/bin consistency (A2).
        "action_dim": int(data_args.action_dim),
        "action_token_offset": action_token_offset,
        "raw_action_token_start": action_token_start,
        "raw_action_token_end": action_token_end,
        "action_token_start": model_action_token_start,
        "action_token_end": model_action_token_end,
        "diff_action_bins": int(diff_action_bins),
        "diff_action_min": action_min,
        "diff_action_max": action_max,
        "diff_action_token_ids": list(
            range(model_action_token_start, model_action_token_end + 1)
        ),
        "diff_action_sigma": float(training_args.student_sigma_final),
        "diff_action_strict_token_check": True,
        "diff_action_log_diagnostics": True,
        # Data/prompt consistency hints.
        "energy_expand_pairwise_samples": bool(
            getattr(data_args, "energy_expand_pairwise_samples", True)
        ),
        "image_aspect_ratio": str(data_args.image_aspect_ratio),
        "query_len": int(training_args.query_len)
        if training_args.query_len is not None
        else None,
        "response_len": int(training_args.response_len)
        if training_args.response_len is not None
        else None,
    }

    serializable_args = {
        k: v
        for k, v in vars(args).items()
        if isinstance(v, (str, int, float, bool, list, dict)) or v is None
    }

    os.makedirs(output_dir, exist_ok=True)
    consistency_path = os.path.join(output_dir, "inference_consistency_config.json")
    args_path = os.path.join(output_dir, "training_args_snapshot.json")
    with open(consistency_path, "w") as fout:
        json.dump(consistency_cfg, fout, indent=2)
    with open(args_path, "w") as fout:
        json.dump(serializable_args, fout, indent=2)

    rank0_print(f"Saved inference consistency config: {consistency_path}")
    rank0_print(f"Saved training args snapshot: {args_path}")


def train():
    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        extra_args,
    ) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    if args.resume_dir is not None:
        checkpoint_dir, completed_training = args.resume_dir, False
    else:
        checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)

    if completed_training:
        rank0_print("Detected that training was already completed!")

    # If training from scratch and teacher_dir is provided, initialize student from teacher
    if checkpoint_dir is None and training_args.teacher_dir is not None:
        checkpoint_dir = training_args.teacher_dir
        rank0_print("Initializing student model from teacher checkpoint:", checkpoint_dir)
    elif checkpoint_dir is None:
        rank0_print("Training from scratch.")
    else:
        rank0_print("Loading from checkpoint:", checkpoint_dir)
        if args.resume_from_training:
            rank0_print("Resuming from training not supported yet. Exiting.")
            exit(1)

    tokenizer_model_name = args.model_name_or_path
    TokenizerClass = AutoTokenizer

    # Tokenizer
    tokenizer = TokenizerClass.from_pretrained(
        tokenizer_model_name,
        cache_dir=args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        truncation_side="right",
        use_fast=False,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]

    # Legacy RM compatibility: use fixed placeholder token/id from the original pipeline.
    # This keeps behavior aligned with finetune_lora_rm.py + common_utils.preprocess().
    data_args.action_placeholder_token = "placeholder"
    action_placeholder_id = 12983
    legacy_id = tokenizer.convert_tokens_to_ids(data_args.action_placeholder_token)
    if legacy_id != action_placeholder_id:
        raise ValueError(
            "Legacy placeholder mismatch: expected token 'placeholder' to map to id 12983, "
            f"got {legacy_id}. Please use the original tokenizer/checkpoint family."
        )
    training_args.action_placeholder_id = action_placeholder_id
    args.action_placeholder_id = action_placeholder_id

    if model_args.vision_tower is not None:
        from llava.model import LlavaLlamaForCausalLM

        with DisableLogger():
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
            )

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True
        model.config.mm_use_im_start_end = (
            data_args.mm_use_im_start_end
        ) = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end

    data_module = make_energy_reward_modeling_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )

    if args.do_train:
        training_data = data_module["train_dataset"]
        rank0_print("Training data size:", len(training_data))
        rank0_print("Training data example:")
        for i in range(min(3, len(training_data))):
            ex = training_data[i]
            ex_input_ids = ex["input_ids"].clone()
            ex_input_ids[ex_input_ids == IMAGE_TOKEN_INDEX] = tokenizer.eos_token_id
            rank0_print(tokenizer.decode(ex_input_ids, skip_special_tokens=False))
            rank0_print("action_continuous:", ex["action_continuous"].tolist())
            rank0_print("rmse:", float(ex["rmse"]))
            rank0_print("=" * 20)
            rank0_print("=" * 20)

        # Guardrail: energy training expects continuous actions in action_value range.
        if str(training_args.rm_loss_type) in ("energy", "energy_kd_score"):
            action_min = (
                float(training_args.action_value_min)
                if training_args.action_value_min is not None
                else -1.0
            )
            action_max = (
                float(training_args.action_value_max)
                if training_args.action_value_max is not None
                else 1.0
            )
            sample_count = min(128, len(training_data))
            out_of_range = 0
            max_abs_action = 0.0
            for i in range(sample_count):
                a = training_data[i]["action_continuous"].float()
                max_abs_action = max(max_abs_action, float(a.abs().max().item()))
                if bool(((a < action_min) | (a > action_max)).any()):
                    out_of_range += 1

            if out_of_range > 0:
                raise ValueError(
                    "Detected out-of-range action_continuous values for energy training. "
                    f"Range=[{action_min}, {action_max}], "
                    f"out_of_range_samples={out_of_range}/{sample_count}, "
                    f"max_abs_action={max_abs_action:.4f}. "
                    "This usually means token IDs were fed as action_continuous. "
                    "Please preprocess dataset to continuous domain (PREPROCESS_ACTION_DATA=true) "
                    "or set dataset_path to *_continuous.json."
                )

    model_action_token_start = int(training_args.action_token_start) - int(
        training_args.action_token_offset
    )
    model_action_token_end = int(training_args.action_token_end) - int(
        training_args.action_token_offset
    )
    if model_action_token_end < model_action_token_start:
        raise ValueError(
            "Invalid model action token range after applying offset: "
            f"[{model_action_token_start}, {model_action_token_end}]"
        )

    config = RewardConfig(backbone_model_name_or_path=model_args.model_name_or_path)

    with DisableLogger():
        model = RewardModel(
            args=args,
            config=config,
            qlora=True,
            checkpoint_dir=checkpoint_dir,
            tokenizer=tokenizer,
            action_dim=data_args.action_dim,
            action_placeholder_id=action_placeholder_id,
            action_token_start=model_action_token_start,
            action_token_end=model_action_token_end,
            action_value_min=training_args.action_value_min,
            action_value_max=training_args.action_value_max,
            reward_output_activation=training_args.reward_output_activation,
        ).to(torch.bfloat16)

    model.backbone_model.config.use_cache = False
    model.action_placeholder_id = action_placeholder_id
    save_inference_consistency_config(
        output_dir=args.output_dir,
        args=args,
        data_args=data_args,
        training_args=training_args,
        action_placeholder_id=action_placeholder_id,
    )
    print_trainable_parameters(args, model)
    print("loaded model")
    set_seed(args.seed)

    # Optionally load teacher model for online KD.
    # If offline teacher scores are provided, we skip loading the teacher to save VRAM.
    teacher = None
    use_online_teacher = (
        training_args.rm_loss_type == "energy_kd_score"
        and float(getattr(training_args, "lambda_kd", 0.0)) > 0.0
        and getattr(data_args, "teacher_score_path", None) is None
    )
    if use_online_teacher:
        teacher_ckpt = training_args.teacher_dir or checkpoint_dir
        teacher = RewardModel(
            args=args,
            config=config,
            qlora=True,
            checkpoint_dir=teacher_ckpt,
            tokenizer=tokenizer,
            action_dim=data_args.action_dim,
            action_placeholder_id=action_placeholder_id,
            action_token_start=model_action_token_start,
            action_token_end=model_action_token_end,
            action_value_min=training_args.action_value_min,
            action_value_max=training_args.action_value_max,
            reward_output_activation=training_args.reward_output_activation,
            is_trainable=False,
        ).to(torch.bfloat16)

        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    trainer = Trainer(
        model=model,
        teacher_model=teacher,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_energy_modeling_metrics,
        **{k: v for k, v in data_module.items() if k != "predict_dataset"},
    )

    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)

    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    all_metrics = {"run_name": args.run_name}

    # Training
    if args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)

    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)

    if args.do_train or args.do_eval:
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))


if __name__ == "__main__":
    train()
