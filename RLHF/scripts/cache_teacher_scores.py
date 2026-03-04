import os
import json
import argparse
import torch
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer
from llava import conversation as conversation_lib

from data_utils.data_utils_energy import (
    EnergyRewardModelingDataset,
    DataCollatorForEnergyRewardModelingDataset,
)
from models.reward_model import RewardConfig, RewardModel  # 按你 repo 路径改

class IndexDataset(Dataset):
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        ex = self.base[idx]
        ex["idx"] = torch.tensor(idx, dtype=torch.long)
        return ex

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", required=True, help="teacher checkpoint_dir")
    ap.add_argument("--model_name_or_path", required=True, help="base llava/llama path (same as training)")
    ap.add_argument("--dataset_path", required=True, help="mini json array file")
    ap.add_argument("--image_folder", default=None)
    ap.add_argument("--output", required=True, help="teacher_scores.pt")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--action_dim", type=int, default=7)
    ap.add_argument("--action_placeholder_token", default="<ACT>")
    ap.add_argument("--image_aspect_ratio", default="square")
    ap.add_argument("--energy_expand_pairwise_samples", action="store_true")
    ap.add_argument(
        "--reward_prompt_file",
        default="./prompts/robot_reward_prompt.txt",
        help="Same reward prompt file used during energy RM training.",
    )
    ap.add_argument("--fp", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument(
        "--bits",
        type=int,
        choices=[4, 8, 16],
        default=16,
        help="Teacher load precision. 16 means non-quantized weights (recommended for caching speed if VRAM allows).",
    )
    args = ap.parse_args()

    device = torch.device(args.device)

    # 1) tokenizer (must match training)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="left",
        truncation_side="right",
        use_fast=False,
    )
    # Keep padding behavior consistent with finetune_lora_energy.py (version "v1").
    tokenizer.pad_token = tokenizer.unk_token

    # Set default conversation template so that common_utils.preprocess()
    # follows the same path as in finetune_lora_energy.py.
    if "v1" in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates["v1"]
    else:
        # Fallback to vicuna_v1 if key names differ.
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            "vicuna_v1"
        ]

    # ensure <ACT> exists (same logic as finetune script)
    num_added = 0
    if args.action_placeholder_token not in tokenizer.get_vocab():
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": [args.action_placeholder_token]}
        )
    action_placeholder_id = tokenizer.convert_tokens_to_ids(args.action_placeholder_token)
    if action_placeholder_id == tokenizer.unk_token_id:
        raise ValueError("placeholder token is unk. tokenizer/add_special_tokens failed.")

    # 2) load teacher model ONLY
    cfg = RewardConfig(backbone_model_name_or_path=args.model_name_or_path)
    # minimal args namespace fields used inside RewardModel.make_generative_vlm/get_accelerate_model
    from argparse import Namespace
    rm_args = Namespace(
        model_name_or_path=args.model_name_or_path,
        vision_tower=None,
        lora_modules=None,
        image_aspect_ratio=args.image_aspect_ratio,
        image_grid_pinpoints=None,
        bits=args.bits,
        fp16=(args.fp == "fp16"),
        bf16=(args.fp == "bf16"),
        double_quant=True,
        quant_type="nf4",
        trust_remote_code=False,
        full_finetune=False,
        gradient_checkpointing=False,
        action_placeholder_id=action_placeholder_id,
        action_token_start=31744,
        action_token_end=31999,
        action_value_min=None,
        action_value_max=None,
        reward_output_activation="identity",  # teacher_score 用 raw 更稳
    )

    teacher = RewardModel(
        args=rm_args,
        config=cfg,
        checkpoint_dir=args.teacher_ckpt,
        tokenizer=tokenizer,
        # Keep consistent with finetune_lora_energy.py: make_generative_vlm
        # currently only implements the qlora=True branch.
        qlora=True,
        is_trainable=False,
        action_dim=args.action_dim,
        action_placeholder_id=action_placeholder_id,
        action_token_start=rm_args.action_token_start,
        action_token_end=rm_args.action_token_end,
        action_value_min=rm_args.action_value_min,
        action_value_max=rm_args.action_value_max,
        reward_output_activation=rm_args.reward_output_activation,
    )

    # resize embeddings if we added <ACT>
    if num_added > 0:
        teacher.backbone_model.resize_token_embeddings(len(tokenizer))
        # init new token embedding as avg (same as your finetune script)
        emb = teacher.backbone_model.get_input_embeddings().weight.data
        emb_avg = emb[:-num_added].mean(dim=0, keepdim=True)
        emb[-num_added:] = emb_avg

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    teacher = teacher.to(device)
    if args.fp == "bf16":
        teacher = teacher.to(torch.bfloat16)
    else:
        teacher = teacher.to(torch.float16)

    # get image_processor from teacher (avoid loading extra llava model)
    # NOTE: depending on your llava wrapper, you may need:
    # vision_tower = teacher.backbone_model.get_vision_tower(); vision_tower.load_model(); processor = vision_tower.image_processor
    vision_tower = teacher.backbone_model.get_vision_tower()
    if hasattr(vision_tower, "is_loaded") and not vision_tower.is_loaded:
        vision_tower.load_model()
    processor = vision_tower.image_processor

    # 3) build dataset/collator (same as training)
    class DataArgs:
        """Lightweight stand-in for LLaVA DataArguments used in preprocessing.

        Only the attributes actually accessed in data_utils_energy/common_utils
        are defined here.
        """
        pass

    data_args = DataArgs()
    data_args.dataset_path = args.dataset_path
    data_args.eval_dataset_path = args.dataset_path
    data_args.dataset_name = "dummy"
    data_args.eval_dataset_name = "dummy"
    data_args.eval_size = 0
    data_args.action_dim = args.action_dim
    data_args.action_placeholder_token = args.action_placeholder_token
    data_args.energy_expand_pairwise_samples = args.energy_expand_pairwise_samples
    data_args.image_folder = args.image_folder
    data_args.image_processor = processor
    data_args.is_multimodal = True
    data_args.image_aspect_ratio = args.image_aspect_ratio
    data_args.reward_prompt_file = args.reward_prompt_file
    # Flags used inside preprocess_multimodal / preprocess.
    data_args.mm_use_im_start_end = False
    data_args.mm_use_im_patch_token = False
    data_args.image_to_caption_file = None

    base_ds = EnergyRewardModelingDataset(
        data=[],
        tokenizer=tokenizer,
        query_len=None,
        response_len=None,
        use_data_frame=False,
        data_args=data_args,
    )
    ds = IndexDataset(base_ds)
    collator = DataCollatorForEnergyRewardModelingDataset(tokenizer=tokenizer)

    def collate_with_idx(instances):
        idx = torch.stack([x["idx"] for x in instances]).view(-1)
        batch = collator(instances)
        batch["idx"] = idx
        return batch

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_with_idx,
    )

    scores = torch.empty(len(base_ds), dtype=torch.float32)  # store on CPU

    # 4) inference loop
    with torch.inference_mode():
        for step, batch in enumerate(dl):
            idx = batch.pop("idx").cpu()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch.get("images", None)
            if images is not None:
                images = images.to(device=device, dtype=teacher.reward_head.weight.dtype)
            action_continuous = batch["action_continuous"].to(device)

            out = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                action_continuous=action_continuous,
                action_placeholder_id=action_placeholder_id,
            )
            s = out.rewards
            s = s.squeeze(-1) if s.dim() > 1 else s
            scores[idx] = s.float().cpu()

            if step % 100 == 0:
                print(f"[cache_teacher] step={step} / {len(dl)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(scores, args.output)
    print(f"[cache_teacher] saved {scores.shape} -> {args.output}")

if __name__ == "__main__":
    main()
