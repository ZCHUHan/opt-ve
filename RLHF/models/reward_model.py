# Copyright 2023 The LLaVA-RLHF Team
# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import Namespace
from dataclasses import dataclass
import os
from typing import Optional, Dict, Sequence, Union

import einops
import torch
from torch import Tensor, nn
import torch.nn.functional as F

import transformers
from transformers.trainer_utils import EvalPrediction
from transformers.utils.generic import ModelOutput

from peft import PeftModel, LoraModel, LoraConfig

from models.qlora_model import get_accelerate_model

from llava.model import *


def unpack_dict(
    d: Dict, keys: Sequence[str], return_type: type = tuple
) -> Union[Sequence, Dict]:
    if return_type in (tuple, list):
        return return_type(d[key] for key in keys)
    elif return_type == dict:
        return {key: d[key] for key in keys}
    else:
        raise ValueError(f"Unknown return_type: {return_type}")


def batch_select(input: Tensor, index: Tensor):
    """Select elements from a batched tensor with a batched index tensor.

    Example:
        input = torch.tensor([
            [0, 1, 2],
            [3, 0, 9],
            [6, 7, 8],
        ])
        index = torch.tensor([[0, 1], [1, 0], [0, 0]])
        batch_select(input, index) = tensor([
            [0, 1],
            [0, 3],
            [6, 6]
        ])
    """
    dummy_index = torch.arange(input.size(0), device=input.device).unsqueeze(-1)
    return input[dummy_index, index]


def make_generative_vlm(
    args: Namespace,
    model_name_or_path: str,
    qlora: bool = False,
    checkpoint_dir: Optional[str] = None,
    adapter_name="lora_default",
    is_trainable=True,
    reuse_base_model=False,
    tokenizer=None,
    **kwargs,
):
    if qlora:
        if checkpoint_dir is None or checkpoint_dir in ["scratch", "none"]:
            return get_accelerate_model(args, None, tokenizer=tokenizer)
        else:
            return get_accelerate_model(
                args,
                checkpoint_dir=checkpoint_dir,
                adapter_name=adapter_name,
                is_trainable=is_trainable,
                reuse_base_model=reuse_base_model,
                tokenizer=tokenizer,
            )
    else:
        raise ValueError(f"Unknown model type: {model_name_or_path}")


def get_transformer_hidden_size(model: transformers.PreTrainedModel):
    if isinstance(model, PeftModel):
        return get_transformer_hidden_size(model.base_model)

    if isinstance(model, LoraModel):
        return get_transformer_hidden_size(model.model)

    if isinstance(model, transformers.GPT2LMHeadModel):
        hidden_size_attr_name = "n_embd"
    elif isinstance(model, transformers.OPTForCausalLM):
        hidden_size_attr_name = "word_embed_proj_dim"
    elif isinstance(model, transformers.T5ForConditionalGeneration):
        hidden_size_attr_name = "d_model"
    elif "modelling_RW.RWModel" in str(
        type(model)
    ) or "modelling_RW.RWForCausalLM" in str(type(model)):
        # TODO(zhiqings): Hack to add support for Falcon.
        hidden_size_attr_name = "hidden_size"
    else:
        # Hack to deal with the fact that transformers library changed the LLaMA model name.
        llama_cls = getattr(
            transformers,
            "LLaMAForCausalLM"
            if hasattr(transformers, "LLaMAForCausalLM")
            else "LlamaForCausalLM",
        )
        if isinstance(model, llama_cls) or "LlamaForCausalLM" in str(type(model)):
            hidden_size_attr_name = "hidden_size"
        else:
            raise ValueError(f"Unknown base_model type: {type(model)}")
        from typing import Any, Mapping
    return getattr(model.config, hidden_size_attr_name)


class RewardConfig(transformers.PretrainedConfig):
    model_type = "reward_model"

    # Huggingface doesn't allow non-kwargs for `__init__`.
    def __init__(self, backbone_model_name_or_path=None, **kwargs):
        super(RewardConfig, self).__init__(**kwargs)
        self.backbone_model_name_or_path = backbone_model_name_or_path

@dataclass
class RewardModelOutput(ModelOutput):
    rewards: Tensor = None
    raw_rewards: Tensor = None


class ActionSoftEmbedder(nn.Module):
    """Differentiable mapping from continuous action values to token embeddings."""

    def __init__(
        self,
        action_dim: int = 7,
        action_token_start: int = 31744,
        action_token_end: int = 31999,
        action_value_min: Optional[float] = None,
        action_value_max: Optional[float] = None,
    ):
        super().__init__()
        if action_token_end < action_token_start:
            raise ValueError("action_token_end must be >= action_token_start")
        if (
            action_value_min is not None
            and action_value_max is not None
            and action_value_max <= action_value_min
        ):
            raise ValueError("action_value_max must be greater than action_value_min")

        self.action_dim = action_dim
        self.action_token_start = action_token_start
        self.action_token_end = action_token_end
        self.num_bins = action_token_end - action_token_start + 1
        self.action_value_min = action_value_min
        self.action_value_max = action_value_max

    def _to_bin_coordinate(self, action_continuous: Tensor) -> Tensor:
        action = action_continuous.float()
        if self.action_value_min is None or self.action_value_max is None:
            return action - float(self.action_token_start)

        denom = max(self.action_value_max - self.action_value_min, 1e-6)
        return (action - self.action_value_min) * (self.num_bins - 1) / denom

    def forward(self, action_continuous: Tensor, token_embedding_weight: Tensor) -> Tensor:
        if action_continuous.dim() != 2:
            raise ValueError(
                f"action_continuous must have shape [B, A], got {tuple(action_continuous.shape)}"
            )
        if action_continuous.size(-1) != self.action_dim:
            raise ValueError(
                f"Action dim mismatch: expected {self.action_dim}, "
                f"got {action_continuous.size(-1)}"
            )

        bin_coordinate = self._to_bin_coordinate(action_continuous)
        max_bin = float(self.num_bins - 1) - 1e-6
        bin_coordinate = torch.clamp(bin_coordinate, min=0.0, max=max_bin)

        lower_bin = torch.floor(bin_coordinate)
        upper_bin = torch.clamp(lower_bin + 1.0, max=float(self.num_bins - 1))
        frac = (bin_coordinate - lower_bin).unsqueeze(-1)

        lower_ids = (lower_bin.long() + self.action_token_start).to(
            device=token_embedding_weight.device
        )
        upper_ids = (upper_bin.long() + self.action_token_start).to(
            device=token_embedding_weight.device
        )

        lower_embed = F.embedding(lower_ids, token_embedding_weight)
        upper_embed = F.embedding(upper_ids, token_embedding_weight)
        frac = frac.to(dtype=lower_embed.dtype, device=lower_embed.device)
        return (1.0 - frac) * lower_embed + frac * upper_embed


class RewardModel(transformers.PreTrainedModel):
    config_class = RewardConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        args: Namespace,
        config: RewardConfig,
        checkpoint_dir: Optional[str] = None,
        adapter_name="lora_default",
        tokenizer=None,
        **kwargs,
    ):
        super(RewardModel, self).__init__(config)
        self.adapter_name = adapter_name
        self.backbone_model = make_generative_vlm(
            args,
            config.backbone_model_name_or_path,
            checkpoint_dir=checkpoint_dir,
            adapter_name=adapter_name,
            tokenizer=tokenizer,
            **kwargs,
        )
        hidden_size = get_transformer_hidden_size(self.backbone_model)
        reward_head = nn.Linear(hidden_size, 1)
        torch.nn.init.zeros_(reward_head.bias)
        device = next(self.backbone_model.parameters()).device
        self.reward_head = reward_head.to(device)
        self.reward_output_activation = kwargs.get("reward_output_activation", "identity")
        self.action_placeholder_id = kwargs.get("action_placeholder_id", None)
        self.action_soft_embedder = ActionSoftEmbedder(
            action_dim=kwargs.get("action_dim", 7),
            action_token_start=kwargs.get("action_token_start", 31744),
            action_token_end=kwargs.get("action_token_end", 31999),
            action_value_min=kwargs.get("action_value_min", None),
            action_value_max=kwargs.get("action_value_max", None),
        )

        if checkpoint_dir is not None:
            reward_head_path = os.path.join(checkpoint_dir, "reward_head")
            if os.path.exists(reward_head_path):
                self.reward_head.load_state_dict(
                    torch.load(
                        reward_head_path,
                        map_location="cpu",
                    )
                )
            else:
                print(f"Warning: reward head not found at {reward_head_path}")

        self.reward_head.requires_grad_(kwargs.get("is_trainable", True))

    def forward(
        self,
        input_ids,
        attention_mask=None,
        images=None,
        return_dict=True,
        action_continuous=None,
        action_embeds=None,
        action_placeholder_id=None,
        **kwargs,
    ):
        # We only compute the rewards and don't compute the logistic regression loss in this function so that it's
        # easier to use for later stages of reranking / RL training.
        self.backbone_model.set_adapter(self.adapter_name)
        self.backbone_model.config.use_cache = False

        if action_embeds is None and action_continuous is not None:
            token_embedding_weight = self.backbone_model.get_input_embeddings().weight
            action_embeds = self.action_soft_embedder(
                action_continuous=action_continuous,
                token_embedding_weight=token_embedding_weight,
            )
        if action_placeholder_id is None:
            action_placeholder_id = self.action_placeholder_id

        backbone_extra_kwargs = {}
        if action_embeds is not None:
            backbone_extra_kwargs["action_embeds"] = action_embeds
        if action_placeholder_id is not None:
            backbone_extra_kwargs["action_placeholder_id"] = action_placeholder_id

        # print(input_ids.shape, images.shape, 'images', images.dtype)
        outputs = self.backbone_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            images=images,
            **backbone_extra_kwargs,
            **kwargs,
        )
        last_hidden_state = outputs.hidden_states[-1]
        assert isinstance(last_hidden_state, torch.Tensor), f"{outputs}"
        # last_hidden_state = outputs.last_hidden_state
        # TODO(zhiqings): Hacking to make sure every parameter is used in the backward pass.
        logits = outputs.logits
        last_hidden_state = last_hidden_state + 0.0 * torch.mean(logits)

        last_hidden_state_at_the_end = last_hidden_state[:, -1, :]
        # TODO(lxuechen): Make returning rewards at all positions and last_hidden_state an option.
        # last_hidden_state_at_the_end = last_hidden_state_at_the_end.type_as(
        #     next(self.reward_head.parameters()) # HACK(sheng): error with data parallel
        # )
        last_hidden_state_at_the_end = last_hidden_state_at_the_end.type_as(
            self.reward_head.weight
        )
        # print(last_hidden_state_at_the_end.device, self.reward_head.weight.device, self.reward_head.bias.device)
        raw_rewards = self.reward_head(last_hidden_state_at_the_end).squeeze(-1)
        rewards = raw_rewards
        if self.reward_output_activation == "sigmoid":
            rewards = torch.sigmoid(raw_rewards)
        elif self.reward_output_activation == "softplus":
            rewards = F.softplus(raw_rewards)
        elif self.reward_output_activation != "identity":
            raise ValueError(
                f"Unknown reward_output_activation={self.reward_output_activation}"
            )

        if return_dict:
            return RewardModelOutput(rewards=rewards, raw_rewards=raw_rewards)
        return (rewards,)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, transformers.LlamaModel):
            module.gradient_checkpointing = value

        # TODO(zhiqings): Hack to add support for Falcon.
        if "RWModel" in str(type(module)):
            module.gradient_checkpointing = value


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class RewardModelTrainer(transformers.Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            # Save the model
            _state_dict = state_dict
            if _state_dict is None:
                # Only save the model itself if we are using distributed training
                model_to_save = unwrap_model(self.model)
                _state_dict = model_to_save.state_dict()

            weight_to_save = {}
            keys_to_match = ["mm_projector", "embed_tokens", "embed_in"]
            for k, v in _state_dict.items():
                if any(key_match in k for key_match in keys_to_match):
                    weight_to_save[k] = v

            current_folder = output_dir.split("/")[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(
                    weight_to_save,
                    os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                )
            else:
                torch.save(
                    weight_to_save, os.path.join(output_dir, f"mm_projector.bin")
                )

        super(RewardModelTrainer, self)._save(output_dir, state_dict)

    def _compute_pairwise_loss(self, model, inputs, return_outputs=False, alpha=0.1):
        input_ids, attention_mask, index_0, index_1, choice, images, nrmse_0, nrmse_1 = unpack_dict(
            inputs,
            keys=(
                "input_ids",
                "attention_mask",
                "index_0",
                "index_1",
                "choice",
                "images",
                "nrmse_0", 
                "nrmse_1" 
            ),
        )
        # repeat images to match the number of candidates
        images = images.unsqueeze(1).repeat(1, input_ids.size(1), 1, 1, 1)
        images = einops.rearrange(images, "b n h w c -> (b n) h w c") # Adapt 'h w c' if needed

        num_candidates, num_pairs = input_ids.size(1), choice.size(1)
        input_ids_flat, attention_mask_flat = tuple(
            einops.rearrange(x, "b c l -> (b c) l") for x in (input_ids, attention_mask)
        )
        outputs = model(
            input_ids=input_ids_flat, attention_mask=attention_mask_flat, images=images
        )
        rewards_flat = outputs.rewards
        rewards = einops.rearrange(
            rewards_flat, "(b c) -> b c", c=num_candidates
        )  

        # Get rewards for the pairs being compared
        rewards_0, rewards_1 = tuple(
            batch_select(rewards, index) for index in (index_0, index_1)
        )  

        logits = rewards_1 - rewards_0  

        logits_dtype = logits.dtype
        logits_device = logits.device
        choice_float = choice.to(dtype=logits_dtype, device=logits_device)
        nrmse_0 = nrmse_0.to(dtype=logits_dtype, device=logits_device)
        nrmse_1 = nrmse_1.to(dtype=logits_dtype, device=logits_device)

        reward_W = choice_float * rewards_1 + (1 - choice_float) * rewards_0
        reward_L = choice_float * rewards_0 + (1 - choice_float) * rewards_1

        predicted_reward_diff = reward_W - reward_L 
        delta_hat = torch.abs(predicted_reward_diff)
        rmse_W = choice_float * nrmse_1 + (1 - choice_float) * nrmse_0
        rmse_L = choice_float * nrmse_0 + (1 - choice_float) * nrmse_1

        delta_star = torch.abs(rmse_W - rmse_L)
        preference_level_penalty = alpha * (delta_star - delta_hat)**2
        modified_reward_diff = predicted_reward_diff - preference_level_penalty

        loss = -F.logsigmoid(modified_reward_diff).mean()
        logged_rewards = torch.stack((rewards_1, rewards_0), dim=-1)

        return (loss, dict(logits=logged_rewards)) if return_outputs else loss

    def _compute_energy_loss(self, model, inputs, return_outputs=False):
        input_ids, attention_mask, action_continuous, rmse = unpack_dict(
            inputs,
            keys=(
                "input_ids",
                "attention_mask",
                "action_continuous",
                "rmse",
            ),
        )
        images = inputs.get("images", None)

        action_continuous = action_continuous.detach().clone().requires_grad_(True)
        action_placeholder_id = getattr(self.args, "action_placeholder_id", None)
        if action_placeholder_id is None and hasattr(model, "action_placeholder_id"):
            action_placeholder_id = model.action_placeholder_id

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            action_continuous=action_continuous,
            action_placeholder_id=action_placeholder_id,
        )
        energy_pred = outputs.rewards
        energy_pred = energy_pred.squeeze(-1) if energy_pred.dim() > 1 else energy_pred
        rmse = rmse.to(dtype=energy_pred.dtype, device=energy_pred.device)

        loss_main = F.smooth_l1_loss(energy_pred, rmse)
        grad_a = torch.autograd.grad(
            outputs=energy_pred.sum(),
            inputs=action_continuous,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_norm = torch.norm(grad_a, p=2, dim=-1)
        gp_cap = float(getattr(self.args, "gp_cap", 1.0))
        lambda_gp = float(getattr(self.args, "lambda_gp", 0.1))
        loss_gp = F.relu(grad_norm - gp_cap).pow(2).mean()

        loss = loss_main + lambda_gp * loss_gp
        logged_energy = energy_pred.unsqueeze(-1)
        return (loss, dict(logits=logged_energy)) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False, alpha=0.1):
        """
        Computes the loss for the reward model training.
        """
        loss_type = getattr(self.args, "rm_loss_type", "pairwise")
        if loss_type == "energy":
            return self._compute_energy_loss(
                model=model,
                inputs=inputs,
                return_outputs=return_outputs,
            )
        return self._compute_pairwise_loss(
            model=model,
            inputs=inputs,
            return_outputs=return_outputs,
            alpha=alpha,
        )


def compute_reward_modeling_metrics(eval_prediction: EvalPrediction) -> Dict:
    # eval_prediction.label_ids is a tuple that matches up with `training_args.label_names`.
    logits = torch.tensor(
        eval_prediction.predictions[..., 0] - eval_prediction.predictions[..., 1]
    ).squeeze(-1)
    labels = torch.tensor(eval_prediction.label_ids[-1]).squeeze(-1)
    predictions = (logits >= 0.0).long()
    accuracy = predictions.eq(labels).float().mean().item()
    label_positive_rate = (labels == 1).float().mean().item()
    average_score = torch.tensor(eval_prediction.predictions).float().mean().item()
    return dict(
        accuracy=accuracy,
        label_positive_rate=label_positive_rate,
        average_score=average_score,
    )


def compute_energy_modeling_metrics(eval_prediction: EvalPrediction) -> Dict:
    predictions = torch.tensor(eval_prediction.predictions).squeeze(-1).float()
    labels = eval_prediction.label_ids
    if isinstance(labels, tuple):
        labels = labels[0]
    labels = torch.tensor(labels).squeeze(-1).float()

    mse = F.mse_loss(predictions, labels).item()
    mae = torch.mean(torch.abs(predictions - labels)).item()

    pred_centered = predictions - predictions.mean()
    label_centered = labels - labels.mean()
    denom = torch.sqrt(torch.sum(pred_centered**2) * torch.sum(label_centered**2))
    if denom.item() > 0:
        pearson = torch.sum(pred_centered * label_centered).item() / denom.item()
    else:
        pearson = 0.0

    return dict(
        mse=mse,
        mae=mae,
        pearson=pearson,
        average_energy=predictions.mean().item(),
        average_rmse=labels.mean().item(),
    )


def load_4bit_reward_model_for_inference(
    checkpoint_dir: str,
    vision_tower: str = None,
    lora_modules: list = None,
    image_aspect_ratio: str = "square",
    image_grid_pinpoints: int = None,
    bits: int = 4,
    fp16: bool = False,
    bf16: bool = False,
    double_quant: bool = True,
    quant_type: str = "nf4",
    gradient_checkpointing: bool = False,
    adapter_name="lora_default",
    is_trainable=True,
    reuse_base_model=False,
    trust_remote_code=False,
    reward_output_activation: str = "identity",
    action_dim: int = 7,
    action_token_start: int = 31744,
    action_token_end: int = 31999,
    action_value_min: Optional[float] = None,
    action_value_max: Optional[float] = None,
    action_placeholder_id: Optional[int] = None,
):
    # Load the model.
    lora_checkpoint_dir = checkpoint_dir
    if os.path.exists(os.path.join(lora_checkpoint_dir, "adapter_model")):
        lora_checkpoint_dir = os.path.join(lora_checkpoint_dir, "adapter_model")
    if os.path.exists(os.path.join(lora_checkpoint_dir, "lora_default")):
        lora_checkpoint_dir = os.path.join(lora_checkpoint_dir, "lora_default")

    lora_config = LoraConfig.from_pretrained(lora_checkpoint_dir)
    config = RewardConfig(
        backbone_model_name_or_path=lora_config.base_model_name_or_path
    )

    args = Namespace(
        model_name_or_path=config.backbone_model_name_or_path,
        vision_tower=vision_tower,
        lora_modules=lora_modules,
        image_aspect_ratio=image_aspect_ratio,
        image_grid_pinpoints=image_grid_pinpoints,
        bits=bits,
        fp16=fp16,
        bf16=bf16,
        double_quant=double_quant,
        quant_type=quant_type,
        trust_remote_code=trust_remote_code,
        full_finetune=False,
        gradient_checkpointing=gradient_checkpointing,
    )

    model = RewardModel(
        args,
        config,
        checkpoint_dir=checkpoint_dir,
        qlora=bits == 4 or bits == 8,
        adapter_name=adapter_name,
        is_trainable=is_trainable,
        reuse_base_model=reuse_base_model,
        reward_output_activation=reward_output_activation,
        action_dim=action_dim,
        action_token_start=action_token_start,
        action_token_end=action_token_end,
        action_value_min=action_value_min,
        action_value_max=action_value_max,
        action_placeholder_id=action_placeholder_id,
    )
    return model
