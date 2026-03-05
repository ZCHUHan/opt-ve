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

from dataclasses import dataclass
import copy
import json
import logging
import os
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import transformers
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset

from data_utils.common_utils import preprocess, preprocess_multimodal

logger = logging.getLogger(__name__)

_ACTION_PATTERN = re.compile(r"\[([\d,\s\.\-eE]+)\]")


def _extract_action_array_from_text(text: str) -> List[float]:
    match = _ACTION_PATTERN.search(text)
    if match is None:
        return []
    values = []
    for token in match.group(1).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def _replace_last_assistant_with_placeholder(
    conversations: Sequence[Dict[str, str]],
    placeholder_token: str,
    action_dim: int,
) -> List[Dict[str, str]]:
    conv = copy.deepcopy(conversations)
    placeholder_response = "The robot should take the action: " + " ".join(
        [placeholder_token] * action_dim
    )

    if isinstance(conv[-1], list):
        assert len(conv) == 1
        conv = conv[-1]

    if len(conv) == 0:
        raise ValueError("Empty conversations are not supported for energy training.")

    if conv[-1]["from"] == "gpt":
        conv[-1]["value"] = placeholder_response
    else:
        conv.append({"from": "gpt", "value": placeholder_response})
    return conv


def _extract_pointwise_fields(
    example: dict,
    action_dim: int,
    pair_candidate_idx: Optional[int] = None,
) -> Tuple[List[float], float]:
    if "action" in example and "rmse" in example:
        action = [float(v) for v in example["action"]]
        rmse = float(example["rmse"])
    elif (
        "output_1" in example
        and "output_2" in example
        and "nrmse_0" in example
        and "nrmse_1" in example
    ):
        action_0 = _extract_action_array_from_text(example["output_1"]["value"])
        action_1 = _extract_action_array_from_text(example["output_2"]["value"])
        if len(action_0) == 0 or len(action_1) == 0:
            raise ValueError("Cannot parse action array from pairwise sample outputs.")
        if pair_candidate_idx is None:
            pair_candidate_idx = 0
        if pair_candidate_idx == 0:
            action = action_0
            rmse = float(example["nrmse_0"])
        else:
            action = action_1
            rmse = float(example["nrmse_1"])
    else:
        raise KeyError(
            "Energy dataset sample must contain either (action, rmse) "
            "or pairwise fields (output_1/output_2/nrmse_0/nrmse_1)."
        )

    if len(action) != action_dim:
        raise ValueError(
            f"Action length mismatch: expected {action_dim}, got {len(action)}. "
            "Please align `--action_dim` with dataset action dimensionality."
        )
    return action, rmse


class EnergyRewardModelingDataset(Dataset):
    def __init__(
        self,
        data: Sequence[dict],
        tokenizer: transformers.PreTrainedTokenizer,
        df_postprocessor: Optional[Callable] = None,
        query_len: Optional[int] = None,
        response_len: Optional[int] = None,
        use_data_frame: bool = True,
        data_args: Optional[Dict] = None,
    ):
        super(EnergyRewardModelingDataset, self).__init__()
        del data
        del df_postprocessor
        del use_data_frame

        list_data_dict = json.load(open(data_args.dataset_path, "r"))
        self.list_data_dict = list_data_dict
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.query_len = query_len
        self.response_len = response_len

        self.action_dim = getattr(data_args, "action_dim", 7)
        self.action_placeholder_token = getattr(
            data_args, "action_placeholder_token", "placeholder"
        )
        self.expand_pairwise_samples = getattr(
            data_args, "energy_expand_pairwise_samples", True
        )

        self.reward_model_prompt = None
        if data_args.reward_prompt_file is not None:
            with open(data_args.reward_prompt_file, "r") as f:
                self.reward_model_prompt = " " + f.read().strip()

        first_sample = self.list_data_dict[0] if len(self.list_data_dict) > 0 else {}
        self.source_is_pairwise = all(
            key in first_sample
            for key in ("output_1", "output_2", "nrmse_0", "nrmse_1")
        ) and ("action" not in first_sample or "rmse" not in first_sample)

        # Optional: pre-computed teacher scores for offline KD.
        # Expect a 1D tensor of length equal to len(self) (after any pairwise expansion).
        self.teacher_scores = None
        teacher_score_path = getattr(self.data_args, "teacher_score_path", None)
        if teacher_score_path:
            teacher_scores = torch.load(teacher_score_path, map_location="cpu")
            if isinstance(teacher_scores, dict) and "scores" in teacher_scores:
                teacher_scores = teacher_scores["scores"]
            teacher_scores = torch.as_tensor(teacher_scores, dtype=torch.float32)
            # Flatten in case it's accidentally saved as shape [N, 1].
            if teacher_scores.dim() > 1:
                teacher_scores = teacher_scores.view(-1)
            self.teacher_scores = teacher_scores

            if len(self.teacher_scores) != len(self):
                raise ValueError(
                    f"teacher_scores length {len(self.teacher_scores)} != dataset length {len(self)}. "
                    "Make sure teacher_score_path matches dataset_path and energy_expand_pairwise_samples."
                )

    def __len__(self):
        if self.source_is_pairwise and self.expand_pairwise_samples:
            return len(self.list_data_dict) * 2
        return len(self.list_data_dict)

    def _resolve_base_index(self, index: int) -> Tuple[int, Optional[int]]:
        if self.source_is_pairwise and self.expand_pairwise_samples:
            return index // 2, index % 2
        return index, None

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        base_i, pair_candidate_idx = self._resolve_base_index(i)
        sample = self.list_data_dict[base_i]

        if "image" in sample:
            image_file = sample["image"]
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
            if self.data_args.image_aspect_ratio == "pad":

                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(
                            pil_img.mode, (width, width), background_color
                        )
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(
                            pil_img.mode, (height, height), background_color
                        )
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(
                    image, tuple(int(x * 255) for x in processor.image_mean)
                )
                image = processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
            else:
                image = processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
            conversations = preprocess_multimodal(
                copy.deepcopy([sample["conversations"]]), self.data_args
            )[0]
        else:
            conversations = copy.deepcopy(sample["conversations"])

        action, rmse = _extract_pointwise_fields(
            sample, action_dim=self.action_dim, pair_candidate_idx=pair_candidate_idx
        )
        conversations = _replace_last_assistant_with_placeholder(
            conversations,
            placeholder_token=self.action_placeholder_token,
            action_dim=self.action_dim,
        )

        text_dict = preprocess(
            [conversations],
            tokenizer=self.tokenizer,
            has_image=("image" in sample),
            mask_target=False,
            query_len=self.query_len,
            response_len=self.response_len,
            reward_model_prompt=self.reward_model_prompt,
            image_captions=None,
        )
        data_dict = dict(
            input_ids=text_dict["input_ids"][0],
            labels=text_dict["labels"][0],
            action_continuous=torch.tensor(action, dtype=torch.float32),
            rmse=torch.tensor(rmse, dtype=torch.float32),
        )

        # Attach offline teacher score if available.
        if self.teacher_scores is not None:
            data_dict["teacher_score"] = self.teacher_scores[i].to(torch.float32)

        if "image" in sample:
            data_dict["image"] = image.to(torch.bfloat16)
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])

        return data_dict


def split_train_into_train_and_eval(
    train_dataset: Dataset, eval_size: int, seed: int = None
) -> Tuple[Dataset, Dataset]:
    del seed
    assert eval_size < len(
        train_dataset
    ), "Requested eval_size cannot be equal/larger than original train data size."

    total_size = len(train_dataset)
    batch_size = 32

    new_train_size = (total_size - eval_size) // batch_size * batch_size
    eval_size = eval_size // batch_size * batch_size

    indices = list(range(total_size))
    train_indices = indices[:new_train_size]
    eval_indices = indices[new_train_size : new_train_size + eval_size]

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    eval_subset = torch.utils.data.Subset(train_dataset, eval_indices)
    return train_subset, eval_subset


def pad_sequence_from_left(
    sequences: Sequence[torch.Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
):
    sequences = tuple(sequence.flip(0) for sequence in sequences)
    padded_sequence = torch._C._nn.pad_sequence(
        sequences, batch_first, padding_value
    )  # noqa
    padded_sequence = padded_sequence.flip(int(batch_first))
    return padded_sequence


@dataclass
class DataCollatorForEnergyRewardModelingDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        input_ids = pad_sequence_from_left(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        action_continuous = torch.stack(
            [instance["action_continuous"] for instance in instances]
        )
        rmse = torch.stack([instance["rmse"] for instance in instances]).view(-1)

        batch = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_continuous=action_continuous,
            rmse=rmse,
        )

        # Optional offline teacher scores for KD.
        if "teacher_score" in instances[0]:
            batch["teacher_score"] = torch.stack(
                [instance["teacher_score"] for instance in instances]
            ).view(-1)

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        return batch


def make_energy_reward_modeling_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
):
    if data_args.dataset_path.endswith("json"):
        train_preference = load_dataset(
            "json", data_files=data_args.dataset_path, streaming=True
        )["train"]
        use_data_frame = False
    else:
        raise ValueError(
            f"Unsupported dataset_path: {data_args.dataset_path}."
            "Only json datasets are supported."
        )

    train_dataset = EnergyRewardModelingDataset(
        data=train_preference,
        tokenizer=tokenizer,
        query_len=training_args.query_len,
        response_len=training_args.response_len,
        use_data_frame=use_data_frame,
        data_args=data_args,
    )

    if (
        data_args.dataset_path == data_args.eval_dataset_path
        and data_args.dataset_name == data_args.eval_dataset_name
    ):
        train_dataset, eval_dataset = split_train_into_train_and_eval(
            train_dataset=train_dataset,
            eval_size=data_args.eval_size,
            seed=training_args.seed,
        )
    else:
        raise NotImplementedError

    data_collator = DataCollatorForEnergyRewardModelingDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
