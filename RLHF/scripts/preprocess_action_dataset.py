#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from data_utils.action_space import ActionSpaceSpec

_ACTION_PATTERN = re.compile(r"\[([\d,\s\.\-eE]+)\]")


def _parse_action_array(text: str) -> Optional[List[float]]:
    match = _ACTION_PATTERN.search(text)
    if match is None:
        return None
    values = []
    for token in match.group(1).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def _format_number(v: float, digits: int = 6) -> str:
    s = f"{float(v):.{digits}f}".rstrip("0").rstrip(".")
    if s == "-0":
        return "0"
    return s


def _format_action_array(values: Sequence[float], as_int: bool) -> str:
    if as_int:
        return "[" + ", ".join(str(int(round(float(v)))) for v in values) + "]"
    return "[" + ", ".join(_format_number(float(v)) for v in values) + "]"


def _replace_action_array_in_text(
    text: str,
    convert_fn: Callable[[Sequence[float]], Sequence[float]],
    as_int: bool,
) -> Tuple[str, bool, Optional[List[float]], Optional[List[float]]]:
    match = _ACTION_PATTERN.search(text)
    if match is None:
        return text, False, None, None
    before = text[: match.start()]
    after = text[match.end() :]
    src = _parse_action_array(text)
    if src is None:
        return text, False, None, None
    dst = list(convert_fn(src))
    new_text = before + _format_action_array(dst, as_int=as_int) + after
    return new_text, True, src, dst


def _convert_example(
    example: Dict,
    spec: ActionSpaceSpec,
    mode: str,
    keep_original: bool,
) -> Tuple[Dict, Dict[str, int]]:
    # Work on a shallow copy so we never mutate the original in-place.
    ex = dict(example)
    stats = {
        "examples_seen": 1,
        "action_list_converted": 0,
        "output_text_converted": 0,
        "conversation_text_converted": 0,
        "skipped_non_token_like": 0,
    }

    if mode == "token_to_continuous":
        def convert_action(arr: Sequence[float]) -> Sequence[float]:
            return spec.token_ids_to_continuous(arr, strict=True)
        out_as_int = False
    else:
        def convert_action(arr: Sequence[float]) -> Sequence[float]:
            return spec.continuous_to_token_ids(arr, clamp=True)
        out_as_int = True

    if "action" in ex and isinstance(ex["action"], list):
        action = ex["action"]
        should_convert = True
        if mode == "token_to_continuous" and not spec.looks_like_token_ids(action):
            should_convert = False
        if should_convert:
            converted = list(convert_action(action))
            if keep_original:
                backup_key = "action_token_ids" if mode == "token_to_continuous" else "action_continuous"
                ex.setdefault(backup_key, list(action))
            ex["action"] = converted
            stats["action_list_converted"] += 1
        else:
            stats["skipped_non_token_like"] += 1

    for out_key in ("output_1", "output_2"):
        block = ex.get(out_key, None)
        if not (isinstance(block, dict) and isinstance(block.get("value", None), str)):
            continue
        text = block["value"]
        arr = _parse_action_array(text)
        if arr is None:
            continue
        if mode == "token_to_continuous" and not spec.looks_like_token_ids(arr):
            stats["skipped_non_token_like"] += 1
            continue
        new_text, changed, src, dst = _replace_action_array_in_text(
            text=text, convert_fn=convert_action, as_int=out_as_int
        )
        if changed:
            if keep_original and src is not None:
                backup_key = (
                    f"{out_key}_action_token_ids"
                    if mode == "token_to_continuous"
                    else f"{out_key}_action_continuous"
                )
                ex.setdefault(backup_key, src)
            block = dict(block)
            block["value"] = new_text
            if keep_original and dst is not None:
                converted_key = (
                    f"{out_key}_action_continuous"
                    if mode == "token_to_continuous"
                    else f"{out_key}_action_token_ids"
                )
                block[converted_key] = list(dst)
            ex[out_key] = block
            stats["output_text_converted"] += 1

    conversations = ex.get("conversations", None)
    if isinstance(conversations, list):
        new_conversations = []
        conv_changed = 0
        for turn in conversations:
            if not isinstance(turn, dict) or not isinstance(turn.get("value", None), str):
                new_conversations.append(turn)
                continue
            text = turn["value"]
            arr = _parse_action_array(text)
            if arr is None:
                new_conversations.append(turn)
                continue
            if mode == "token_to_continuous" and not spec.looks_like_token_ids(arr):
                stats["skipped_non_token_like"] += 1
                new_conversations.append(turn)
                continue
            new_text, changed, _, _ = _replace_action_array_in_text(
                text=text, convert_fn=convert_action, as_int=out_as_int
            )
            if changed:
                t = dict(turn)
                t["value"] = new_text
                new_conversations.append(t)
                conv_changed += 1
            else:
                new_conversations.append(turn)
        if conv_changed > 0:
            ex["conversations"] = new_conversations
            stats["conversation_text_converted"] += conv_changed

    return ex, stats


def _merge_stats(all_stats: Dict[str, int], cur: Dict[str, int]) -> None:
    for k, v in cur.items():
        all_stats[k] = int(all_stats.get(k, 0) + int(v))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert action fields in dataset JSON with a canonical action space mapping. "
            "Recommended path: token_id -> continuous for energy training."
        )
    )
    parser.add_argument("--input", required=True, help="Input json file (array of examples).")
    parser.add_argument("--output", required=True, help="Output json file.")
    parser.add_argument(
        "--mode",
        choices=("token_to_continuous", "continuous_to_token"),
        default="token_to_continuous",
        help="Mapping direction.",
    )
    parser.add_argument("--action_token_start", type=int, default=31744)
    parser.add_argument("--action_token_end", type=int, default=31999)
    parser.add_argument("--action_value_min", type=float, default=-1.0)
    parser.add_argument("--action_value_max", type=float, default=1.0)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument(
        "--mapping_style",
        choices=("openvla_digitize", "linear"),
        default="openvla_digitize",
        help=(
            "Action mapping style. `openvla_digitize` matches ActionTokenizer "
            "(np.digitize + bin_centers)."
        ),
    )
    parser.add_argument(
        "--keep_original",
        action="store_true",
        help="Keep backup fields (recommended for audit/rollback).",
    )
    parser.add_argument(
        "--write_sidecar",
        action="store_true",
        help="Write <output>.action_space.json with mapping metadata.",
    )
    args = parser.parse_args()

    spec = ActionSpaceSpec(
        action_token_start=args.action_token_start,
        action_token_end=args.action_token_end,
        action_value_min=args.action_value_min,
        action_value_max=args.action_value_max,
        action_dim=args.action_dim,
        mapping_style=args.mapping_style,
    )
    spec.validate()

    with open(args.input, "r") as fin:
        data = json.load(fin)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of examples.")

    out_data = []
    stats = {
        "examples_seen": 0,
        "action_list_converted": 0,
        "output_text_converted": 0,
        "conversation_text_converted": 0,
        "skipped_non_token_like": 0,
    }
    for ex in data:
        new_ex, cur_stats = _convert_example(
            example=ex,
            spec=spec,
            mode=args.mode,
            keep_original=bool(args.keep_original),
        )
        out_data.append(new_ex)
        _merge_stats(stats, cur_stats)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as fout:
        json.dump(out_data, fout, indent=2, ensure_ascii=False)

    if args.write_sidecar:
        sidecar = {
            "mode": args.mode,
            "action_token_start": int(spec.action_token_start),
            "action_token_end": int(spec.action_token_end),
            "action_value_min": float(spec.action_value_min),
            "action_value_max": float(spec.action_value_max),
            "action_dim": int(spec.action_dim),
            "num_bins": int(spec.num_bins),
            "mapping_style": str(spec.mapping_style),
            "source": os.path.abspath(args.input),
            "output": os.path.abspath(args.output),
        }
        sidecar_path = args.output + ".action_space.json"
        with open(sidecar_path, "w") as fout:
            json.dump(sidecar, fout, indent=2, ensure_ascii=False)
        print(f"[preprocess_action_dataset] wrote sidecar: {sidecar_path}")

    print("[preprocess_action_dataset] done")
    for k in sorted(stats.keys()):
        print(f"  {k}: {stats[k]}")
    print(f"  mode: {args.mode}")
    print(f"  input: {args.input}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
