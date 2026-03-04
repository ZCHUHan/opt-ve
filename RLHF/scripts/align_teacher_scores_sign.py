#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, List, Optional, Tuple

import torch


def _is_pairwise_sample(sample: dict) -> bool:
    return all(k in sample for k in ("output_1", "output_2", "nrmse_0", "nrmse_1"))


def _resolve_index(
    index: int,
    source_is_pairwise: bool,
    expand_pairwise_samples: bool,
) -> Tuple[int, Optional[int]]:
    if source_is_pairwise and expand_pairwise_samples:
        return index // 2, index % 2
    return index, None


def _extract_rmse_for_index(
    sample: dict,
    pair_candidate_idx: Optional[int],
) -> float:
    if "rmse" in sample:
        return float(sample["rmse"])
    if _is_pairwise_sample(sample):
        idx = 0 if pair_candidate_idx is None else int(pair_candidate_idx)
        return float(sample["nrmse_0"] if idx == 0 else sample["nrmse_1"])
    raise KeyError(
        "Dataset sample must contain either `rmse` or pairwise fields "
        "`output_1/output_2/nrmse_0/nrmse_1`."
    )


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = torch.sqrt(torch.sum(x_c * x_c) * torch.sum(y_c * y_c))
    if float(denom) <= 0:
        return 0.0
    return float(torch.sum(x_c * y_c) / denom)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Align offline teacher score sign for energy training by checking "
            "corr(teacher_score, rmse). If negative, scores are flipped."
        )
    )
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--teacher_scores", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--energy_expand_pairwise_samples", action="store_true")
    ap.add_argument(
        "--min_abs_corr",
        type=float,
        default=0.02,
        help=(
            "Minimum |corr(score, rmse)| after alignment. "
            "Used with --strict to fail when KD signal is too weak."
        ),
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail when |corr_after| < min_abs_corr after sign alignment.",
    )
    args = ap.parse_args()

    with open(args.dataset_path, "r") as fin:
        data = json.load(fin)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("dataset_path must be a non-empty JSON list.")

    raw_scores: Any = torch.load(args.teacher_scores, map_location="cpu")
    if isinstance(raw_scores, dict) and "scores" in raw_scores:
        scores = torch.as_tensor(raw_scores["scores"], dtype=torch.float32).view(-1)
    else:
        scores = torch.as_tensor(raw_scores, dtype=torch.float32).view(-1)

    source_is_pairwise = _is_pairwise_sample(data[0]) and (
        ("action" not in data[0]) or ("rmse" not in data[0])
    )
    expand_pairwise = bool(args.energy_expand_pairwise_samples)
    expected_len = len(data) * 2 if (source_is_pairwise and expand_pairwise) else len(data)
    if int(scores.numel()) != int(expected_len):
        raise ValueError(
            f"teacher_scores length {scores.numel()} != expected dataset length {expected_len}. "
            "Check dataset_path / teacher_scores / energy_expand_pairwise_samples."
        )

    rmse_list: List[float] = []
    for i in range(expected_len):
        base_idx, pair_idx = _resolve_index(i, source_is_pairwise, expand_pairwise)
        rmse_list.append(_extract_rmse_for_index(data[base_idx], pair_idx))
    rmse = torch.tensor(rmse_list, dtype=torch.float32)

    corr_before = _pearson_corr(scores, rmse)
    flipped = False
    aligned_scores = scores
    if corr_before < 0:
        aligned_scores = -scores
        flipped = True
    corr_after = _pearson_corr(aligned_scores, rmse)

    abs_corr_after = abs(corr_after)
    if args.strict and abs_corr_after < float(args.min_abs_corr):
        raise ValueError(
            "Teacher score sign aligned but correlation magnitude is too low: "
            f"|corr_after|={abs_corr_after:.6f} < min_abs_corr={args.min_abs_corr:.6f}. "
            "KD target may be too noisy for reliable distillation."
        )
    if (not args.strict) and abs_corr_after < float(args.min_abs_corr):
        print(
            "[align_teacher_scores_sign][warning] "
            f"|corr_after|={abs_corr_after:.6f} < min_abs_corr={args.min_abs_corr:.6f}. "
            "KD target may be weak/noisy."
        )

    report = {
        "dataset_path": os.path.abspath(args.dataset_path),
        "teacher_scores_path": os.path.abspath(args.teacher_scores),
        "source_is_pairwise": bool(source_is_pairwise),
        "energy_expand_pairwise_samples": bool(expand_pairwise),
        "num_scores": int(aligned_scores.numel()),
        "corr_before": float(corr_before),
        "flipped": bool(flipped),
        "corr_after": float(corr_after),
        "min_abs_corr": float(args.min_abs_corr),
        "strict": bool(args.strict),
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if isinstance(raw_scores, dict) and "scores" in raw_scores:
        out_obj = dict(raw_scores)
        out_obj["scores"] = aligned_scores
        out_obj["sign_alignment"] = report
        torch.save(out_obj, args.output)
    else:
        torch.save(aligned_scores, args.output)

    report_path = args.output + ".sign_alignment.json"
    with open(report_path, "w") as fout:
        json.dump(report, fout, indent=2)

    print("[align_teacher_scores_sign] done")
    print(f"  corr_before={corr_before:.6f}")
    print(f"  flipped={flipped}")
    print(f"  corr_after={corr_after:.6f}")
    print(f"  output={args.output}")
    print(f"  report={report_path}")


if __name__ == "__main__":
    main()
