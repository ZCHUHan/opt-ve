import argparse, json
import numpy as np
import ijson
import decimal

def reservoir_sample(stream, k, seed=42):
    rng = np.random.default_rng(seed)
    sample = []
    seen = 0
    for seen, item in enumerate(stream, start=1):
        if len(sample) < k:
            sample.append(item)
        else:
            j = rng.integers(0, seen)
            if j < k:
                sample[j] = item
    return sample, seen

def head_sample(stream, k):
    out = []
    seen = 0
    for seen, item in enumerate(stream, start=1):
        out.append(item)
        if len(out) >= k:
            break
    return out, seen

def json_default(o):
    if isinstance(o, decimal.Decimal):
        return float(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["reservoir", "head"], default="reservoir")
    args = ap.parse_args()

    with open(args.input, "rb") as f:
        stream = ijson.items(f, "item")  # JSON array elements
        if args.mode == "head":
            subset, seen = head_sample(stream, args.num)
        else:
            subset, seen = reservoir_sample(stream, args.num, seed=args.seed)

    with open(args.output, "w") as w:
        json.dump(subset, w, default=json_default)

    print(f"[subset] mode={args.mode} requested={args.num} wrote={len(subset)} seen={seen} -> {args.output}")

if __name__ == "__main__":
    main()