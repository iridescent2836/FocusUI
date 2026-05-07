#!/usr/bin/env python3
"""
Convert a HuggingFace safetensors checkpoint to bfloat16 in-place.

DeepSpeed saves fp32 master weights by default, doubling disk usage. After
training, run this to halve checkpoint size with no inference accuracy loss.

Usage:
    python convert_ckpt_to_bf16.py checkpoints/focusui_3b_ft_scorer
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def convert(ckpt_dir: Path, dtype: torch.dtype = torch.bfloat16):
    files = sorted(ckpt_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files under {ckpt_dir}")

    total_before = total_after = 0
    for f in files:
        size_before = f.stat().st_size
        total_before += size_before

        tensors = {}
        with safe_open(f, framework="pt") as src:
            metadata = src.metadata() or {}
            for k in src.keys():
                t = src.get_tensor(k)
                if t.is_floating_point() and t.dtype != dtype:
                    t = t.to(dtype)
                tensors[k] = t

        tmp = f.with_suffix(".safetensors.tmp")
        save_file(tensors, str(tmp), metadata=metadata)
        shutil.move(str(tmp), str(f))

        size_after = f.stat().st_size
        total_after += size_after
        print(f"  {f.name}: {size_before/1e9:.2f} GB -> {size_after/1e9:.2f} GB")

    # Update the index file's total_size if present
    index_files = list(ckpt_dir.glob("*.safetensors.index.json"))
    for idx_path in index_files:
        idx = json.loads(idx_path.read_text())
        if "metadata" in idx and "total_size" in idx["metadata"]:
            idx["metadata"]["total_size"] = total_after
            idx_path.write_text(json.dumps(idx, indent=2))
            print(f"  updated {idx_path.name} total_size -> {total_after}")

    print(f"\nTotal: {total_before/1e9:.2f} GB -> {total_after/1e9:.2f} GB "
          f"({100*(1-total_after/total_before):.1f}% smaller)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", type=Path,
                    help="Directory containing model-*.safetensors files")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16"])
    args = ap.parse_args()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    convert(args.ckpt_dir, dtype)
