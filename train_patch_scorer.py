#!/usr/bin/env python3
"""
Standalone training script for PatchScorerModel only.

Trains just the patch_scorer module (vision_enhancer + text_enhancer) inside the
FocusUI-Qwen2.5-VL wrapper. All other model parameters are frozen and the full
LM decoder is never run (patch_scorer_early_exit=True), so this is much cheaper
than running train_focusui.py.

Usage:
    python train_patch_scorer.py \
        --model_name_or_path /path/to/Qwen2.5-VL-3B-Instruct \
        --data_path data/data_config.yaml \
        --image_folder "" \
        --output_dir checkpoints/patch_scorer
"""

import argparse
import os
from types import SimpleNamespace

import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from focusui.constants import (
    ADDITIONAL_SPECIAL_TOKENS,
    ADDITIONAL_SPECIAL_TOKENS_IMAGE_DROP,
)
from focusui.dataset import LazySupervisedDataset
from focusui.modeling_focusui_qwen25vl import FocusUI_Qwen2_5_VLForConditionalGenerationWithPointer
from train_focusui import (
    DataCollatorForSupervisedDataset,
    smart_tokenizer_and_embedding_resize,
    update_pointer_token_ids,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train only the PatchScorer module")

    # model / data
    p.add_argument("--model_name_or_path", required=True,
                   help="HuggingFace path or local dir for Qwen2.5-VL / FocusUI model")
    p.add_argument("--data_path", required=True,
                   help="Path to .json or .yaml dataset config (same format as train_focusui.py)")
    p.add_argument("--image_folder", default="",
                   help="Root prefix for image paths in the dataset")
    p.add_argument("--output_dir", required=True,
                   help="Directory to save patch_scorer checkpoints")
    p.add_argument("--patch_scorer_ckpt", default=None,
                   help="Optional: path to an existing patch_scorer .pt file to resume from")

    # image resolution (should match the run that produced patch_scores_label)
    p.add_argument("--min_pixels", type=int, default=3136)
    p.add_argument("--max_pixels", type=int, default=5720064)
    p.add_argument("--model_max_length", type=int, default=24576)
    p.add_argument("--grounding_system_message", default="qwen25vl",
                   choices=["qwen25vl", "qwen3vl"])

    # training hyper-params
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Must be 1: patch counts vary per image so batching > 1 is unsupported")
    p.add_argument("--grad_accum_steps", type=int, default=32,
                   help="Accumulate gradients over this many data steps before one optimizer step")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--save_steps", type=int, default=5000,
                   help="Save a checkpoint every N optimizer steps")
    p.add_argument("--logging_steps", type=int, default=10,
                   help="Log loss every N optimizer steps")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--dataloader_num_workers", type=int, default=8)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load full FocusUI model ─────────────────────────────────────────
    print(f"Loading model from {args.model_name_or_path} ...")
    model = FocusUI_Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=False,
    )
    # Skip the LM decoder entirely during scorer training
    model.patch_scorer_early_exit = True
    # No need for token selection bookkeeping
    model.apply_visual_token_select = False

    # ── 2. Tokenizer + special tokens ─────────────────────────────────────
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
    )
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict={
            "additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS + ADDITIONAL_SPECIAL_TOKENS_IMAGE_DROP
        },
        tokenizer=tokenizer,
        model=model,
    )
    update_pointer_token_ids(model.config, tokenizer)

    # ── 3. Processor ───────────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    processor.tokenizer = tokenizer

    # ── 4. Freeze everything except patch_scorer ───────────────────────────
    print("Freezing all parameters except patch_scorer ...")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.patch_scorer.parameters():
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_trainable:,} / {n_total:,} parameters")

    # ── 5. Optionally load a patch_scorer checkpoint ───────────────────────
    if args.patch_scorer_ckpt:
        print(f"Loading patch_scorer weights from {args.patch_scorer_ckpt} ...")
        ckpt = torch.load(args.patch_scorer_ckpt, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if any(k.startswith("module.") for k in ckpt):
            ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
        model.patch_scorer.load_state_dict(ckpt, strict=False)

    model = model.to(device)

    # ── 6. Dataset & DataLoader ────────────────────────────────────────────
    data_args = SimpleNamespace(
        image_folder=args.image_folder,
        grounding_system_message=args.grounding_system_message,
        max_conv_turns=10,
        shuffle=True,
        early_mix_text=False,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    print(f"Loading dataset from {args.data_path} ...")
    dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        processor=processor,
        data_path=args.data_path,
        data_args=data_args,
    )
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    print(f"Dataset: {len(dataset)} samples | {len(dataloader)} steps/epoch")

    # ── 7. Optimizer & cosine LR scheduler with warmup ────────────────────
    optimizer = torch.optim.AdamW(
        model.patch_scorer.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_opt_steps = (len(dataloader) * args.num_epochs) // args.grad_accum_steps
    warmup_steps = max(1, int(total_opt_steps * args.warmup_ratio))
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps,
    )
    print(f"Total optimizer steps: {total_opt_steps} | warmup: {warmup_steps}")

    # ── 8. Training loop ───────────────────────────────────────────────────
    # Frozen backbone stays in eval (disables its dropout).
    # Only patch_scorer is set to train mode.
    model.eval()
    model.patch_scorer.train()

    global_step = 0
    accum_loss = 0.0   # loss accumulated within one grad_accum block
    log_loss = 0.0     # loss accumulated across logging_steps opt steps
    skipped = 0
    last_data_step = -1

    optimizer.zero_grad()

    for epoch in range(args.num_epochs):
        for data_step, batch in enumerate(dataloader):
            last_data_step = data_step

            # Skip incomplete batches
            if (
                batch.get("pixel_values") is None
                or batch.get("focus_input_ids") is None
                or batch.get("patch_scores_label") is None
            ):
                skipped += 1
                continue

            # Move inputs to device
            input_ids       = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
            pixel_values    = batch["pixel_values"].to(device)
            image_grid_thw  = batch["image_grid_thw"].to(device)
            focus_input_ids = batch["focus_input_ids"].to(device)
            focus_attn_mask = batch["focus_attention_mask"].to(device)
            patch_scores_label = batch["patch_scores_label"]  # list[Tensor], handled by model

            # Forward (early-exits after PatchScorer; LM decoder never runs)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if args.bf16 else torch.float32):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    focus_input_ids=focus_input_ids,
                    focus_attention_mask=focus_attn_mask,
                    patch_scores_label=patch_scores_label,
                )

            loss = outputs.loss
            if loss is None:
                skipped += 1
                continue

            (loss / args.grad_accum_steps).backward()
            accum_loss += loss.item() / args.grad_accum_steps

            # Optimizer step after accumulating grad_accum_steps data steps
            if (data_step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.patch_scorer.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                log_loss += accum_loss
                accum_loss = 0.0
                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg_loss = log_loss / args.logging_steps
                    lr_now = scheduler.get_last_lr()[0]
                    print(
                        f"[epoch {epoch+1}/{args.num_epochs}] "
                        f"step={global_step} | loss={avg_loss:.4f} | "
                        f"lr={lr_now:.2e} | skipped={skipped}"
                    )
                    log_loss = 0.0
                    skipped = 0

                if global_step % args.save_steps == 0:
                    ckpt_path = os.path.join(args.output_dir, f"patch_scorer_step{global_step}.pt")
                    torch.save(model.patch_scorer.state_dict(), ckpt_path)
                    print(f"Saved → {ckpt_path}")

    # Flush any leftover accumulated gradients from the last partial block
    if last_data_step >= 0 and (last_data_step + 1) % args.grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.patch_scorer.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # ── 9. Final save ──────────────────────────────────────────────────────
    final_path = os.path.join(args.output_dir, "patch_scorer_final.pt")
    torch.save(model.patch_scorer.state_dict(), final_path)
    print(f"\nTraining complete. Final checkpoint saved to {final_path}")


if __name__ == "__main__":
    main()
