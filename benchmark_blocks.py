"""
Benchmark the inference latency of visual encoder and patch scorer for FocusUI Qwen2_5_VL.

This script measures the two key stages separately:
  1. visual encoder latency (image -> image_embeds)
  2. patch scorer latency (image_embeds + text_embeds -> patch_scores)

Example usage:
    python ./benchmark_blocks.py   --model-path ./checkpoints/FocusUI-3B   --image-path ./assets/example_screenshot.png   --text "Go to 'Watch Live"   --iters 10
"""

import argparse
import time
from pathlib import Path
import json
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoTokenizer
from tqdm import tqdm
from focusui.modeling_focusui_qwen25vl import FocusUI_Qwen2_5_VLForConditionalGenerationWithPointer
from focusui.preprocess_focusui import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark visual encoder and patch scorer latency for FocusUI Qwen2_5_VL."
    )
    parser.add_argument("--model-path", type=str, required=True, help="Path or model ID for the Qwen2_5_VL FocusUI checkpoint")
    parser.add_argument("--image-path", type=str, required=True, help="Path to an RGB image for benchmarking")
    parser.add_argument("--text", type=str, default="Locate the most important UI element on the screen.", help="Instruction text for patch scorer embedding")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the benchmark on")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=10, help="Number of benchmark iterations")
    return parser.parse_args()


def load_model(model_path: str, device: torch.device):
    kwargs = {}
    # if device.type == "cuda":
    #     kwargs["device_map"] = "cuda"
    #     kwargs["torch_dtype"] = torch.bfloat16

    model = FocusUI_Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        **kwargs,
    )
    return model.eval()


def prepare_inputs(image_path: str, text: str, tokenizer, image_processor, device: torch.device):
    image = Image.open(image_path).convert("RGB")
    image_inputs = image_processor(images=image, return_tensors="pt")

    pixel_values = image_inputs["pixel_values"].to(device)
    image_grid_thw = image_inputs.get("image_grid_thw")
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)

    text_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)

    return pixel_values, image_grid_thw, input_ids, attention_mask

def benchmark_uigraph_score(iters: int, images: list, scorer_type: str):
    times = []
    pbar = tqdm(range(iters), desc="🚀 Benchmarking")

    for _ in pbar:
        # print(f"🚀 Benchmarking: {i+1}/{iters}...", end="\r")

        for image in images:
            t0 = time.perf_counter()
            _ = preprocess_focusui_data(image, ui_graph_score_type=scorer_type)
            t1 = time.perf_counter()
            curr_time = t1 - t0
            times.append(curr_time)
            pbar.set_postfix({
                "latency": f"{curr_time:.4f}s",
                "avg": f"{sum(times)/len(times):.4f}"
            })

    # print(f"\n✅ Benchmark complete for {iters} iterations.")
    return times

def benchmark(fn, warmup: int, iters: int, device: torch.device):

    if warmup > 0:
        for _ in tqdm(range(warmup), desc="🔥 Warmup", leave=False):
            _ = fn()

    times = []
    last_output = None

    pbar = tqdm(range(iters, desc="🚀 Benchmarking"))
    for _ in pbar:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        last_output = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        curr_time = t1 - t0
        times.append(curr_time)

        # 获取当前显存占用 (GB)
        mem_stats = ""
        if device.type == "cuda":
            # 获取当前峰值显存
            peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
            mem_stats = f"{peak_mem:.2f}GB"
            torch.cuda.empty_cache() # 你的需求：每轮清理
            # 重置峰值统计，以便下一轮测到的是单次 forward 的真实峰值
            torch.cuda.reset_peak_memory_stats(device)

        # 实时更新进度条右侧的统计信息
        pbar.set_postfix({
            "latency": f"{curr_time:.4f}s",
            "peak_vram": mem_stats,
            "avg": f"{sum(times)/len(times):.4f}s"
        })

        # if device.type == "cuda":
        #     torch.cuda.empty_cache() # 既然是做 Benchmark，每一轮清理一下能获得更准确的 Peak VRAM 数据
    # print(f"\n✅ Benchmark complete for {iters} iterations.")
    return times, last_output


def summarize(name: str, times):

    times = sorted(times)
    n = len(times)

    # 计算统计数据
    stats = {
        "name": name,
        "mean": sum(times) / n,
        "p50": times[n // 2],
        "p90": times[int(n * 0.9) - 1],
        "p95": times[int(n * 0.95) - 1],
        "min": times[0],
        "max": times[-1],
        "count": n
    }
    return stats


def test_benchmark_uigraph_score():
    args = parse_args()
    print("Benchmark uigraph score...")
    test_image = Image.open(args.image_path)
    uigraph_score_times = benchmark_uigraph_score(10, [test_image], "l2-norm")
    result = summarize("uigraph score", uigraph_score_times)
    print(result)

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print(f"Loading model from {args.model_path} on {device}...")
    model = load_model(args.model_path, device)

    print("Loading tokenizer and image processor...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    image_processor = AutoImageProcessor.from_pretrained(args.model_path)

    pixel_values, image_grid_thw, input_ids, attention_mask = prepare_inputs(
        args.image_path,
        args.text,
        tokenizer,
        image_processor,
        device,
    )

    results = {}

    def encode_image():
        if hasattr(model, "get_image_features"):
            result = model.get_image_features(pixel_values, image_grid_thw)
            return result[0] if isinstance(result, tuple) else result
        if hasattr(model, "visual"):
            return model.visual(pixel_values, grid_thw=image_grid_thw)
        raise RuntimeError("Model does not expose a visual encoder method.")

    print("Warmup and benchmark visual encoder...")
    image_times, image_embeds = benchmark(encode_image, args.warmup, args.iters, device)
    if isinstance(image_embeds, tuple):
        image_embeds = image_embeds[0]
    results["encoder_result"] = summarize("Visual encoder", image_times)

    def scorer_forward():
        text_embeds = model.get_input_embeddings()(input_ids)
        return model.patch_scorer(
            image_embeds=image_embeds,
            text_embeds=text_embeds,
            return_dict=True,
        )["patch_scores"]

    print("Warmup and benchmark patch scorer...")
    scorer_times, patch_scores = benchmark(scorer_forward, args.warmup, args.iters, device)
    results["scorer_result"] = summarize("Patch scorer", scorer_times)
    print(f"Output patch_scores shape: {patch_scores.shape}")


    print("Benchmark uigraph score using l2-norm...")
    test_image = Image.open(args.image_path)
    uigraph_score_times_l2_norm = benchmark_uigraph_score(10, [test_image], "l2-norm")
    results["uigraph_score_l2_norm"] = summarize("uigraph score l2-norm", uigraph_score_times_l2_norm)

    print("Benchmark uigraph score using ssim")
    test_image = Image.open(args.image_path)
    uigraph_score_times_ssim = benchmark_uigraph_score(10, [test_image], "ssim")
    results["uigraph_score_ssim"] = summarize("uigraph score ssim", uigraph_score_times_ssim)

    # TODO: Storing results to json and csv file.


if __name__ == "__main__":
    test_benchmark_uigraph_score()
    # main()
