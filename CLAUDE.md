# FocusUI

UI grounding model built on top of Qwen2.5-VL. Core idea: a lightweight `PatchScorerModel` predicts patch importance scores from image + instruction embeddings, those scores are used to selectively drop low-importance visual tokens before the LM decoder runs, reducing compute while preserving grounding accuracy.

## Architecture

```
pixel_values ──► visual encoder (Qwen2.5-VL frozen) ──► image_embeds
focus_input_ids ► embed_tokens (frozen)               ──► text_embeds
                                                              │
                                              PatchScorerModel (trainable)
                                                              │
                                              patch_scores [B, V]
                                                              │
                              visual token selection (drop low-score patches)
                                                              │
                                              LM decoder ──► logits / loss
```

**`PatchScorerModel`** (`focusui/modeling_patch_scorer.py`):
- Two `MHATokenFeatureEnhancer` modules: `vision_enhancer` and `text_enhancer`
- Computes cosine similarity between enhanced image patches and text tokens
- Loss: KL divergence against `patch_scores_label` (ground truth = bbox overlap score)
- `projection_dim` defaults to `vision_config.out_hidden_size` (1536 for 3B)

**`FocusUI_Qwen2_5_VLForConditionalGenerationWithPointer`** (`focusui/modeling_focusui_qwen25vl.py`):
- Wraps `Qwen2_5_VLForConditionalGeneration`
- Adds `patch_scorer`, `multi_patch_pointer_head` (for bbox supervision)
- `patch_scorer_early_exit=True`: returns immediately after patch scorer, skipping LM decoder entirely (used for scorer-only training)
- Combined loss: `lm_loss_weight * lm_loss + pointer_loss_weight * pointer_loss + ps_loss_weight * ps_loss`

## Key Files

| File | Purpose |
|---|---|
| `focusui/modeling_patch_scorer.py` | `PatchScorerModel` definition |
| `focusui/modeling_focusui_qwen25vl.py` | Main FocusUI model (Qwen2.5-VL) |
| `focusui/modeling_focusui_qwen3vl.py` | Same for Qwen3-VL |
| `focusui/dataset.py` | `LazySupervisedDataset`, `DataCollatorForSupervisedDataset` |
| `focusui/preprocess_focusui.py` | Generates `patch_scores_label` from bbox + UI graph |
| `focusui/constants.py` | Special tokens, system messages, chat templates |
| `train_focusui.py` | Full training entry point (complex, uses HfArgumentParser + DeepSpeed) |
| `train_patch_scorer.py` | **Simplified standalone scorer-only training script** (added in this session) |
| `data/data_config.yaml` | Full 6-dataset training config |
| `data/data_config_guiact_only.yaml` | GUIAct-only config for testing (added in this session) |

## Dataset Setup

Raw images must be downloaded separately from `cckevinn/GUI-Actor-Data` on HuggingFace. The filtered JSON annotation files are already in `datasets/FocusUI-Training-Data/`.

| Dataset | JSON samples | Image size | Image path |
|---|---|---|---|
| GUIAct | 28K | ~4 GB | `datasets/GUI-Actor-Data/GUIAct/web_imgs/` |
| GUIEnv | 136K | ~6 GB | `datasets/GUI-Actor-Data/guienvs/images/` |
| Wave-UI | 38K | ~24 GB | `datasets/GUI-Actor-Data/Wave-UI/images_fixed/` |
| AndroidControl | 26K | ~49 GB | `datasets/GUI-Actor-Data/AndroidControl/tfrecord/images/` |
| AMEX | 98K | ~92 GB | `datasets/GUI-Actor-Data/AMEX/screenshot/` |
| UGround | ~741K | ~214 GB | `datasets/GUI-Actor-Data/UGround-V1-Data-Box/images/` |

**Download script** (currently configured to download only GUIAct for testing):
```bash
bash download_gui_actor_data.sh
```
If extracted paths don't match expectations, run `bash fix_paths.sh` to diagnose and create symlinks.

**JSON file naming**:
- `*_filtered.json` → for Qwen2.5-VL (coordinate format: `pyautogui.click(x=0.35, y=0.06)`)
- `*_filtered_xy.json` → for Qwen3-VL (coordinate format: `(350, 60)` in 0–1000 range)

## Training

### Stage 1: Train PatchScorer only (simplified script)
```bash
python train_patch_scorer.py \
    --model_name_or_path huggingface/Qwen2.5-VL-3B-Instruct \
    --data_path data/data_config.yaml \
    --image_folder "" \
    --output_dir checkpoints/patch_scorer \
    --num_epochs 1 \
    --grad_accum_steps 32 \
    --save_steps 5000

# Test run with GUIAct only:
python train_patch_scorer.py \
    --model_name_or_path huggingface/Qwen2.5-VL-3B-Instruct \
    --data_path data/data_config_guiact_only.yaml \
    --image_folder "" \
    --output_dir checkpoints/patch_scorer_test \
    --grad_accum_steps 4 --save_steps 200 --logging_steps 5
```

Saves `patch_scorer_step{N}.pt` and `patch_scorer_final.pt` (just the scorer weights, not the full model).

### Stage 1: Train PatchScorer (full script, 8-GPU)
```bash
bash stage_1_ft_focusui_scorer.sh   # 8×GPU, DeepSpeed ZeRO-2
# or single-GPU version:
bash scripts/train/*.sh
```

Key flags in `train_focusui.py` for scorer-only training:
- `--train_patch_scorer_only True` → sets `patch_scorer_early_exit=True`, skips LM
- `--unfreeze_patch_scorer True` → only scorer params get gradients
- `--lm_loss_weight 0.0 --pointer_loss_weight 0.0 --ps_loss_weight 1.0`

### Resume scorer from checkpoint
```bash
python train_patch_scorer.py \
    --patch_scorer_ckpt checkpoints/patch_scorer/patch_scorer_step5000.pt \
    ...
# or in train_focusui.py:
    --train_patch_scorer_from_ckpt checkpoints/patch_scorer/patch_scorer_final.pt
```

## `patch_scores_label` Generation

`focusui/preprocess_focusui.py::preprocess_focusui_data()` generates labels per sample:
1. **Bbox score**: patch overlap with ground-truth bounding box → 1 if inside, 0 if outside
2. **UI-graph score**: visually unique patches get higher weight (Union-Find clustering on L2 distance between adjacent patches)
3. Combined: `gt_bbox_weight * bbox_score + gt_uigraph_weight * uigraph_score`, scaled to `[-1, 1]`, then spatially merged (÷4 patches per token due to `merge_size=2`)

Current default: `gt_bbox_weight=1.0`, `gt_uigraph_weight=0` (bbox only).

## Inference

```bash
python inference_focusui.py  # see file for args
```

To use PatchScorer during inference, the full FocusUI model checkpoint is needed (not just the scorer `.pt`). Load with:
```python
model = FocusUI_Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained("yyyang/FocusUI-3B")
```

## Evaluation

```bash
bash eval_qwen25vl_3b_screenspot_v2.sh   # ScreenSpot-V2
bash eval_qwen25vl_3b_screenspot_pro.sh  # ScreenSpot-Pro
bash eval_qwen25vl_3b_osworld_g.sh       # OSWorld-G
bash eval_qwen25vl_3b_ui_vision.sh       # UI-Vision
```

Eval scripts call into `evaluation/` directory. Benchmark JSONs are in `datasets/UI-Grounding-Benchmarks/`.

## Pretrained Checkpoints

```bash
bash downloading_datasets_checkpoints.sh
# Downloads to: checkpoints/FocusUI-3B, checkpoints/FocusUI-7B
```
