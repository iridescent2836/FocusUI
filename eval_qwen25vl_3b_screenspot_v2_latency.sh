
UI_GROUNDING_BENCH_BASE_DIR="datasets/UI-Grounding-Benchmarks"

MODEL_TYPE="focusui_3b"
# MODEL_PATH="checkpoints/focusui_3b_ft_scorer"
MODEL_PATH="checkpoints/FocusUI-3B"

SAVE_PATH="eval_results/focusui_3b/ScreenSpot-V2_new_scorer_latency"
EVAL_VISUAL_REDUCT_RATIOS="0.30 0.50 0.70"
SCORER_TYPE="l2-norm"

# EVAL_VISUAL_REDUCT_RATIOS="0.90"
# SCORER_TYPE="hist ncc"

for drop in $EVAL_VISUAL_REDUCT_RATIOS; do
    for scorer in $SCORER_TYPE; do
        echo "ScreenSpot-V2 | drop=${drop} | scorer=${scorer} | combined scorer"
        python -m evaluation.ss_v2_eval \
            --save_path "${SAVE_PATH}/drop_${drop}_${scorer}_${MODEL_TYPE}" \
            --visual_reduct_ratio "${drop}" \
            --scorer_type "${scorer}" \
            --model_type "${MODEL_TYPE}" \
            --data_path "${UI_GROUNDING_BENCH_BASE_DIR}/ScreenSpot-V2" \
            --model_name_or_path "${MODEL_PATH}" \
            --using_combined_scorer \
            --using_random_samples \
            --num_samples 3
    done
done

echo "ScreenSpot-V2 | no drop | no visual token select"
python -m evaluation.ss_v2_eval \
    --save_path "${SAVE_PATH}/no_drop_${MODEL_TYPE}" \
    --model_type "${MODEL_TYPE}" \
    --no-apply_visual_token_select \
    --data_path "${UI_GROUNDING_BENCH_BASE_DIR}/ScreenSpot-V2" \
    --model_name_or_path "${MODEL_PATH}" \
    --using_random_samples \
    --num_samples 3

# 1272 images