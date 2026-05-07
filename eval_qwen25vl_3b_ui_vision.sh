
UI_GROUNDING_BENCH_BASE_DIR="datasets/UI-Grounding-Benchmarks"

MODEL_TYPE="focusui_3b"
MODEL_PATH="checkpoints/focusui_3b_ft_scorer"
SAVE_PATH="eval_results/focusui_3b/ui_vision_new_scorer"
EVAL_VISUAL_REDUCT_RATIOS="0.50 0.70"
SCORER_TYPE="l2-norm ssim hist"

# EVAL_VISUAL_REDUCT_RATIOS="0.90"
# SCORER_TYPE="hist ncc"

for drop in $EVAL_VISUAL_REDUCT_RATIOS; do
    for scorer in $SCORER_TYPE; do
        echo "UI-Vision | drop=${drop} | scorer=${scorer}"
        python -m evaluation.ui_vision_eval \
            --save_path "${SAVE_PATH}/drop_${drop}_${scorer}_${MODEL_TYPE}" \
            --visual_reduct_ratio "${drop}" \
            --scorer_type "${scorer}" \
            --model_type "${MODEL_TYPE}" \
            --data_path "${UI_GROUNDING_BENCH_BASE_DIR}/UI-Vision" \
            --model_name_or_path "${MODEL_PATH}" \
            --using_combined_scorer
    done
done

for drop in $EVAL_VISUAL_REDUCT_RATIOS; do
    echo "UI-Vision | drop=${drop} | scorer=scorer | new scorer only"
    python -m evaluation.ui_vision_eval \
        --save_path "${SAVE_PATH}/drop_${drop}_scorer_${MODEL_TYPE}" \
        --visual_reduct_ratio "${drop}" \
        --scorer_type "scorer" \
        --model_type "${MODEL_TYPE}" \
        --data_path "${UI_GROUNDING_BENCH_BASE_DIR}/UI-Vision" \
        --model_name_or_path "${MODEL_PATH}"
done

echo "UI-Vision | no drop | no visual token select"
python -m evaluation.ui_vision_eval \
    --save_path "${SAVE_PATH}/no_drop_${MODEL_TYPE}" \
    --model_type "${MODEL_TYPE}" \
    --no-apply_visual_token_select \
    --data_path "${UI_GROUNDING_BENCH_BASE_DIR}/UI-Vision" \
    --model_name_or_path "${MODEL_PATH}"

# 5479 images