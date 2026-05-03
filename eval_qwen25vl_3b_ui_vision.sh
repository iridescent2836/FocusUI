
UI_GROUNDING_BENCH_BASE_DIR="datasets/UI-Grounding-Benchmarks"

MODEL_TYPE="focusui_3b"
MODEL_PATH="checkpoints/focusui_3b"
SAVE_PATH="eval_results/focusui_3b/ui_vusion"
EVAL_VISUAL_REDUCT_RATIOS="0.30 0.50 0.70 0.90"
SCORER_TYPE="scorer l2-norm ssim hist ncc random"

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
            --data_path "${UI_GROUNDING_BENCH_BASE_DIR}/UI-Vision"
    done
done


echo "UI-Vision | no drop | no visual token select"
python -m evaluation.ui_vision_eval \
    --save_path "${SAVE_PATH}/no_drop_${MODEL_TYPE}" \
    --model_type "${MODEL_TYPE}" \
    --no-apply_visual_token_select \
    --data_path "${UI_GROUNDING_BENCH_BASE_DIR}/UI-Vision"