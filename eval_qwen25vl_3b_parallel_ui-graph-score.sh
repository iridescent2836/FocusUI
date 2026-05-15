
UI_GROUNDING_BENCH_BASE_DIR="datasets/UI-Grounding-Benchmarks"

MODEL_TYPE="focusui_3b"
# MODEL_PATH="checkpoints/focusui_3b_ft_scorer"
MODEL_PATH="checkpoints/FocusUI-3B"
SAVE_PATH="eval_results/focusui_3b/ScreenSpot-V2_e2e_latency"
EVAL_VISUAL_REDUCT_RATIOS="0.70"
SCORER_TYPE="l2-norm"

# EVAL_VISUAL_REDUCT_RATIOS="0.90"
# SCORER_TYPE="hist ncc"



echo "ScreenSpot-V2 | drop=${EVAL_VISUAL_REDUCT_RATIOS} | scorer=${SCORER_TYPE} | end to end latency | parallel computation"
python -m evaluation.ss_v2_eval \
    --save_path "${SAVE_PATH}/drop_${drop}_${SCORER_TYPE}_${MODEL_TYPE}_parallel" \
    --visual_reduct_ratio "${EVAL_VISUAL_REDUCT_RATIOS}" \
    --scorer_type "${SCORER_TYPE}" \
    --model_type "${MODEL_TYPE}" \
    --data_path "${UI_GROUNDING_BENCH_BASE_DIR}/ScreenSpot-V2" \
    --model_name_or_path "${MODEL_PATH}" \
    --using_random_samples

echo "ScreenSpot-V2 | drop=${EVAL_VISUAL_REDUCT_RATIOS} | scorer=${SCORER_TYPE} | end to end latency |  sequential computation"
python -m evaluation.ss_v2_eval \
    --save_path "${SAVE_PATH}/drop_${drop}_${SCORER_TYPE}_${MODEL_TYPE}_sequential" \
    --visual_reduct_ratio "${EVAL_VISUAL_REDUCT_RATIOS}" \
    --scorer_type "${SCORER_TYPE}" \
    --model_type "${MODEL_TYPE}" \
    --data_path "${UI_GROUNDING_BENCH_BASE_DIR}/ScreenSpot-V2" \
    --model_name_or_path "${MODEL_PATH}" \
    --using_sequential_computation


# 1272 images