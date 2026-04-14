
UI_GROUNDING_BENCH_BASE_DIR="datasets/UI-Grounding-Benchmarks"

MODEL_TYPE="focusui_3b"
MODEL_PATH="checkpoints/focusui_3b"
SAVE_PATH="eval_results/focusui_3b/screenspot_pro"
# ehen eval_visual_reduct_ratio = 0, bug happens.
EVAL_VISUAL_REDUCT_RATIOS="0.50 0.70"
SCORER_TYPE="scorer l2-norm ssim"

for drop in $EVAL_VISUAL_REDUCT_RATIOS; do
    for scorer in $SCORER_TYPE; do
        echo "ScreenSpot-Pro | drop=${drop} | scorer=${scorer}"
        python -m evaluation.ss_pro_eval \
            --save_path "${SAVE_PATH}/drop_${drop}_${scorer}" \
            --visual_reduct_ratio "${drop}" \
            --scorer_type "${scorer}"

    done
done
    # echo "ScreenSpot-Pro | drop=${drop} | device=cuda:0"
    # python -m evaluation.ss_pro_eval \
    #     --model_type "${MODEL_TYPE}" \
    #     --model_name_or_path "${MODEL_PATH}" \
    #     --save_path "${SAVE_PATH}/drop_${drop}" \
    #     --data_path "${UI_GROUNDING_BENCH_BASE_DIR}/ScreenSpot-Pro" \
    #     --topk 3 \
    #     --device "cuda:0" \
    #     --visual_reduct_ratio "${drop}"
# done


