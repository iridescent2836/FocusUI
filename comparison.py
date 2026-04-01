import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from focusui import*
import torch

def compare_scorer_vs_gt(pred_scores: np.ndarray, gt_scores: np.ndarray, top_k_percent: float = 0.2):
    """
    比较 Scorer 预测值与 Ground Truth 的相似度

    Args:
        pred_scores: Scorer 输出的 1D array (已映射到 [-1, 1] 或 [0, 1])
        gt_scores: UF 计算出的 1D array
        top_k_percent: 选取前百分之多少的 patch 作为“核心关注区”进行 IoU 计算
    """
    # 确保 shape 一致
    assert pred_scores.shape == gt_scores.shape, "Shape mismatch!"

    # 1. MSE (数值接近程度)
    # 衡量每个 patch 上的绝对误差
    mse = mean_squared_error(gt_scores, pred_scores)

    # 2. Pearson Correlation (线性相关性)
    # 衡量两者整体趋势是否同步变大或变小
    pearson_corr, _ = pearsonr(pred_scores, gt_scores)

    # 3. Spearman Rank Correlation (排序相关性) - 最关键！
    # 因为 FocusUI 最终要按分数高低删 Token，所以“谁排第一”比“分数是 0.8 还是 0.9”重要得多
    spearman_corr, _ = spearmanr(pred_scores, gt_scores)

    # 4. Top-K IoU (核心区域重合度)
    # 模拟 Token Pruning：如果我要保留前 20% 的重要 patch，两者选中的区域重合度是多少？
    k = int(len(pred_scores) * top_k_percent)
    top_k_pred_idx = np.argsort(pred_scores)[-k:]
    top_k_gt_idx = np.argsort(gt_scores)[-k:]

    intersection = len(np.intersect1d(top_k_pred_idx, top_k_gt_idx))
    union = len(np.union1d(top_k_pred_idx, top_k_gt_idx))
    iou = intersection / union if union > 0 else 1.0

    return {
        "MSE": mse,
        "Pearson": pearson_corr,
        "Spearman": spearman_corr,
        "Top-K IoU": iou
    }

# --- 模拟测试逻辑 ---
# gt = build_patch_score_from_uigraph(...)
# pred = model_scorer_head(image_features)
gt = torch.load("patch_scores_label.pt").numpy()  # 从 preprocess_focusui_data 的输出中加载 GT
pred = torch.load("patch_score_pred.pt").numpy()  # 从模型输出中加载

metrics = compare_scorer_vs_gt(pred, gt)
print(metrics)