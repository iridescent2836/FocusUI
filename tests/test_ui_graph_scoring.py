'''
Visualizing patch score from ui-graph only.
'''

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm as cm
from focusui.preprocess_focusui import PATCH_SIZE, build_patch_score_from_uigraph

def visualize_patch_scores(image: Image.Image, scores: np.ndarray, patch_size: int = PATCH_SIZE):
    """
    将 patch scores 可视化为热力图并覆盖在原图上。
    """

    width, height = image.size
    grid_h = height // patch_size
    grid_w = width // patch_size

    # 1. 将 1D 分数重新排列回 2D 网格 (针对单帧图像)
    # 注意：如果有 temporal 维度，这里取第一帧
    num_spatial_patches = grid_h * grid_w

    print(f"Image size: {image.size}, Scores shape: {scores.shape}, num_spatial_patches: {num_spatial_patches}")

    score_grid = scores[:num_spatial_patches].reshape(grid_h, grid_w)

    # 2. 归一化分数到 [0, 1] 空间，方便映射颜色
    score_min, score_max = score_grid.min(), score_grid.max()
    if score_max > score_min:
        score_norm = (score_grid - score_min) / (score_max - score_min)
    else:
        score_norm = score_grid

    # 3. 创建画布
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # 左侧：原始图像
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # 右侧：热力图叠加
    ax[1].imshow(image) # 先画底层原图

    # 使用 jet 或 viridis 等颜色映射，并进行插值使其平滑
    heatmap = ax[1].imshow(
        score_norm,
        cmap='jet',
        alpha=0.5,      # 透明度，0.5 方便看清底层细节
        extent=(0, width, height, 0), # 将小网格拉伸到图像尺寸
        interpolation='bilinear'      # 线性插值让热力图不那么“方块感”
    )

    ax[1].set_title("UI-Graph Patch Scores (Uniqueness)")
    ax[1].axis("off")

    # 添加颜色条
    plt.colorbar(heatmap, ax=ax[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

# 使用示例：
image = Image.open("./tmp/point_on_image.png")
scores = build_patch_score_from_uigraph(image)
visualize_patch_scores(image, scores)