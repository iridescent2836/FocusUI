'''
Visulizing the scoring result of focusui preprocessing.
'''

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
import matplotlib.cm as cm
from focusui.preprocess_focusui import *

def save_overlay_only(resized_img, final_merged_2d, smart_w, smart_h, save_path="overlay_only.png"):
    plt.figure(figsize=(10, 10))
    plt.imshow(resized_img)
    # 使用 jet 映射并将数值映射到图像上
    plt.imshow(final_merged_2d, cmap='jet', alpha=0.5,
               extent=(0, smart_w, smart_h, 0), interpolation='bilinear')
    plt.axis('off')

    # 去除白边并保存
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close() # 关闭画布防止内存占用
    print(f"Overlay 图片已保存至: {save_path}")

# 在 visualize_focusui_preprocessing 函数末尾调用它：


def visualize_focusui_preprocessing(ele_image: Image.Image, ele_bbox: tuple, results: dict):
    """
    可视化 preprocess_focusui_data 的完整输出结果（已修正尺寸匹配问题）
    """
    width, height = ele_image.size
    p_size = PATCH_SIZE  # 使用导入的全局变量
    m_size = MERGE_SIZE
    patch_merge_size = p_size * m_size

    # --- 关键修正：重新计算 smart_resize 后的尺寸 ---
    smart_h, smart_w = smart_resize_with_factor(
        height, width,
        factor=patch_merge_size,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    # 基于 resize 后的尺寸计算网格
    grid_h = smart_h // p_size
    grid_w = smart_w // p_size

    # 为了绘图，我们需要一个 resize 后的图像副本
    resized_img = ele_image.resize((smart_w, smart_h))
    # --------------------------------------------

    # 1. 准备图像 (在 resize 后的图上画 BBox)
    img_with_bbox = resized_img.copy()
    draw = ImageDraw.Draw(img_with_bbox)

    x1, y1, x2, y2 = ele_bbox
    # 如果是归一化坐标，使用 smart 尺寸转换
    if x1 <= 1 and x2 <= 1:
        x1, x2, y1, y2 = x1*smart_w, x2*smart_w, y1*smart_h, y2*smart_h
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # 2. 提取并还原 2D 矩阵
    # 基础 patch 级别 (grid_h x grid_w)
    bbox_2d = results["patch_score_bbox"].numpy().reshape(grid_h, grid_w)
    uigraph_2d = results["patch_score_uigraph"].numpy().reshape(grid_h, grid_w)
    combined_2d = results["patch_scores_label_unmerged"].numpy().reshape(grid_h, grid_w)

    # 最终合并级别 (final_h x final_w)
    final_h, final_w = grid_h // m_size, grid_w // m_size
    final_merged_2d = results["patch_scores_label"].numpy().reshape(final_h, final_w)

    # 3. 绘图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    axes[0, 0].imshow(img_with_bbox)
    axes[0, 0].set_title(f"1. Resized + BBox\n({smart_w}x{smart_h})")

    im1 = axes[0, 1].imshow(bbox_2d, cmap='jet')
    axes[0, 1].set_title("2. BBox Score")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(uigraph_2d, cmap='jet')
    axes[0, 2].set_title("3. UI-Graph Score")
    plt.colorbar(im2, ax=axes[0, 2])

    im3 = axes[1, 0].imshow(combined_2d, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 0].set_title("4. Combined Label (Unmerged)")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(final_merged_2d, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 1].set_title(f"5. Final Merged Label\n({final_w}x{final_h})")
    plt.colorbar(im4, ax=axes[1, 1])

    axes[1, 2].imshow(resized_img)
    axes[1, 2].imshow(final_merged_2d, cmap='jet', alpha=0.5,
                      extent=(0, smart_w, smart_h, 0),    interpolation='nearest' )
    axes[1, 2].set_title("6. Final Label Overlay")

    save_overlay_only(resized_img, final_merged_2d, smart_w, smart_h)


    for ax in axes.flatten():
        ax.axis('off')

    plt.show()

if __name__ == "__main__":

    # --- 运行示例 ---
    # 1. 模拟输入
    # test_path = "./datasets/Example-Data/images/1c6422e3-8eea-44db-9d70-67e74920ae02.png"
    test_path = "/home/iridescent/Code/FocusUI/datasets/UI-Grounding-Benchmarks/OSWorld-G/images/0lp8IshCDB.png"
    test_path_1 = "./tmp/point_on_image.png"
    test_img =  Image.open(test_path)
    test_bbox =  [0.098,0.762,0.269,0.829] # 假设中间有一个元素

    test_image_grid_thw = torch.tensor([ 1, 56, 96])  # 假设 smart resize 后是 384x384，patch size 是16，那么就是24x24的网格

    is_visulize = True
    is_using_bbox = False

    if is_visulize:
        for scorer_type in ["ssim"]:
            processed_data = preprocess_focusui_data(test_img, ui_graph_scorer_type=scorer_type)
            visualize_focusui_preprocessing(test_img, test_bbox, processed_data)


        # if is_using_bbox:
        #     processed_data = preprocess_focusui_data(test_img, test_bbox)
        # else:
        #     processed_data = preprocess_focusui_data(test_img)

        # visualize_focusui_preprocessing(test_img, test_bbox, processed_data)


    else:
        if is_using_bbox:
            processed_data = preprocess_focusui_data(test_img, test_bbox, image_grid_thw=test_image_grid_thw)
        else:
            processed_data = preprocess_focusui_data(test_img, image_grid_thw=test_image_grid_thw)

        patch_scores_label = processed_data.get("patch_scores_label", None)
        if patch_scores_label is not None:
            print(f"Patch scores label shape: {patch_scores_label.shape}")
            torch.save(patch_scores_label, "patch_scores_label.pt")
        else:
            print("No patch scores label found in the processed data.")