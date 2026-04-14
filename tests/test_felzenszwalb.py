'''
Testing the performance of separation ui components using felzenszwalb,
which is unsatisfying.
'''

import matplotlib.pyplot as plt
import numpy as np
from skimage import data, segmentation, color
from skimage import graph
from PIL import Image
import numpy as np

# 1. 加载示例图像（也可以用 data.coffee() 或 data.astronaut()）
img = data.astronaut()

image_path = "../datasets/Example-Data/images/1c6422e3-8eea-44db-9d70-67e74920ae02.png"

# 加载图片


# 加载并转换为 RGB 格式
img_pil = Image.open(image_path).convert("RGB")

# 转换为 numpy 数组供 skimage 使用
img = np.array(img_pil)

# 注意：如果 png 图片带透明通道（RGBA），FH 算法可能会报错
# 建议只取前 3 个通道 (RGB)
if img.shape[-1] == 4:
    img = img[:, :, :3]

# 2. 使用 FH 算法进行分割
# scale: 对应算法中的 k，越大块越少
# sigma: 高斯模糊标准差，用于预处理去噪
# min_size: 最小区域大小（像素数），小于此值的区域会被合并
smaller_scale = 50
larger_scale = 100
segments_small_k = segmentation.felzenszwalb(img, scale=smaller_scale, sigma=0.5, min_size=196)
segments_large_k = segmentation.felzenszwalb(img, scale=larger_scale, sigma=0.5, min_size=196)

# 3. 将分割结果转化为边界图
out_small = segmentation.mark_boundaries(img, segments_small_k)
out_large = segmentation.mark_boundaries(img, segments_large_k)

# 4. 可视化对比
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(img)
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(out_small)
ax[1].set_title(f"FH Segmentation (scale={smaller_scale})\nRegions: {len(np.unique(segments_small_k))}")
ax[1].axis('off')

ax[2].imshow(out_large)
ax[2].set_title(f"FH Segmentation (scale={larger_scale})\nRegions: {len(np.unique(segments_large_k))}")
ax[2].axis('off')

plt.tight_layout()
# 5. 保存为图片
save_path = "segmentation_result.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"结果已保存至: {save_path}")

# 如果不需要显示，可以关闭 plot 释放内存
plt.close()