'''
Testing felzenszwalb scoring for ui component separation.
And the performance is not satisfying.
'''

import matplotlib.pyplot as plt
import numpy as np
from skimage import segmentation, color
from PIL import Image

# 1. 加载并准备图像 (延续你的路径)
image_path = "./datasets/Example-Data/images/1c6422e3-8eea-44db-9d70-67e74920ae02.png"
img_pil = Image.open(image_path).convert("RGB")
img = np.array(img_pil)

# 2. FH 算法分割
scale = 100
segments = segmentation.felzenszwalb(img, scale=scale, sigma=0.5, min_size=196)

# 3. 计算每个区域的得分
# 获取所有区域的 ID 和对应的像素数量
unique_labels, counts = np.unique(segments, return_counts=True)

# 评分逻辑：像素越少分越高。这里使用 1/counts 并归一化到 [0, 1]
# 如果你想让分数差距更明显，可以使用 np.log(1/counts)
scores = 1.0 / counts
scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

# 创建得分映射图 (Heatmap base)
score_map = np.zeros(segments.shape, dtype=np.float32)
for label, score in zip(unique_labels, scores_normalized):
    score_map[segments == label] = score

# 4. 可视化绘制
plt.figure(figsize=(12, 10))

# 首先画原图
plt.imshow(img)

# 然后叠加热力图 (使用 'jet' 或 'hot' 颜色映射)
# alpha 控制透明度，alpha=0.6 表示原图和热力图各占一部分
heatmap = plt.imshow(score_map, cmap='jet', alpha=0.5)

# 添加颜色条，显示分数高低
plt.colorbar(heatmap, label='Score (Higher = Fewer Pixels)')

plt.title(f"Segment Size Heatmap (Scale={scale})\nRed = Small Regions, Blue = Large Regions")
plt.axis('off')

# 保存结果
plt.savefig("segment_heatmap.png", dpi=300, bbox_inches='tight')
print("热力图已保存为: segment_heatmap.png")
plt.show()