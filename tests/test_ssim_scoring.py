'''
Visulizing ssim scoring in images. The result is great.
'''

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.size[rx] < self.size[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]



def compute_heatmap(image, patch_size=16, ssim_thresh=0.9):
    h, w = image.shape[:2]

    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Number of patches
    ph = h // patch_size
    pw = w // patch_size
    num_patches = ph * pw

    uf = UnionFind(num_patches)

    def patch_idx(i, j):
        return i * pw + j

    # Compare adjacent patches
    for i in range(ph):
        for j in range(pw):
            patch1 = gray[i*patch_size:(i+1)*patch_size,
                          j*patch_size:(j+1)*patch_size]

            # Right neighbor
            if j + 1 < pw:
                patch2 = gray[i*patch_size:(i+1)*patch_size,
                              (j+1)*patch_size:(j+2)*patch_size]
                score = ssim(patch1, patch2)
                if score > ssim_thresh:
                    uf.union(patch_idx(i, j), patch_idx(i, j+1))

            # Down neighbor
            if i + 1 < ph:
                patch2 = gray[(i+1)*patch_size:(i+2)*patch_size,
                              j*patch_size:(j+1)*patch_size]
                score = ssim(patch1, patch2)
                if score > ssim_thresh:
                    uf.union(patch_idx(i, j), patch_idx(i+1, j))

    # Compute component sizes
    comp_size = {}
    for idx in range(num_patches):
        root = uf.find(idx)
        comp_size[root] = comp_size.get(root, 0) + 1

    # Score: smaller component => higher score
    scores = {}
    for root, size in comp_size.items():
        scores[root] = 1.0 / size

    # Normalize scores
    max_score = max(scores.values())
    for k in scores:
        scores[k] /= max_score

    # Build heatmap
    heatmap = np.zeros((h, w), dtype=np.float32)

    for i in range(ph):
        for j in range(pw):
            idx = patch_idx(i, j)
            root = uf.find(idx)
            val = scores[root]
            heatmap[i*patch_size:(i+1)*patch_size,
                    j*patch_size:(j+1)*patch_size] = val

    # Resize to original (in case of cropping)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)

    return overlay, heatmap_color


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--ssim_thresh", type=float, default=0.9)
    args = parser.parse_args()

    img = cv2.imread(args.image)

    overlay, heatmap = compute_heatmap(
        img,
        patch_size=args.patch_size,
        ssim_thresh=args.ssim_thresh
    )

    cv2.imwrite("heatmap.png", heatmap)
    cv2.imwrite("overlay.png", overlay)

    print("Saved heatmap.png and overlay.png")
