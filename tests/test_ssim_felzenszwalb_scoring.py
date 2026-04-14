'''
Testing felzenszwalb scoring for ui component separation.
And the performance is not satisfying.
We considered combining the ssim algorithm as well, but this idea
is abandoned soon because ssim is pair-wise scoring.
'''

import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.segmentation import felzenszwalb


def compute_patch_ssim_graph(image, patch_size):
    """
    Convert image into patch-level representation (downsampled grid)
    """
    h, w = image.shape[:2]

    ph = h // patch_size
    pw = w // patch_size

    patches = []
    for i in range(ph):
        row = []
        for j in range(pw):
            patch = image[
                i * patch_size:(i + 1) * patch_size,
                j * patch_size:(j + 1) * patch_size
            ]
            row.append(patch)
        patches.append(row)

    return np.array(patches), ph, pw


def patch_similarity_map(patches):
    """
    Build similarity-based image for segmentation
    Each patch reduced to mean color (for Felzenszwalb input)
    """
    ph, pw = patches.shape[:2]

    feature_img = np.zeros((ph, pw, 6), dtype=np.float32)

    for i in range(ph):
        for j in range(pw):
            mean = patches[i, j].mean(axis=(0,1))
            std = patches[i, j].std(axis=(0,1))
            feature = np.concatenate([mean, std])
            feature_img[i, j] = feature
            # feature_img[i, j] = patches[i, j].mean(axis=(0, 1))

    return feature_img


def compute_ui_heatmap(image: Image.Image, patch_size=16, scale=100, sigma=0.5, min_size=20):
    """
    UI-aware heatmap using Felzenszwalb segmentation + patch scoring
    """

    image_np = np.array(image, dtype=np.float32) / 255.0

    if image_np.ndim == 2:
        image_np = np.stack([image_np]*3, axis=-1)

    # Step 1: patchify
    patches, ph, pw = compute_patch_ssim_graph(image_np, patch_size)

    # Step 2: build feature image (patch-level)
    feature_img = patch_similarity_map(patches)

    # Step 3: segmentation (Felzenszwalb)
    segments = felzenszwalb(feature_img, scale=scale, sigma=sigma, min_size=min_size)

    # Step 4: compute cluster sizes
    unique, counts = np.unique(segments, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))

    # Step 5: scoring (inverse size)
    scores = {k: 1.0 / v for k, v in cluster_sizes.items()}

    # normalize
    max_score = max(scores.values())
    scores = {k: v / max_score for k, v in scores.items()}

    # Step 6: build heatmap (patch-level → pixel-level)
    h, w = image_np.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    for i in range(ph):
        for j in range(pw):
            seg_id = segments[i, j]
            val = scores[seg_id]
            heatmap[
                i * patch_size:(i + 1) * patch_size,
                j * patch_size:(j + 1) * patch_size
            ] = val

    # Step 7: visualize
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted((image_np * 255).astype(np.uint8), 0.6, heatmap_color, 0.4, 0)

    return overlay, heatmap_color, segments


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--scale", type=float, default=500)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--min_size", type=int, default=20)
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")

    overlay, heatmap, segments = compute_ui_heatmap(
        img,
        patch_size=args.patch_size,
        scale=args.scale,
        sigma=args.sigma,
        min_size=args.min_size
    )

    cv2.imwrite("heatmap.png", heatmap)
    cv2.imwrite("overlay.png", overlay)

    print("Saved heatmap.png and overlay.png")
