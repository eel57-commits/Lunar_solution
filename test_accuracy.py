import cv2
import numpy as np
import time
from utils import run_vit_matcher, compute_homography_and_metrics, run_tiled_vit_matcher

# Load test images
src_path = "moon-base-equatorial (0).png"
ref_path = "moon-base-equatorial (1).png"

src_img = cv2.imread(src_path)
ref_img = cv2.imread(ref_path)

if src_img is None or ref_img is None:
    print("Could not load test images.")
    exit(1)

print("Running Standard LoFTR...")
start = time.time()
kp_src, kp_ref, matches = run_vit_matcher(src_img, ref_img)
H, mask, metrics = compute_homography_and_metrics(kp_src, kp_ref, matches, src_img, ref_img)
end = time.time()

print("=== Standard LoFTR Results ===")
print(f"Time: {end - start:.2f}s")
if metrics:
    print(f"Inliers: {metrics['inlier_count']} / {len(matches)}")
    print(f"Inlier Ratio: {metrics['inlier_ratio'] * 100:.2f}%")
    print(f"RMSE: {metrics['rmse']:.3f} px")
else:
    print("Registration failed.")

print("\nRunning High-Resolution Tiled LoFTR...")
start = time.time()
kp_src_tiled, kp_ref_tiled, matches_tiled = run_tiled_vit_matcher(src_img, ref_img)
H_tiled, mask_tiled, metrics_tiled = compute_homography_and_metrics(kp_src_tiled, kp_ref_tiled, matches_tiled, src_img, ref_img)
end = time.time()

print("=== Tiled LoFTR Results ===")
print(f"Time: {end - start:.2f}s")
if metrics_tiled:
    print(f"Inliers: {metrics_tiled['inlier_count']} / {len(matches_tiled)}")
    print(f"Inlier Ratio: {metrics_tiled['inlier_ratio'] * 100:.2f}%")
    print(f"RMSE: {metrics_tiled['rmse']:.3f} px")
else:
    print("Registration failed.")
