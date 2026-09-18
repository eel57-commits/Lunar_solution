import cv2
from utils import load_image, detect_and_compute, match_features, compute_homography_and_metrics, run_vit_matcher

def test_pair(src_path, ref_path):
    print(f"Testing {src_path} vs {ref_path}")
    src_img = load_image(src_path)
    ref_img = load_image(ref_path)
    
    # 1. Test SIFT
    print("\n--- Running SIFT ---")
    kp_src_sift, des_src_sift = detect_and_compute(src_img, 'sift')
    kp_ref_sift, des_ref_sift = detect_and_compute(ref_img, 'sift')
    matches_sift = match_features(kp_src_sift, des_src_sift, kp_ref_sift, des_ref_sift, 'sift')
    
    H_sift, mask_sift, metrics_sift = compute_homography_and_metrics(kp_src_sift, kp_ref_sift, matches_sift)
    
    if H_sift is not None:
        print(f"SIFT Success! Inliers: {metrics_sift['inlier_count']}, Inlier Ratio: {metrics_sift['inlier_ratio']:.2f}, RMSE: {metrics_sift['rmse']:.2f}")
    else:
        print("SIFT FAILED to find a valid transformation.")
        
    # 2. Test ViT (LoFTR)
    print("\n--- Running Vision Transformers (LoFTR) ---")
    kp_src_vit, kp_ref_vit, matches_vit = run_vit_matcher(src_img, ref_img)
    H_vit, mask_vit, metrics_vit = compute_homography_and_metrics(kp_src_vit, kp_ref_vit, matches_vit)
    
    if H_vit is not None:
        print(f"ViT Success! Inliers: {metrics_vit['inlier_count']}, Inlier Ratio: {metrics_vit['inlier_ratio']:.2f}, RMSE: {metrics_vit['rmse']:.2f}")
    else:
        print("ViT FAILED to find a valid transformation.")

if __name__ == "__main__":
    test_pair("distorted_source/ds-10.png", "reference_images/r-10.png")
