import cv2
import glob
import os
import shutil

def find_matches_in_datasets():
    print("Finding matching pairs between your extracted datasets...")
    
    # Let's say HAR tiles are our "Source" candidates
    source_candidates = glob.glob("extracted_tiles/*.png")
    
    # And the 32 "Save As" tiles are our "Reference" candidates
    reference_candidates = glob.glob("moon-base-equatorial*.png")
    
    if not source_candidates or not reference_candidates:
        print("Could not find both sets of tiles.")
        return
        
    # We will use ORB for speed since we are doing N x M comparisons
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Precompute features for references to save time
    ref_features = []
    for r_path in reference_candidates:
        img = cv2.imread(r_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            kp, des = orb.detectAndCompute(img, None)
            if des is not None:
                ref_features.append((r_path, img, kp, des))
                
    pair_count = 6 # We already have pairs 1-5
    
    for s_path in source_candidates:
        s_img = cv2.imread(s_path, cv2.IMREAD_GRAYSCALE)
        if s_img is None:
            continue
            
        kp_s, des_s = orb.detectAndCompute(s_img, None)
        if des_s is None:
            continue
            
        best_match_path = None
        best_inliers = 0
        
        # Compare this source against all references
        for r_path, r_img, kp_r, des_r in ref_features:
            matches = bf.match(des_s, des_r)
            
            # If we have enough matches to test Homography
            if len(matches) > 10:
                # Extract locations of good matches
                import numpy as np
                src_pts = np.float32([kp_s[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_r[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if mask is not None:
                    inliers = np.sum(mask)
                    if inliers > best_inliers:
                        best_inliers = inliers
                        best_match_path = r_path
                        
        # If the best match has a significant number of inliers, it's a true match!
        if best_match_path and best_inliers > 20: # Threshold of 20 robust inliers
            print(f"Found Match! {s_path} matches {best_match_path} ({best_inliers} inliers)")
            
            s_dst = f"source_images/s-{pair_count}.png"
            r_dst = f"reference_images/r-{pair_count}.png"
            
            shutil.copy(s_path, s_dst)
            shutil.copy(best_match_path, r_dst)
            pair_count += 1
            
    print(f"\nSuccessfully discovered {pair_count - 6} true pairs between the two datasets!")
    print(f"They have been organized in the source_images and reference_images folders.")

if __name__ == "__main__":
    find_matches_in_datasets()
