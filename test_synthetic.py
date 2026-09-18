import cv2
import numpy as np
import os
import subprocess

def create_synthetic_images():
    # Create a reference image (e.g. 500x500 grayscale with some features)
    ref_img = np.zeros((500, 500), dtype=np.uint8)
    
    # Add some random shapes/corners to act as features
    for _ in range(50):
        x = np.random.randint(50, 450)
        y = np.random.randint(50, 450)
        w = np.random.randint(10, 50)
        h = np.random.randint(10, 50)
        cv2.rectangle(ref_img, (x, y), (x+w, y+h), 255, -1)
        
    # Add some noise
    noise = np.random.normal(0, 10, ref_img.shape).astype(np.uint8)
    ref_img = cv2.add(ref_img, noise)

    # Create a source image by translating, rotating, and scaling the reference
    M = cv2.getRotationMatrix2D((250, 250), 15, 0.8) # 15 degrees, 0.8 scale
    M[0, 2] += 20 # translation x
    M[1, 2] -= 30 # translation y
    
    src_img = cv2.warpAffine(ref_img, M, (500, 500))
    
    # Save them
    cv2.imwrite("ref_synthetic.png", ref_img)
    cv2.imwrite("src_synthetic.png", src_img)
    
    print("Created synthetic test images.")

if __name__ == "__main__":
    create_synthetic_images()
    print("Running registration pipeline...")
    
    # Call the register.py script
    subprocess.run([
        "python", "register.py",
        "--source", "src_synthetic.png",
        "--reference", "ref_synthetic.png",
        "--outdir", "test_output"
    ])
    
    print("Synthetic test completed.")
