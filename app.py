import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np

# Import our existing registration logic
from utils import load_image, detect_and_compute, match_features, compute_homography_and_metrics, warp_image, plot_registration

app = Flask(__name__, static_folder='ui')
CORS(app)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://hhbtgggnovqazucjwtas.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_publishable_1w9RKM6TrnAuqlvgQJOeaQ_u36pSESc")

def upload_to_supabase(file_path, bucket_name, destination_path):
    """
    Helper function to upload files to Supabase Storage.
    To use this, you must install the 'supabase' python package and set your ENV credentials.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None # Supabase not configured
        
    try:
        from supabase import create_client, Client
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        with open(file_path, 'rb') as f:
            supabase.storage.from_(bucket_name).upload(file=f, path=destination_path, file_options={"content-type": "image/png"})
        
        return supabase.storage.from_(bucket_name).get_public_url(destination_path)
    except Exception as e:
        print(f"Supabase upload failed: {e}")
        return None

@app.route('/')
def serve_ui():
    return send_from_directory('ui', 'index.html')

@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route('/register', methods=['POST'])
def run_registration():
    if 'source' not in request.files or 'reference' not in request.files:
        return jsonify({'error': 'Missing source or reference image files'}), 400

    source_file = request.files['source']
    reference_file = request.files['reference']
    method = request.form.get('method', 'sift')

    # Generate unique IDs for this job
    job_id = str(uuid.uuid4())[:8]
    
    src_path = os.path.join(UPLOAD_FOLDER, f"src_{job_id}.png")
    ref_path = os.path.join(UPLOAD_FOLDER, f"ref_{job_id}.png")
    
    source_file.save(src_path)
    reference_file.save(ref_path)

    # --- Supabase Storage Hook (Background) ---
    import threading
    threading.Thread(target=upload_to_supabase, args=(src_path, "lunar-images", f"uploads/src_{job_id}.png")).start()
    threading.Thread(target=upload_to_supabase, args=(ref_path, "lunar-images", f"uploads/ref_{job_id}.png")).start()

    try:
        src_img = load_image(src_path)
        ref_img = load_image(ref_path)
        
        # 1. Feature Detection and Matching
        if method == 'vit':
            from utils import run_vit_matcher
            kp_src, kp_ref, matches = run_vit_matcher(src_img, ref_img)
        else:
            kp_src, des_src = detect_and_compute(src_img, method)
            kp_ref, des_ref = detect_and_compute(ref_img, method)
            matches = match_features(kp_src, des_src, kp_ref, des_ref, method)
        
        if len(matches) < 4:
            return jsonify({'error': 'Not enough matching features found.'}), 400
            
        H, mask, metrics = compute_homography_and_metrics(kp_src, kp_ref, matches, src_img=src_img, ref_img=ref_img)
        
        if H is None:
            return jsonify({'error': 'Failed to align images. RANSAC could not find a valid transformation.'}), 400
            
        if np.isnan(H).any() or np.isinf(H).any():
            return jsonify({'error': 'Degenerate transformation. RANSAC generated an invalid matrix.'}), 400
            
        # Fix numpy JSON serialization issues and key names
        metrics['inliers'] = int(metrics.pop('inlier_count', 0))
        metrics['inlier_ratio'] = float(metrics.get('inlier_ratio', 0.0))
        metrics['rmse'] = float(metrics.get('rmse', 0.0))
        
        # Extract inlier coordinates to send to the frontend for Canvas drawing
        matches_mask = mask.ravel().tolist()
        inlier_coords = []
        for i, m in enumerate(matches):
            if matches_mask[i] == 1:
                inlier_coords.append({
                    'src': [float(kp_src[m.queryIdx].pt[0]), float(kp_src[m.queryIdx].pt[1])],
                    'ref': [float(kp_ref[m.trainIdx].pt[0]), float(kp_ref[m.trainIdx].pt[1])]
                })

        # Cap at 50 matches so we don't overload the browser canvas
        inlier_coords = inlier_coords[:50]

        return jsonify({
            'success': True,
            'metrics': metrics,
            'H_matrix': H.tolist(),
            'matches': inlier_coords
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Lunar Registration Backend Server on http://localhost:5000")
    app.run(debug=True, port=5000)
