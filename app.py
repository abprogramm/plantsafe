import os
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import torch
import base64
import traceback

app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = 'plantModel.pt'
# Set device based on availability - Render typically provides CPU instances
# unless you configure GPU usage specifically (which costs more).
# Force CPU if you know you won't have CUDA on Render standard tiers.
# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEAF_CLASS_ID = 0
SPOT_CLASS_ID = 1

# --- Model Loading ---
model = None
try:
    print(f"Using device: {device}")
    print(f"Loading Ultralytics YOLO model from: {MODEL_PATH}")
    # Ensure the model path is correct within the deployed environment
    if not os.path.exists(MODEL_PATH):
         print(f"ERROR: Model file not found at path: {MODEL_PATH}")
         # Optionally raise an error or exit if model is critical
         # raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    else:
        model = YOLO(MODEL_PATH)
        print("YOLO Model loaded successfully.")

except Exception as e:
    print(f"FATAL: Error loading YOLO model: {e}")
    model = None # Ensure model is None so endpoint returns error

# --- Helper Function ---
def calculate_intersection_area(box_a, box_b):
    # (Function remains the same)
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    return inter_area

# --- API Endpoint ---
@app.route('/analyze', methods=['POST'])
def analyze():
    # Use loaded model variable directly
    if model is None:
        print("Error: Analyze endpoint called but model was not loaded.")
        return jsonify({'error': 'Model is not available on the server'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400

    try:
        image_bytes = file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image_pil)
        image_cv2_draw = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # --- Run YOLO Prediction ---
        # Let YOLO handle device placement if possible, or specify cpu/cuda
        results = model.predict(image_pil, device=device, verbose=False) # verbose=False for cleaner logs

        leaf_boxes = []
        spot_boxes = []
        leaf_results_for_json = []

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.tolist()
            classes = results[0].boxes.cls.tolist()
            print(f"Detected {len(boxes)} boxes.") # Logging

            for box, cls in zip(boxes, classes):
                int_box = [int(coord) for coord in box]
                if int(cls) == LEAF_CLASS_ID:
                    leaf_boxes.append(int_box)
                elif int(cls) == SPOT_CLASS_ID:
                    spot_boxes.append(int_box)
        else:
             print("No bounding boxes detected in results.")

        # --- Analyze Detected Leaves ---
        print(f"Found {len(leaf_boxes)} leaves and {len(spot_boxes)} spots.") # Logging
        for i, leaf in enumerate(leaf_boxes):
            lx1, ly1, lx2, ly2 = leaf
            leaf_area = (lx2 - lx1) * (ly2 - ly1)
            total_spot_intersection_area = 0

            if leaf_area <= 0:
                 print(f"Warning: Leaf {i+1} has zero or negative area: {leaf}. Skipping.")
                 leaf_results_for_json.append({'leaf_index': i + 1, 'percentage': 0.0})
                 cv2.rectangle(image_cv2_draw, (lx1, ly1), (lx2, ly2), (0, 255, 255), 1)
                 continue

            for spot in spot_boxes:
                intersection_area = calculate_intersection_area(leaf, spot)
                total_spot_intersection_area += intersection_area

            percentage = (total_spot_intersection_area / leaf_area) * 100.0
            percentage = round(max(0.0, min(100.0, percentage)), 2)

            leaf_results_for_json.append({'leaf_index': i + 1, 'percentage': percentage})

            # --- Annotate Image ---
            label = f"L{i+1}: {percentage:.1f}%"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image_cv2_draw, (lx1, ly1 - h - 7), (lx1 + w + 4, ly1), (0, 255, 0), -1)
            cv2.putText(image_cv2_draw, label, (lx1 + 2, ly1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        for spot in spot_boxes:
            sx1, sy1, sx2, sy2 = spot
            cv2.rectangle(image_cv2_draw, (sx1, sy1), (sx2, sy2), (0, 0, 255), 1)

        # --- Prepare Annotated Image ---
        image_rgb_annotated = cv2.cvtColor(image_cv2_draw, cv2.COLOR_BGR2RGB)
        annotated_image_pil = Image.fromarray(image_rgb_annotated)
        img_io = io.BytesIO()
        annotated_image_pil.save(img_io, 'JPEG', quality=85)
        img_io.seek(0)
        annotated_image_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

        # --- Prepare Final JSON Response ---
        response_data = {
            'annotated_image': annotated_image_base64,
            'leaf_results': leaf_results_for_json
        }
        print(f"Analysis complete. Returning results for {len(leaf_results_for_json)} leaves.") # Logging
        return jsonify(response_data)

    except Exception as e:
        print("An error occurred during analysis endpoint:")
        traceback.print_exc()
        return jsonify({'error': 'An internal error occurred while processing the image.'}), 500

# --- Run the App ---
if __name__ == '__main__':
    # Render/Gunicorn will handle host/port/debug settings via Procfile/environment vars
    # This block might not even run when deployed via Gunicorn, but is useful for local testing
    port = int(os.environ.get("PORT", 5000)) # Default to 5000 locally
    print(f"Attempting to run Flask development server on host 0.0.0.0, port {port}")
    # DO NOT use debug=True in production (on Render)
    app.run(host='0.0.0.0', port=port, debug=False)