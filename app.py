import os
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify # Keep jsonify, remove send_file for now
from ultralytics import YOLO
from PIL import Image
import torch
import base64 # Import base64 for encoding the image
import traceback # Keep for better error logging

app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = 'plantModel.pt' # Your YOLO model path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Verify these IDs match how your YOLO model was trained
LEAF_CLASS_ID = 0 # Assuming class 0 is 'leaf'
SPOT_CLASS_ID = 1 # Assuming class 1 is 'spot'

# --- Model Loading ---
model = None
try:
    print(f"Using device: {device}")
    print(f"Loading Ultralytics YOLO model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    # No need to move YOLOv8 models explicitly with .to(device) usually,
    # it handles device placement during predict. Fuse is also optional here.
    # model.fuse() # Optional: Can speed up inference slightly but might affect some metrics
    print("YOLO Model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

# --- Helper Function (Unchanged) ---
def calculate_intersection_area(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    return inter_area

# --- API Endpoint ---
@app.route('/analyze', methods=['POST'])
def analyze():
    global model
    if model is None:
        return jsonify({'error': 'Model is not loaded or failed to load'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400

    try:
        image_bytes = file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image_pil)
        # Convert RGB (PIL) to BGR (OpenCV) for drawing
        image_cv2_draw = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # --- Run YOLO Prediction ---
        # Pass PIL image directly, specify device if needed (though often automatic)
        results = model.predict(image_pil, device=device, verbose=False)

        leaf_boxes = []
        spot_boxes = []
        leaf_results_for_json = [] # Store results for JSON output

        # Check if results[0].boxes exists and is not None
        if results and results[0].boxes is not None:
            # Get boxes (xyxy format) and classes as lists
            boxes = results[0].boxes.xyxy.tolist()
            classes = results[0].boxes.cls.tolist()

            for box, cls in zip(boxes, classes):
                # Ensure box coordinates are integers for drawing and area calculation
                int_box = [int(coord) for coord in box]
                if int(cls) == LEAF_CLASS_ID: # Compare class ID as int
                    leaf_boxes.append(int_box)
                elif int(cls) == SPOT_CLASS_ID: # Compare class ID as int
                    spot_boxes.append(int_box)
        else:
             print("No boxes detected in results.")


        # --- Analyze Detected Leaves ---
        for i, leaf in enumerate(leaf_boxes):
            lx1, ly1, lx2, ly2 = leaf
            # Calculate leaf bounding box area
            leaf_area = (lx2 - lx1) * (ly2 - ly1)
            total_spot_intersection_area = 0

            # Skip calculation if leaf area is non-positive
            if leaf_area <= 0:
                 print(f"Warning: Leaf {i+1} has zero or negative area. Skipping.")
                 # Append a result with 0 percentage, but maybe log this
                 leaf_results_for_json.append({
                    'leaf_index': i + 1, # Use 1-based index for display
                    'percentage': 0.0
                 })
                 # Optionally draw the invalid box differently or not at all
                 cv2.rectangle(image_cv2_draw, (lx1, ly1), (lx2, ly2), (0, 255, 255), 1) # Yellow border for zero area?
                 continue # Move to the next leaf

            # Calculate intersection with detected spots
            for spot in spot_boxes:
                intersection_area = calculate_intersection_area(leaf, spot)
                total_spot_intersection_area += intersection_area

            # Calculate percentage, ensuring leaf_area > 0
            percentage = (total_spot_intersection_area / leaf_area) * 100.0
            # Clamp percentage between 0 and 100
            percentage = round(max(0.0, min(100.0, percentage)), 2) # Round to 2 decimal places

            # Store result for JSON output
            leaf_results_for_json.append({
                'leaf_index': i + 1, # Use 1-based index
                'percentage': percentage
            })

            # --- Annotate Image (OpenCV - BGR format) ---
            # Draw leaf bounding box (Green)
            cv2.rectangle(image_cv2_draw, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
            # Prepare label text
            label = f"L{i+1}: {percentage:.1f}%" # Shorter label: L1: 5.7%
            # Calculate text size to draw background rectangle
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            # Draw background rectangle for text
            cv2.rectangle(image_cv2_draw, (lx1, ly1 - h - 7), (lx1 + w + 4, ly1), (0, 255, 0), -1)
            # Put label text (Black on Green background)
            cv2.putText(image_cv2_draw, label, (lx1 + 2, ly1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw spot bounding boxes (Red) after leaves so they are on top if overlapping
        for spot in spot_boxes:
            sx1, sy1, sx2, sy2 = spot
            cv2.rectangle(image_cv2_draw, (sx1, sy1), (sx2, sy2), (0, 0, 255), 1) # Thinner red border for spots


        # --- Prepare Annotated Image for Sending ---
        # Convert annotated OpenCV image (BGR) back to RGB for PIL
        image_rgb_annotated = cv2.cvtColor(image_cv2_draw, cv2.COLOR_BGR2RGB)
        annotated_image_pil = Image.fromarray(image_rgb_annotated)

        # Save annotated image to a BytesIO buffer as JPEG
        img_io = io.BytesIO()
        annotated_image_pil.save(img_io, 'JPEG', quality=85) # Use quality=85-90 generally
        img_io.seek(0)

        # Encode image bytes as Base64 string
        annotated_image_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

        # --- Prepare Final JSON Response ---
        response_data = {
            'annotated_image': annotated_image_base64,
            'leaf_results': leaf_results_for_json
        }

        # Return JSON
        return jsonify(response_data)

    except Exception as e:
        print("An error occurred during analysis:")
        traceback.print_exc() # Print detailed traceback to console
        # Return a generic error message to the client
        return jsonify({'error': 'An internal error occurred while processing the image.'}), 500

# --- Run the App ---
if __name__ == '__main__':
    # Make sure port matches what Android app connects to (e.g., 5000 or 8080)
    # Set debug=False for production/deployment
    app.run(host='0.0.0.0', port=5000, debug=True) # Changed port back to 5000 for consistency