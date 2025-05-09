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

MODEL_PATH = 'plantModel.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEAF_CLASS_ID = 0
SPOT_CLASS_ID = 1

model = None
try:
    print(f"Using device: {device}")
    print(f"Loading Ultralytics YOLO model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
         print(f"ERROR: Model file not found at path: {MODEL_PATH}")
    else:
        model = YOLO(MODEL_PATH)
        print("YOLO Model loaded successfully.")

except Exception as e:
    print(f"FATAL: Error loading YOLO model: {e}")
    traceback.print_exc()
    model = None

def calculate_intersection_area(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    return inter_area

@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None:
        print("Error: Analyze endpoint called but model was not loaded.")
        return jsonify({'error': 'Model is not available on the server'}), 500

    if 'image' not in request.files:
        print("Error: 'image' file part not found in the request.")
        return jsonify({'error': 'No image file part in the request'}), 400

    file = request.files['image']

    if file.filename == '':
        print("Error: No file selected for upload.")
        return jsonify({'error': 'No image file selected'}), 400

    try:
        image_bytes = file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image_pil)
        image_cv2_draw = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        print(f"Image received and converted. Shape: {image_cv2_draw.shape}")

        print("Running YOLO prediction...")
        results = model.predict(image_pil, device=device, verbose=False)
        print("YOLO prediction complete.")

        leaf_boxes = []
        spot_boxes = []
        leaf_results_for_json = []

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.tolist()
            classes = results[0].boxes.cls.tolist()
            print(f"Detected {len(boxes)} boxes.")

            for box, cls in zip(boxes, classes):
                int_box = [int(coord) for coord in box]
                if int(cls) == LEAF_CLASS_ID:
                    leaf_boxes.append(int_box)
                elif int(cls) == SPOT_CLASS_ID:
                    spot_boxes.append(int_box)
        else:
             print("No bounding boxes detected in results.")

        print(f"Analyzing {len(leaf_boxes)} detected leaves...")
        for i, leaf in enumerate(leaf_boxes):
            lx1, ly1, lx2, ly2 = leaf
            leaf_area = (lx2 - lx1) * (ly2 - ly1)
            total_spot_intersection_area = 0

            cv2.rectangle(image_cv2_draw, (lx1, ly1), (lx2, ly2), (0, 200, 0), 2)

            if leaf_area <= 0:
                 print(f"Warning: Leaf {i+1} bounding box has zero/negative area: {leaf}. Setting percentage to 0.")
                 leaf_results_for_json.append({
                    'leaf_index': i + 1,
                    'percentage': 0.0
                 })
                 continue

            for spot in spot_boxes:
                intersection_area = calculate_intersection_area(leaf, spot)
                total_spot_intersection_area += intersection_area

            percentage = (total_spot_intersection_area / leaf_area) * 100.0
            percentage = round(max(0.0, min(100.0, percentage)), 2)

            leaf_results_for_json.append({
                'leaf_index': i + 1,
                'percentage': percentage
            })

            label = f"L{i+1}: {percentage:.1f}%"
            font_scale = 0.6
            font_thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX

            (w, h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            cv2.rectangle(image_cv2_draw, (lx1, ly1 - h - 7), (lx1 + w + 4, ly1), (0, 200, 0), -1)
            cv2.putText(image_cv2_draw, label, (lx1 + 2, ly1 - 5), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        print(f"Drawing {len(spot_boxes)} spot boxes...")
        for spot in spot_boxes:
            sx1, sy1, sx2, sy2 = spot
            cv2.rectangle(image_cv2_draw, (sx1, sy1), (sx2, sy2), (0, 0, 200), 1)


        print("Encoding annotated image to Base64...")
        image_rgb_annotated = cv2.cvtColor(image_cv2_draw, cv2.COLOR_BGR2RGB)
        annotated_image_pil = Image.fromarray(image_rgb_annotated)

        img_io = io.BytesIO()
        annotated_image_pil.save(img_io, 'JPEG', quality=85)
        img_io.seek(0)

        annotated_image_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        print("Image encoding complete.")

        response_data = {
            'annotated_image': annotated_image_base64,
            'leaf_results': leaf_results_for_json
        }

        print(f"Analysis complete. Returning JSON results for {len(leaf_results_for_json)} leaves.")
        return jsonify(response_data)

    except Exception as e:
        print("An error occurred during the /analyze request:")
        traceback.print_exc()
        return jsonify({'error': 'An internal server error occurred while processing the image.'}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 6000))
    print(f"Flask development server starting on host 0.0.0.0, port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)