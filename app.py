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

# --- IMPORTANT ---
# You MUST replace this with a YOLO segmentation model.
# A detection model will not work with this code.
MODEL_PATH = 'plantModel.pt'
LEAF_CLASS_ID = 0
SPOT_CLASS_ID = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

try:
    print(f"Using device: {device}")
    print(f"Loading Ultralytics YOLO Segmentation model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at path: {MODEL_PATH}")
    else:
        # This part is the same
        model = YOLO(MODEL_PATH)
        print("YOLO Segmentation Model loaded successfully.")
except Exception as e:
    print(f"FATAL: Error loading YOLO model: {e}")
    traceback.print_exc()
    model = None

# --- NEW HELPER FUNCTION ---
def create_binary_mask(segments, class_id, img_shape):
    """Creates a single binary mask for all instances of a specific class."""
    class_mask = np.zeros(img_shape, dtype=np.uint8)
    if segments:
        for seg in segments[class_id]:
            # Reshape for fillPoly which expects a list of contours
            contour = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(class_mask, [contour], 255) # 255 is our "on" pixel
    return class_mask


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['image']
        image_bytes = file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image_pil)

        print("Running YOLO segmentation prediction...")
        results = model.predict(image_pil, device=device, verbose=False, retina_masks=True)
        print("Prediction complete.")

        # Extract segmentation masks
        masks = results[0].masks.data.cpu().numpy()  # shape: (num_instances, H, W)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)  # shape: (num_instances,)

        # Initialize binary masks
        leaf_mask_total = np.zeros_like(masks[0], dtype=np.uint8)
        spot_mask_total = np.zeros_like(masks[0], dtype=np.uint8)

        for mask, cls_id in zip(masks, classes):
            if cls_id == LEAF_CLASS_ID:
                leaf_mask_total = np.logical_or(leaf_mask_total, mask).astype(np.uint8)
            elif cls_id == SPOT_CLASS_ID:
                spot_mask_total = np.logical_or(spot_mask_total, mask).astype(np.uint8)

        # Calculate pixel counts
        total_leaf_pixels = np.sum(leaf_mask_total)
        total_spot_pixels = np.sum(spot_mask_total)

        print(f"Leaf pixels: {total_leaf_pixels}, Spot pixels: {total_spot_pixels}")

        if total_leaf_pixels == 0:
            percentage = 0.0
        else:
            percentage = round((total_spot_pixels / total_leaf_pixels) * 100.0, 2)

        # Create overlay image
        overlay = image_np.copy()
        overlay[leaf_mask_total == 1] = [0, 255, 0]   # Green for leaves
        overlay[spot_mask_total == 1] = [255, 0, 0]   # Red for spots

        # Encode overlay as base64
        annotated_image = Image.fromarray(overlay)
        buffered = io.BytesIO()
        annotated_image.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'annotated_image': encoded_image,
            'leaf_results': [{'leaf_index': 1, 'percentage': percentage}]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'Processing error'}), 500


if __name__ == '__main__':
    # Use the PORT environment variable for services like Hugging Face Spaces
    port = int(os.environ.get("PORT", 3000)) 
    print(f"Flask server starting on host 0.0.0.0, port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)