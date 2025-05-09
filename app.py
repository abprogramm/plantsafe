import os
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify # Keep jsonify
from ultralytics import YOLO
from PIL import Image
import torch
import base64 # Import base64 for encoding the image
import traceback # Keep for better error logging

app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = 'plantModel.pt' # Your YOLO model path
# Set device based on availability - Render typically provides CPU instances
# unless you configure GPU usage specifically (which costs more).
# Force CPU if you know you won't have CUDA on Render standard tiers.
# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEAF_CLASS_ID = 0 # Assuming class 0 is 'leaf' in your trained model
SPOT_CLASS_ID = 1 # Assuming class 1 is 'spot' in your trained model

# --- Model Loading ---
model = None
try:
    print(f"Using device: {device}")
    print(f"Loading Ultralytics YOLO model from: {MODEL_PATH}")
    # Ensure the model path is correct within the deployed environment
    if not os.path.exists(MODEL_PATH):
         print(f"ERROR: Model file not found at path: {MODEL_PATH}")
         # Consider raising an error if the model is essential for the app to run
         # raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    else:
        model = YOLO(MODEL_PATH)
        print("YOLO Model loaded successfully.")

except Exception as e:
    print(f"FATAL: Error loading YOLO model: {e}")
    traceback.print_exc() # Print full traceback for loading errors
    model = None # Ensure model is None so endpoint returns error

# --- Helper Function ---
def calculate_intersection_area(box_a, box_b):
    """Calculates the intersection area of two bounding boxes."""
    # Determine the coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # Compute the area of intersection rectangle
    # Ensure dimensions are positive (if no overlap, width/height will be <= 0)
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    return inter_area

# --- API Endpoint ---
@app.route('/analyze', methods=['POST'])
def analyze():
    # Use loaded model variable directly
    if model is None:
        print("Error: Analyze endpoint called but model was not loaded.")
        return jsonify({'error': 'Model is not available on the server'}), 500

    # Check if the 'image' file part is in the request
    if 'image' not in request.files:
        print("Error: 'image' file part not found in the request.")
        return jsonify({'error': 'No image file part in the request'}), 400

    file = request.files['image']

    # Check if a file was selected
    if file.filename == '':
        print("Error: No file selected for upload.")
        return jsonify({'error': 'No image file selected'}), 400

    try:
        # Read image bytes from the uploaded file
        image_bytes = file.read()
        # Open image using PIL and ensure it's RGB
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        # Convert PIL Image to NumPy array for OpenCV processing
        image_np = np.array(image_pil)
        # Convert RGB (PIL) to BGR (OpenCV standard) for drawing functions
        image_cv2_draw = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        print(f"Image received and converted. Shape: {image_cv2_draw.shape}")

        # --- Run YOLO Prediction ---
        print("Running YOLO prediction...")
        # Pass the PIL image to the model
        results = model.predict(image_pil, device=device, verbose=False)
        print("YOLO prediction complete.")

        leaf_boxes = []
        spot_boxes = []
        leaf_results_for_json = [] # List to store results for JSON output

        # Process detection results
        # Ensure results object and boxes attribute are valid
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            # Extract bounding boxes (xyxy format) and class IDs
            boxes = results[0].boxes.xyxy.tolist()
            classes = results[0].boxes.cls.tolist()
            print(f"Detected {len(boxes)} boxes.")

            # Separate leaf and spot boxes
            for box, cls in zip(boxes, classes):
                # Convert coordinates to integers
                int_box = [int(coord) for coord in box]
                if int(cls) == LEAF_CLASS_ID:
                    leaf_boxes.append(int_box)
                elif int(cls) == SPOT_CLASS_ID:
                    spot_boxes.append(int_box)
        else:
             print("No bounding boxes detected in results.")

        # --- Analyze Detected Leaves and Annotate Image ---
        print(f"Analyzing {len(leaf_boxes)} detected leaves...")
        for i, leaf in enumerate(leaf_boxes):
            lx1, ly1, lx2, ly2 = leaf
            # Calculate leaf bounding box area
            leaf_area = (lx2 - lx1) * (ly2 - ly1)
            total_spot_intersection_area = 0

            # Draw leaf bounding box (Green, thickness 2)
            cv2.rectangle(image_cv2_draw, (lx1, ly1), (lx2, ly2), (0, 200, 0), 2) # Slightly darker green

            # Skip percentage calculation if leaf area is non-positive
            if leaf_area <= 0:
                 print(f"Warning: Leaf {i+1} bounding box has zero/negative area: {leaf}. Setting percentage to 0.")
                 leaf_results_for_json.append({
                    'leaf_index': i + 1,
                    'percentage': 0.0
                 })
                 # No label drawn for zero-area leaves
                 continue # Move to the next leaf

            # Calculate total intersection area with all detected spots
            for spot in spot_boxes:
                intersection_area = calculate_intersection_area(leaf, spot)
                total_spot_intersection_area += intersection_area

            # Calculate percentage of leaf area covered by spots
            percentage = (total_spot_intersection_area / leaf_area) * 100.0
            # Clamp percentage between 0 and 100 and round
            percentage = round(max(0.0, min(100.0, percentage)), 2)

            # Store result for the JSON response
            leaf_results_for_json.append({
                'leaf_index': i + 1,
                'percentage': percentage
            })

            # --- Annotate Leaf with Label and Percentage ---
            label = f"L{i+1}: {percentage:.1f}%" # e.g., L1: 15.7%
            font_scale = 0.6
            font_thickness = 1 # Use 1 for thinner text, 2 for bolder
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Calculate text size to draw background rectangle
            (w, h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            # Draw background rectangle slightly above the leaf box corner
            cv2.rectangle(image_cv2_draw, (lx1, ly1 - h - 7), (lx1 + w + 4, ly1), (0, 200, 0), -1) # Green background
            # Put label text (Black text)
            cv2.putText(image_cv2_draw, label, (lx1 + 2, ly1 - 5), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        # Draw spot bounding boxes (Red, thickness 1) after leaves
        print(f"Drawing {len(spot_boxes)} spot boxes...")
        for spot in spot_boxes:
            sx1, sy1, sx2, sy2 = spot
            cv2.rectangle(image_cv2_draw, (sx1, sy1), (sx2, sy2), (0, 0, 200), 1) # Darker Red, thinner line


        # --- Prepare Annotated Image for JSON Response ---
        print("Encoding annotated image to Base64...")
        # Convert final annotated OpenCV image (BGR) back to RGB format for PIL
        image_rgb_annotated = cv2.cvtColor(image_cv2_draw, cv2.COLOR_BGR2RGB)
        # Create PIL Image from NumPy array
        annotated_image_pil = Image.fromarray(image_rgb_annotated)

        # Save annotated image to an in-memory BytesIO buffer as JPEG format
        img_io = io.BytesIO()
        annotated_image_pil.save(img_io, 'JPEG', quality=85) # Adjust quality as needed (85 is usually good)
        img_io.seek(0) # Rewind the buffer to the beginning

        # Encode the image bytes from the buffer into a Base64 string
        annotated_image_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        print("Image encoding complete.")

        # --- Prepare Final JSON Response ---
        response_data = {
            'annotated_image': annotated_image_base64,
            'leaf_results': leaf_results_for_json # The list we populated
        }

        print(f"Analysis complete. Returning JSON results for {len(leaf_results_for_json)} leaves.")
        return jsonify(response_data) # Return the JSON object

    except Exception as e:
        # Log the full error traceback to the server console for debugging
        print("An error occurred during the /analyze request:")
        traceback.print_exc()
        # Return a generic error message to the client
        return jsonify({'error': 'An internal server error occurred while processing the image.'}), 500

# --- Run the App ---
if __name__ == '__main__':
    # This block is mainly for local testing. Gunicorn uses the 'app' instance directly.
    # Get port from environment variable (used by Render/Heroku etc.) or default to 5000
    port = int(os.environ.get("PORT", 6000))
    print(f"Flask development server starting on host 0.0.0.0, port {port}")
    # Set debug=False when deploying to production environments like Render
    app.run(host='0.0.0.0', port=port, debug=False)