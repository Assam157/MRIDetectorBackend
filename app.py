import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import io
import base64
import traceback
from PIL import Image, UnidentifiedImageError

from skimage import measure
import trimesh
import kagglehub

# --------------------------------------------------
# Flask setup
# --------------------------------------------------
app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# Load hybrid model from Kaggle (Public)
# --------------------------------------------------
print("🔄 Fetching model from Kaggle...")
try:
    # Downloads the model framework variation from your public link
    model_dir = kagglehub.model_download("akashbabuji/1/tensorFlow2/default")
    
    # Locate the 1.keras file inside the downloaded folder directory
    MODEL_PATH = os.path.join(model_dir, "1.keras")
    print(f"📂 Model downloaded to local cache: {MODEL_PATH}")
    
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model from Kaggle: {str(e)}")
    raise e

IMG_SIZE = model.input_shape[1:3]  # (H, W)
print("🧠 Model input size:", IMG_SIZE)

# --------------------------------------------------
# Thresholds
# --------------------------------------------------
MASK_THRESHOLD = 0.5
CLS_THRESHOLD = 0.5
VISIBLE_PIXEL_THRESHOLD = 50
STL_THICKNESS = 10

# --------------------------------------------------
# Utils
# --------------------------------------------------
def preprocess_image(pil_img):
    """
    MRI-safe preprocessing (NO ImageNet normalization)
    """
    img = pil_img.resize(IMG_SIZE)
    img = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def draw_red_mask_on_original(pil_img, seg_mask):
    """
    Overlay red tumor mask on original image
    """
    img = np.array(pil_img).copy()

    # Resize mask to original image size
    mask_resized = cv2.resize(
        seg_mask,
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    binary_mask = mask_resized > MASK_THRESHOLD

    # Create red overlay
    red_overlay = np.zeros_like(img)
    red_overlay[:, :, 0] = 255  # Red channel

    alpha = 0.45  # transparency

    img[binary_mask] = cv2.addWeighted(
        img[binary_mask],
        1 - alpha,
        red_overlay[binary_mask],
        alpha,
        0
    )

    return Image.fromarray(img)

def encode_pil_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# --------------------------------------------------
# Segmentation → Bounding Box
# --------------------------------------------------
def mask_to_bbox(mask):
    binary = (mask > MASK_THRESHOLD).astype(np.uint8)

    labels = measure.label(binary)
    regions = measure.regionprops(labels)

    if not regions:
        return None

    largest = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = largest.bbox

    return (minc, minr, maxc, maxr)

def scale_bbox_to_original(bbox, orig_size):
    """
    Scale bbox from model space → original image space
    """
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    orig_w, orig_h = orig_size

    scale_x = orig_w / IMG_SIZE[1]
    scale_y = orig_h / IMG_SIZE[0]

    return (
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y),
    )

def draw_bbox_on_original(pil_img, bbox):
    img = np.array(pil_img)

    if bbox is None:
        return Image.fromarray(img)

    x1, y1, x2, y2 = bbox

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.putText(
        img,
        "Tumor",
        (x1, max(10, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    return Image.fromarray(img)

# --------------------------------------------------
# STL generation
# --------------------------------------------------
def mask_to_stl_base64(binary_mask):
    volume = np.stack([binary_mask] * STL_THICKNESS, axis=-1)
    verts, faces, normals, _ = measure.marching_cubes(volume, level=0.5)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    stl_bytes = mesh.export(file_type="stl")
    return base64.b64encode(stl_bytes).decode("utf-8")

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        try:
            image = Image.open(file).convert("RGB")
        except UnidentifiedImageError:
            return jsonify({"error": "Invalid image format"}), 400

        orig_size = image.size  # (W, H)

        # ---------------- INFERENCE ----------------
        input_tensor = preprocess_image(image)
        pred_seg, pred_cls = model.predict(input_tensor, verbose=0)

        # ---------------- CLASSIFICATION ----------------
        tumor_probability = float(pred_cls[0][0])
        classifier_prediction = tumor_probability >= CLS_THRESHOLD

        # ---------------- SEGMENTATION ----------------
        seg_mask = pred_seg[0, :, :, 0]
        visible_pixels = int(np.sum(seg_mask > MASK_THRESHOLD))
        visible_tumor = visible_pixels > VISIBLE_PIXEL_THRESHOLD

        # ---------------- BOUNDING BOX ----------------
        bbox_model = None
        bbox_original = None

        if visible_tumor:
            bbox_model = mask_to_bbox(seg_mask)
            bbox_original = scale_bbox_to_original(bbox_model, orig_size)

        overlay_img = image

        if visible_tumor:
            overlay_img = draw_red_mask_on_original(image, seg_mask)

        overlay_base64 = encode_pil_to_base64(overlay_img)

        # ---------------- STL ----------------
        stl_base64 = None
        if visible_tumor:
            binary_mask = (seg_mask > MASK_THRESHOLD).astype(np.uint8)
            stl_base64 = mask_to_stl_base64(binary_mask)

        # ---------------- RESPONSE ----------------
        return jsonify({
            "tumor_probability": round(tumor_probability, 4),
            "classifier_prediction": classifier_prediction,
            "visible_tumor": visible_tumor,
            "visible_pixels": visible_pixels,
            "bbox_model_space": bbox_model,
            "bbox_original_space": bbox_original,
            "overlay_2d": overlay_base64,
            "stl_base64": stl_base64
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Hybrid MRI backend running"})

# --------------------------------------------------
# Run server
# --------------------------------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False
    )
