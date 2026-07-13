import os
import io
import base64
import traceback
import glob

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
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
    model_dir = kagglehub.model_download("akashbabuji/1/tensorFlow2/default")
    print(f"📂 Model downloaded to: {model_dir}")

    # Look for a .keras file (may be named differently)
    keras_files = glob.glob(os.path.join(model_dir, "*.keras"))
    if not keras_files:
        # If not found, try to find any model file inside
        keras_files = glob.glob(os.path.join(model_dir, "**", "*.keras"), recursive=True)
    if not keras_files:
        raise FileNotFoundError(f"No .keras file found in {model_dir}")

    MODEL_PATH = keras_files[0]  # use first found .keras
    print(f"🧠 Using model file: {MODEL_PATH}")
    
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {str(e)}")
    raise e

IMG_SIZE = tuple(model.input_shape[1:3])  # (H, W)
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
    """Simple resize & normalize (no ImageNet stats)."""
    img = pil_img.resize(IMG_SIZE)
    img = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def draw_red_mask_on_original(pil_img, seg_mask):
    """Overlay semi‑transparent red mask on original."""
    img = np.array(pil_img).copy()
    mask_resized = cv2.resize(seg_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    binary_mask = mask_resized > MASK_THRESHOLD

    red_overlay = np.zeros_like(img)
    red_overlay[:, :, 0] = 255  # red channel only
    alpha = 0.45

    # Blend only where mask is true
    blended = cv2.addWeighted(img, 1 - alpha, red_overlay, alpha, 0)
    img[binary_mask] = blended[binary_mask]
    return Image.fromarray(img)

def encode_pil_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# --------------------------------------------------
# Bounding box helpers
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
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    orig_w, orig_h = orig_size
    scale_x = orig_w / IMG_SIZE[1]
    scale_y = orig_h / IMG_SIZE[0]
    return (int(x1 * scale_x), int(y1 * scale_y),
            int(x2 * scale_x), int(y2 * scale_y))

def draw_bbox_on_original(pil_img, bbox):
    img = np.array(pil_img)
    if bbox is None:
        return Image.fromarray(img)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(img, "Tumor", (x1, max(10, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return Image.fromarray(img)

# --------------------------------------------------
# 3D STL generation
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

        # ---------- Inference ----------
        input_tensor = preprocess_image(image)
        pred_seg, pred_cls = model.predict(input_tensor, verbose=0)

        # ---------- Classification ----------
        tumor_probability = float(pred_cls[0][0])
        classifier_prediction = tumor_probability >= CLS_THRESHOLD

        # ---------- Segmentation ----------
        seg_mask = pred_seg[0, :, :, 0]
        visible_pixels = int(np.sum(seg_mask > MASK_THRESHOLD))
        visible_tumor = visible_pixels > VISIBLE_PIXEL_THRESHOLD

        # ---------- Bounding Box ----------
        bbox_model = mask_to_bbox(seg_mask) if visible_tumor else None
        bbox_original = scale_bbox_to_original(bbox_model, orig_size)

        # ---------- 2D Overlay ----------
        overlay_img = image
        if visible_tumor:
            overlay_img = draw_red_mask_on_original(image, seg_mask)
            # Optionally add bounding box on top
            overlay_img = draw_bbox_on_original(overlay_img, bbox_original)
        overlay_base64 = encode_pil_to_base64(overlay_img)

        # ---------- 3D STL ----------
        stl_base64 = None
        if visible_tumor:
            binary_mask = (seg_mask > MASK_THRESHOLD).astype(np.uint8)
            stl_base64 = mask_to_stl_base64(binary_mask)

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
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Hybrid MRI backend running"})

# --------------------------------------------------
# Run server (Render compatible)
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
