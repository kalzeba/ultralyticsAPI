from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import gc

app = Flask(__name__)

# Load model
model = YOLO("yolov8n.pt")
model.fuse()
CLASS_NAMES = model.names


# ----------------------------
# IMAGE DECODING SAFE
# ----------------------------
def decode_image(b64: str):
    try:
        if "," in b64:
            b64 = b64.split(",", 1)[1]

        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        return img
    except Exception as e:
        print("decode_image error:", e)
        return None


# ----------------------------
# PREPROCESS SAFE + RESIZE
# ----------------------------
def preprocess(img):
    if img is None:
        raise ValueError("Image is None (decode failed)")

    if len(img.shape) != 3:
        raise ValueError("Invalid image shape")

    # 🔥 resize protection (CRITICAL for Render CPU)
    h, w = img.shape[:2]
    max_size = 640

    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # CLAHE (safe now)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ----------------------------
# HEALTH CHECK
# ----------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "yolov8n",
        "classes": len(CLASS_NAMES)
    })


# ----------------------------
# PREDICTION ROUTE
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("REQUEST RECEIVED")

        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "missing image"}), 400

        img = decode_image(data["image"])
        if img is None:
            return jsonify({"error": "invalid image decode"}), 400

        print("IMAGE DECODED")

        img = preprocess(img)

        print("START YOLO")

        # 🔥 OPTIMIZED INFERENCE
        results = model.predict(
            img,
            imgsz=320,
            conf=0.25,
            iou=0.45,
            verbose=False
        )[0]

        print("END YOLO")

        detections = []

        for box in results.boxes:
            x, y, bw, bh = box.xywh[0].tolist()
            cls_id = int(box.cls[0])

            detections.append({
                "class": CLASS_NAMES.get(cls_id, f"object_{cls_id}"),
                "class_id": cls_id,
                "confidence": round(float(box.conf[0]), 4),
                "bbox": {
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "w": round(bw, 2),
                    "h": round(bh, 2),
                },
            })

        gc.collect()

        return jsonify({
            "count": len(detections),
            "items": detections
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# ----------------------------
# RUN (local only)
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
