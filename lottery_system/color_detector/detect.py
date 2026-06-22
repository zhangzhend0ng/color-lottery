import cv2
import numpy as np
import pickle
import json
import time
import requests
import os
import sys
from train import extract_features, COLOR_NAMES, get_camera_source, open_camera

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
MAPPING_PATH = os.path.join(os.path.dirname(__file__), "color_mapping.json")
SERVER_URL = "http://localhost:5000/api/detect"

DEBOUNCE_FRAMES = 10
COOLDOWN_SEC = 5.0

REQUIRED_FIELDS = {"prize", "emoji", "type"}
VALID_TYPES = {"ssr", "sr", "r", "n"}


def load_model():
    """Load trained KNN model from pickle file."""
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found: {MODEL_PATH}")
        print("Run generate_model.py for a quick pre-trained model, or train.py to collect real samples.")
        sys.exit(1)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_mapping():
    """Load and validate color-to-prize mapping from JSON.

    Harness: common-input-validation (N) — whitelist schema validation.
    Harness: python-input-deserialization (N) — validate parsed JSON structure.
    """
    if not os.path.exists(MAPPING_PATH):
        print(f"ERROR: Mapping not found: {MAPPING_PATH}")
        sys.exit(1)
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    if not isinstance(mapping, dict):
        print("ERROR: color_mapping.json must be a JSON object")
        sys.exit(1)

    for color_name in COLOR_NAMES:
        if color_name not in mapping:
            print(f"ERROR: Missing color '{color_name}' in color_mapping.json")
            sys.exit(1)
        entry = mapping[color_name]
        if not isinstance(entry, dict):
            print(f"ERROR: Entry for '{color_name}' must be an object")
            sys.exit(1)
        missing = REQUIRED_FIELDS - set(entry.keys())
        if missing:
            print(f"ERROR: '{color_name}' missing fields: {missing}")
            sys.exit(1)
        if entry["type"] not in VALID_TYPES:
            print(f"ERROR: '{color_name}' has invalid type '{entry['type']}', must be one of {VALID_TYPES}")
            sys.exit(1)

    return mapping


def detect_color(model, hsv_roi):
    """Return (color_name, confidence) for an HSV ROI patch."""
    features = extract_features(hsv_roi).reshape(1, -1)
    label = model.predict(features)[0]
    distances, _ = model.kneighbors(features)
    confidence = 1.0 / (1.0 + distances.mean())
    return COLOR_NAMES[label], confidence


if __name__ == "__main__":
    model = load_model()
    mapping = load_mapping()
    source = get_camera_source()
    cap, cam_label = open_camera(source)

    current_color = None
    streak = 0
    print("=== Color Detector Started ===")
    print(f"Camera: {cam_label}")
    print(f"Debounce: {DEBOUNCE_FRAMES} frames, Cooldown: {COOLDOWN_SEC}s")
    print("Place ball in the center ROI box")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        x1, y1 = w // 2 - 100, h // 2 - 100
        x2, y2 = w // 2 + 100, h // 2 + 100
        roi_bgr = frame[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        color_name, confidence = detect_color(model, hsv_roi)

        if color_name == current_color:
            streak += 1
        else:
            current_color = color_name
            streak = 1

        now = time.time()
        if streak >= DEBOUNCE_FRAMES and (now - last_trigger) > COOLDOWN_SEC:
            prize = mapping.get(color_name, {})
            if prize:
                print(f"Detected: {color_name} -> {prize['prize']} (confidence {confidence:.2f})")
                payload = {
                    "color": color_name,
                    "prize": prize["prize"],
                    "emoji": prize["emoji"],
                    "type": prize["type"],
                    "confidence": round(confidence, 3),
                }
                try:
                    resp = requests.post(SERVER_URL, json=payload, timeout=2)
                    if not resp.ok:
                        print(f"  Server returned {resp.status_code}")
                except requests.exceptions.ConnectionError:
                    print("  Server not running, event not sent")
                except requests.exceptions.Timeout:
                    print("  Server timeout")
                last_trigger = now

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{color_name} ({confidence:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Color Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
