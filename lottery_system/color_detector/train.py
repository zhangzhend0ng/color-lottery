import cv2
import numpy as np
import pickle
import json
import os
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict

COLOR_NAMES = ["gold", "red", "blue", "green", "yellow", "white"]
SAMPLES_PER_COLOR = 50
OUTPUT_MODEL = os.path.join(os.path.dirname(__file__), "model.pkl")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "camera_config.json")


def get_camera_source():
    """Read camera source from config. Returns int index or str URL."""
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        src = cfg.get("source", "0")
    except (FileNotFoundError, json.JSONDecodeError):
        src = "0"

    try:
        return int(src)
    except ValueError:
        return src


def open_camera(source):
    """Open camera and return (cap, label). Exits on failure."""
    label = f"webcam {source}" if isinstance(source, int) else "phone camera"
    print(f"Opening camera: {label} ({source})")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera ({source})")
        if isinstance(source, str):
            print("Make sure the phone IP camera app is running and the URL is correct.")
            print("Recommended apps: IP Webcam (Android) / Iriun Webcam (iOS/Android)")
        exit(1)
    return cap, label


def extract_features(image_hsv):
    """Extract HSV color features from ROI region (center crop)."""
    h, w = image_hsv.shape[:2]
    roi = image_hsv[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

    mean = roi.mean(axis=(0, 1))
    std = roi.std(axis=(0, 1))

    hist_h = cv2.calcHist([roi], [0], None, [8], [0, 180]).flatten()
    hist_h /= hist_h.sum() + 1e-7
    hist_s = cv2.calcHist([roi], [1], None, [8], [0, 256]).flatten()
    hist_s /= hist_s.sum() + 1e-7

    return np.concatenate([mean, std, hist_h, hist_s])


def collect_samples():
    """Interactive sample collection with GUI overlay."""
    source = get_camera_source()
    cap, _ = open_camera(source)

    data = defaultdict(list)

    print("=== Color Sample Collection ===")
    print("Keys: 1=Gold 2=Red 3=Blue 4=Green 5=Yellow 6=White  q=Quit  s=Save")
    print("Place ball in the center ROI box and press the corresponding number key")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        x1, y1 = w // 2 - 100, h // 2 - 100
        x2, y2 = w // 2 + 100, h // 2 + 100
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for i, name in enumerate(COLOR_NAMES):
            count = len(data[name])
            cv2.putText(
                frame,
                f"{i + 1}:{name}({count})",
                (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Sample Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            break
        elif ord("1") <= key <= ord("6"):
            idx = key - ord("1")
            roi_bgr = frame[y1:y2, x1:x2]
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            features = extract_features(hsv)
            data[COLOR_NAMES[idx]].append(features)
            print(f"  {COLOR_NAMES[idx]}: {len(data[COLOR_NAMES[idx]])} samples")

    cap.release()
    cv2.destroyAllWindows()
    return data


def train_model(data):
    """Train a KNN classifier on collected samples."""
    X, y = [], []
    for label_idx, color_name in enumerate(COLOR_NAMES):
        for features in data[color_name]:
            X.append(features)
            y.append(label_idx)

    X = np.array(X)
    y = np.array(y)

    if len(X) < len(COLOR_NAMES) * 5:
        print("WARNING: Very few samples. Collect at least 5 per color for reliable results.")

    model = KNeighborsClassifier(n_neighbors=3, weights="distance")
    model.fit(X, y)

    accuracy = model.score(X, y)
    print(f"\nTraining complete: KNN k=3, training accuracy {accuracy:.1%}")

    if accuracy < 0.85:
        print("WARNING: Accuracy < 85%. Increase samples or check lighting conditions.")
    return model


if __name__ == "__main__":
    data = collect_samples()
    if data is None:
        exit(1)
    model = train_model(data)
    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved: {OUTPUT_MODEL}")