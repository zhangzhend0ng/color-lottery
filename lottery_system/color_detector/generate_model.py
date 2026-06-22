"""Generate a pre-trained KNN model from synthetic HSV color samples.

This avoids manual training. The synthetic model works as a starting point
and can be refined later with real samples via train.py if needed.

Note: Real-world accuracy depends on ball colors and lighting. For best
results, adjust HSV_RANGES below to match your actual balls.
"""
import numpy as np
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier

OUTPUT_MODEL = os.path.join(os.path.dirname(__file__), "model.pkl")

# HSV reference ranges (OpenCV: H 0-180, S 0-255, V 0-255)
# Format: (H_low, H_high, S_low, S_high, V_low, V_high)
HSV_RANGES = {
    "gold":   (15, 30,  180, 255, 180, 255),
    "red":    [(0, 10,  150, 255, 100, 255),   # red wraps around in HSV
               (170, 180, 150, 255, 100, 255)],
    "blue":   (95, 125, 120, 255, 80, 255),
    "green":  (45, 80,  100, 255, 80, 255),
    "yellow": (25, 35,  150, 255, 200, 255),
    "white":  (0, 180,  0, 40,   180, 255),
}

SAMPLES_PER_COLOR = 200
NOISE_STD = 3.0  # synthetic noise std dev per channel


def extract_features(hsv_patch):
    """Mirrors train.py extract_features for consistent feature shape."""
    h, w = hsv_patch.shape[:2]
    roi = hsv_patch[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

    mean = roi.mean(axis=(0, 1))
    std = roi.std(axis=(0, 1))

    hist_h = cv2.calcHist([roi], [0], None, [8], [0, 180]).flatten()
    hist_h /= hist_h.sum() + 1e-7
    hist_s = cv2.calcHist([roi], [1], None, [8], [0, 256]).flatten()
    hist_s /= hist_s.sum() + 1e-7

    return np.concatenate([mean, std, hist_h, hist_s])


def generate_synthetic_patch(h_range, s_range, v_range, size=200):
    """Generate a synthetic HSV patch with noise for a given color range."""
    patch = np.zeros((size, size, 3), dtype=np.uint8)

    # Sample base color from the range
    h = np.random.randint(h_range[0], h_range[1] + 1)
    s = np.random.randint(s_range[0], s_range[1] + 1)
    v = np.random.randint(v_range[0], v_range[1] + 1)

    # Fill patch with base + per-pixel noise
    noise_h = np.random.normal(0, NOISE_STD, (size, size)).astype(np.int16)
    noise_s = np.random.normal(0, NOISE_STD, (size, size)).astype(np.int16)
    noise_v = np.random.normal(0, NOISE_STD * 1.5, (size, size)).astype(np.int16)

    patch[:, :, 0] = np.clip(h + noise_h, 0, 180).astype(np.uint8)
    patch[:, :, 1] = np.clip(s + noise_s, 0, 255).astype(np.uint8)
    patch[:, :, 2] = np.clip(v + noise_v, 0, 255).astype(np.uint8)

    return patch


def generate_samples():
    """Generate synthetic training data for all colors."""
    import cv2  # local import so the top-level import error is clear

    color_names = list(HSV_RANGES.keys())
    X, y = [], []

    for label_idx, color_name in enumerate(color_names):
        ranges = HSV_RANGES[color_name]

        # Handle red's dual range
        if isinstance(ranges, list):
            range_list = ranges
        else:
            range_list = [ranges]

        per_subrange = SAMPLES_PER_COLOR // len(range_list)

        for h_low, h_high, s_low, s_high, v_low, v_high in range_list:
            for _ in range(per_subrange):
                patch = generate_synthetic_patch(
                    (h_low, h_high), (s_low, s_high), (v_low, v_high)
                )
                features = extract_features(patch)
                X.append(features)
                y.append(label_idx)

        print(f"  {color_name}: {len([l for l in y if l == label_idx])} samples")

    return np.array(X), np.array(y), color_names


if __name__ == "__main__":
    import cv2

    print("=== Generating Synthetic Training Data ===")
    print(f"Samples per color: {SAMPLES_PER_COLOR}, Noise sigma: {NOISE_STD}")
    print()

    X, y, color_names = generate_samples()

    model = KNeighborsClassifier(n_neighbors=3, weights="distance")
    model.fit(X, y)

    accuracy = model.score(X, y)
    print(f"\nTraining complete: KNN k=3, accuracy {accuracy:.1%}")
    print(f"Total samples: {len(X)}")

    if accuracy < 0.90:
        print("WARNING: Synthetic accuracy < 90%. Real-world accuracy may be lower.")
        print("Consider adjusting HSV_RANGES or running train.py with real ball samples.")

    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved: {OUTPUT_MODEL}")
    print()
    print("The model is ready. Run detect.py to start the color detection pipeline.")
    print("If colors are misidentified, re-run train.py with real ball samples for better accuracy.")