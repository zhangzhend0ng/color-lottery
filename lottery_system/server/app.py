from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import time
import queue
import threading
import os
import sys


def _base_dir():
    """Return the base directory, works both in dev and PyInstaller."""
    if getattr(sys, "frozen", False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


STATIC_DIR = os.path.join(_base_dir(), "static")

app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

clients = []
clients_lock = threading.Lock()
last_event = None

# --- Input validation: whitelist schema for /api/detect ---
# Harness: common-input-validation (N) — whitelist what IS allowed
ALLOWED_COLORS = {"gold", "red", "blue", "green", "yellow", "white"}
ALLOWED_TYPES = {"ssr", "sr", "r", "n"}
MAX_PRIZE_LEN = 64
MAX_EMOJI_LEN = 8
MAX_CONFIDENCE = 1.0
MIN_CONFIDENCE = 0.0


def _validate_detect_payload(data):
    """Validate the POST /api/detect payload against whitelist schema.

    Returns (True, cleaned) on success, (False, error_message) on failure.
    Harness: python-input-deserialization (N) — validate after parse.
    """
    if not isinstance(data, dict):
        return False, "Payload must be a JSON object"

    # Required fields
    for field in ("color", "prize", "emoji", "type", "confidence"):
        if field not in data:
            return False, f"Missing required field: {field}"

    # color: whitelist
    color = data["color"]
    if not isinstance(color, str) or color not in ALLOWED_COLORS:
        return False, f"Invalid color: {color}"

    # type: whitelist
    ptype = data["type"]
    if not isinstance(ptype, str) or ptype not in ALLOWED_TYPES:
        return False, f"Invalid type: {ptype}"

    # prize: string, length limit
    prize = data["prize"]
    if not isinstance(prize, str) or len(prize) > MAX_PRIZE_LEN:
        return False, "Invalid prize name"

    # emoji: string, length limit
    emoji = data["emoji"]
    if not isinstance(emoji, str) or len(emoji) > MAX_EMOJI_LEN:
        return False, "Invalid emoji"

    # confidence: float, range [0, 1]
    try:
        confidence = float(data["confidence"])
    except (TypeError, ValueError):
        return False, "Invalid confidence value"
    if not (MIN_CONFIDENCE <= confidence <= MAX_CONFIDENCE):
        return False, "Confidence out of range [0, 1]"

    return True, {
        "color": color,
        "prize": prize,
        "emoji": emoji,
        "type": ptype,
        "confidence": round(confidence, 3),
    }


@app.route("/api/detect", methods=["POST"])
def detect():
    """Receive color detection result and broadcast to all SSE clients.

    Harness: common-input-validation (N) — validate ALL fields before use.
    Harness: common-input-validation (C) — reject on failure, generic error to caller.
    """
    global last_event

    raw = request.get_json(silent=True)
    if raw is None:
        return jsonify({"error": "Invalid JSON"}), 400

    ok, result = _validate_detect_payload(raw)
    if not ok:
        # (C) Log sanitized failure (no raw payload)
        print(f"[validation] rejected: {result}")
        return jsonify({"error": "Invalid input"}), 400

    result["timestamp"] = time.time()

    with clients_lock:
        last_event = result
        dead = []
        for q in clients:
            try:
                q.put_nowait(result)
            except queue.Full:
                dead.append(q)
        for q in dead:
            clients.remove(q)

    print(f"[{result['prize']}] {result['color']} -> {len(clients)} clients")
    return jsonify({"ok": True, "clients": len(clients)})


@app.route("/events")
def events():
    """SSE endpoint for frontend to receive push events."""
    def stream():
        q = queue.Queue(maxsize=10)
        with clients_lock:
            clients.append(q)

        try:
            if last_event:
                yield f"data: {json.dumps(last_event, ensure_ascii=False)}\n\n"

            while True:
                data = q.get()
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        except GeneratorExit:
            with clients_lock:
                if q in clients:
                    clients.remove(q)

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/detect")
def detect_page():
    """Phone-side color detection page."""
    return app.send_static_file("detect.html")


@app.route("/api/mapping")
def get_mapping():
    """Serve the color-to-prize mapping for the phone detector."""
    mapping_path = os.path.join(_base_dir(), "color_mapping.json")
    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"error": "mapping not found"}), 404


@app.route("/")
def index():
    return app.send_static_file("reveal.html")


if __name__ == "__main__":
    print("=== Lottery Server Starting ===")
    print("  Display:  http://localhost:5000")
    print("  Phone:    http://<this-PC-IP>:5000/detect")
    print("  Tip: Use Firefox on phone, or Chrome with flag set (see page hint)")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)