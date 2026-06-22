"""Integration test: simulate color detection events through the full pipeline.

Usage:
    Terminal 1: cd server && python app.py
    Terminal 2: python test_integration.py
    Browser:    http://localhost:5000

Expected: 6 reveal animations play in sequence (3s interval).
"""
import requests
import time
import sys

SERVER = "http://localhost:5000/api/detect"

TEST_EVENTS = [
    {"color": "gold",   "prize": "SSR·至尊传说", "emoji": "⭐", "type": "ssr", "confidence": 0.95},
    {"color": "red",    "prize": "SR·稀有奖励",  "emoji": "💜", "type": "sr",  "confidence": 0.92},
    {"color": "blue",   "prize": "R·精良奖励",   "emoji": "💙", "type": "r",   "confidence": 0.88},
    {"color": "yellow", "prize": "R·幸运参与",   "emoji": "💛", "type": "r",   "confidence": 0.86},
    {"color": "green",  "prize": "N·阳光普照",   "emoji": "🍀", "type": "n",   "confidence": 0.90},
    {"color": "white",  "prize": "N·谢谢参与",   "emoji": "🤍", "type": "n",   "confidence": 0.91},
]

print("=== Integration Test ===")
print("Prerequisites: server running (python app.py)")
print("Prerequisites: browser open at http://localhost:5000")
print()

for i, event in enumerate(TEST_EVENTS, 1):
    print(f"[{i}/{len(TEST_EVENTS)}] Sending: {event['prize']} ({event['color']})", end=" ... ")
    try:
        resp = requests.post(SERVER, json=event, timeout=2)
        if resp.ok:
            data = resp.json()
            print(f"OK (clients: {data.get('clients', '?')})")
        else:
            print(f"FAIL: {resp.status_code} {resp.text}")
    except requests.exceptions.ConnectionError:
        print("FAIL: Server not running")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("FAIL: Timeout")
    time.sleep(3)

print("\nAll events sent. Verify each one triggered the reveal animation in the browser.")