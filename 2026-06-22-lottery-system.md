# 智能摇奖系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `dispatching-parallel-agents` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 3D 打印物理摇奖机 → 摄像头识别小球颜色 → 触发屏幕出奖揭示动画，形成完整抽奖流水线

**Architecture:** 三个独立子系统，通过 HTTP 事件松耦合。Python 端持续采集摄像头画面并做颜色分类，识别到球色后 POST 到本地 Web 服务器；前端页面监听 Server-Sent Events，收到颜色码后触发对应奖项的粒子爆炸+揭示动画。

**Tech Stack:** Python 3.11+ / OpenCV / scikit-learn (KNN) / Flask + Flask-SocketIO / 前端纯 HTML+CSS+JS (tsparticles + Genshin-style reveal animation)

**3D 模型推荐:** Printables 622130 (Portable Lottery Machine) — 手摇出球通道清晰，适合架摄像头

---

## 系统架构图

```
┌─────────────────┐     USB Camera      ┌──────────────────┐     HTTP POST      ┌───────────────────┐
│  物理摇奖机       │ ─────────────────→ │  color_detector   │ ─────────────────→ │  Flask + SocketIO  │
│  (3D Printed)    │    出球口对准        │  (Python/OpenCV)  │  /api/detect      │  (本地 5000 端口)   │
│  6 色彩球          │                    │  KNN 颜色分类器    │                    │  SSE 推送事件       │
└─────────────────┘                    └──────────────────┘                    └────────┬──────────┘
                                                                                       │ SSE stream
                                                                          ┌────────────▼───────────┐
                                                                          │  reveal.html            │
                                                                          │  tsparticles 礼花       │
                                                                          │  Genshin-style 揭示动画   │
                                                                          │  奖项信息卡片弹出         │
                                                                          └────────────────────────┘
```

---

## 项目文件结构

```
lottery_system/
├── color_detector/
│   ├── train.py              # KNN 颜色分类器训练脚本
│   ├── detect.py             # 摄像头实时检测主程序
│   ├── camera_config.json    # 摄像头源配置 (USB / 手机 IP)
│   ├── model.pkl             # 训练好的 KNN 模型（生成物）
│   └── color_mapping.json    # 颜色名 → 奖项映射 (gacha 风格)
├── server/
│   ├── app.py                # Flask + SocketIO 服务端
│   └── static/
│       └── reveal.html       # 前端揭示动画页面
└── README.md                 # 环境搭建 + 使用说明
```

---

## Task 1: 训练颜色分类器

**Files:**
- Create: `lottery_system/color_detector/train.py`
- Create: `lottery_system/color_detector/color_mapping.json`
- Create: `lottery_system/color_detector/model.pkl`

- [ ] **Step 1: 建立 color_mapping.json — 颜色到奖项的映射**

```json
{
  "gold":    {"prize": "SSR·至尊传说", "emoji": "⭐", "type": "ssr"},
  "red":     {"prize": "SR·稀有奖励",  "emoji": "💜", "type": "sr"},
  "blue":    {"prize": "R·精良奖励",   "emoji": "💙", "type": "r"},
  "yellow":  {"prize": "R·幸运参与",   "emoji": "💛", "type": "r"},
  "green":   {"prize": "N·阳光普照",   "emoji": "🍀", "type": "n"},
  "white":   {"prize": "N·谢谢参与",   "emoji": "🤍", "type": "n"}
}
```

- [ ] **Step 2: 编写 train.py — 带 GUI 标注界面的训练脚本**

```python
import cv2
import numpy as np
import pickle
import json
import os
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict

COLOR_NAMES = ["gold", "red", "blue", "green", "yellow", "white"]
# 每种颜色手动采集 50 个样本（HSV 均值 + 直方图特征）
SAMPLES_PER_COLOR = 50
OUTPUT_MODEL = os.path.join(os.path.dirname(__file__), "model.pkl")

def extract_features(image_hsv):
    """提取 ROI 区域的 HSV 颜色特征"""
    # 中心区域取样，避免背景干扰
    h, w = image_hsv.shape[:2]
    roi = image_hsv[h//4:3*h//4, w//4:3*w//4]

    # 特征：HSV 均值 + 每个通道的 std
    mean = roi.mean(axis=(0, 1))
    std = roi.std(axis=(0, 1))

    # 直方图特征（各通道 8 bins）
    hist_h = cv2.calcHist([roi], [0], None, [8], [0, 180]).flatten()
    hist_h /= hist_h.sum() + 1e-7
    hist_s = cv2.calcHist([roi], [1], None, [8], [0, 256]).flatten()
    hist_s /= hist_s.sum() + 1e-7

    return np.concatenate([mean, std, hist_h, hist_s])


def collect_samples():
    """交互式采集训练样本"""
    cap = cv2.VideoCapture(0)
    data = defaultdict(list)

    print("=== 颜色样本采集 ===")
    print("按键: 1=金 2=红 3=蓝 4=绿 5=黄 6=白  q=退出 s=保存")
    print("将小球放在出球口区域，按对应数字键采集")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 画出 ROI 框（中心 200x200）
        h, w = frame.shape[:2]
        x1, y1 = w//2 - 100, h//2 - 100
        x2, y2 = w//2 + 100, h//2 + 100
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 显示各颜色已采集数量
        for i, name in enumerate(COLOR_NAMES):
            count = len(data[name])
            cv2.putText(frame, f"{i+1}:{name}({count})", (10, 30 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Sample Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            break
        elif ord('1') <= key <= ord('6'):
            idx = key - ord('1')
            roi = frame[y1:y2, x1:x2]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            features = extract_features(hsv)
            data[COLOR_NAMES[idx]].append(features)
            print(f"  {COLOR_NAMES[idx]}: {len(data[COLOR_NAMES[idx]])} samples")

    cap.release()
    cv2.destroyAllWindows()
    return data


def train_model(data):
    """训练 KNN 分类器"""
    X, y = [], []
    for label_idx, color_name in enumerate(COLOR_NAMES):
        for features in data[color_name]:
            X.append(features)
            y.append(label_idx)

    X = np.array(X)
    y = np.array(y)

    model = KNeighborsClassifier(n_neighbors=3, weights='distance')
    model.fit(X, y)

    accuracy = model.score(X, y)
    print(f"\n训练完成: KNN k=3, 训练集准确率 {accuracy:.1%}")

    if accuracy < 0.85:
        print("⚠ 准确率偏低，建议增加样本量或检查光照条件")
    return model


if __name__ == "__main__":
    data = collect_samples()
    model = train_model(data)
    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump(model, f)
    print(f"模型已保存: {OUTPUT_MODEL}")
```

- [ ] **Step 3: 运行训练并验证**

```bash
cd lottery_system/color_detector
python train.py
# 依次放入 6 种颜色小球，各采集 30-50 个样本
# 确认训练集准确率 > 85%
```

- [ ] **Step 4: Commit**

```bash
git add lottery_system/color_detector/train.py color_mapping.json
git commit -m "feat: add color classifier training script with KNN"
```

---

## Task 2: 摄像头实时检测 + 事件发布

**Files:**
- Create: `lottery_system/color_detector/detect.py`

- [ ] **Step 1: 编写 detect.py — 实时检测主程序**

```python
import cv2
import numpy as np
import pickle
import json
import time
import requests
import os
from train import extract_features, COLOR_NAMES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
MAPPING_PATH = os.path.join(os.path.dirname(__file__), "color_mapping.json")
SERVER_URL = "http://localhost:5000/api/detect"

# 防抖：同一颜色连续 N 帧确认后才触发
DEBOUNCE_FRAMES = 10
# 两次触发的最小间隔（秒）
COOLDOWN_SEC = 5.0


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_mapping():
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_color(model, hsv_roi):
    """返回 (颜色名, 置信度距离)"""
    features = extract_features(hsv_roi).reshape(1, -1)
    label = model.predict(features)[0]
    # 用最近邻距离衡量置信度
    distances, _ = model.kneighbors(features)
    confidence = 1.0 / (1.0 + distances.mean())
    return COLOR_NAMES[label], confidence


if __name__ == "__main__":
    model = load_model()
    mapping = load_mapping()
    cap = cv2.VideoCapture(0)

    # 防抖状态
    current_color = None
    streak = 0
    last_trigger = 0

    print("=== 摄像头颜色检测已启动 ===")
    print(f"防抖: {DEBOUNCE_FRAMES} 帧确认, 冷却: {COOLDOWN_SEC}s")
    print("将小球放入出球口区域 (画面中心方框)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ROI：中心 200x200
        h, w = frame.shape[:2]
        x1, y1 = w//2 - 100, h//2 - 100
        x2, y2 = w//2 + 100, h//2 + 100
        roi = frame[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        color_name, confidence = detect_color(model, hsv_roi)

        # 防抖：连续 N 帧同一颜色才确认
        if color_name == current_color:
            streak += 1
        else:
            current_color = color_name
            streak = 1

        # 触发判断
        now = time.time()
        if streak >= DEBOUNCE_FRAMES and (now - last_trigger) > COOLDOWN_SEC:
            prize = mapping.get(color_name, {})
            if prize:
                print(f"🏆 检测到: {color_name} → {prize['prize']} (置信度 {confidence:.2f})")
                try:
                    resp = requests.post(SERVER_URL, json={
                        "color": color_name,
                        "prize": prize["prize"],
                        "emoji": prize["emoji"],
                        "type": prize["type"],
                        "confidence": round(confidence, 3)
                    }, timeout=2)
                except requests.exceptions.ConnectionError:
                    print("  ⚠ 服务器未启动，事件未发送")
                last_trigger = now

        # 画面叠加
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{color_name} ({confidence:.2f})", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Color Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```

- [ ] **Step 2: 验证检测脚本可启动**

```bash
cd lottery_system/color_detector
python detect.py
# 确认摄像头打开，画面中心出现识别框
# 红色小球放入 → 控制台打印识别结果
```

- [ ] **Step 3: Commit**

```bash
git add lottery_system/color_detector/detect.py
git commit -m "feat: add real-time color detection with debounce and HTTP event publishing"
```

---

## Task 3: Flask 服务端 + SSE 推送

**Files:**
- Create: `lottery_system/server/app.py`

- [ ] **Step 1: 编写 app.py — 事件接收 + SSE 广播**

```python
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import time
import queue
import threading

app = Flask(__name__)
CORS(app)

# SSE 客户端队列
clients = []
clients_lock = threading.Lock()

# 事件历史（用于新客户端重放最近一条）
last_event = None


@app.route("/api/detect", methods=["POST"])
def detect():
    """接收颜色检测结果并广播给所有 SSE 客户端"""
    global last_event
    data = request.get_json()
    if not data:
        return jsonify({"error": "invalid json"}), 400

    # 补时间戳
    data["timestamp"] = time.time()

    with clients_lock:
        last_event = data
        # 广播给所有连接中的客户端
        dead = []
        for q in clients:
            try:
                q.put_nowait(data)
            except queue.Full:
                dead.append(q)
        for q in dead:
            clients.remove(q)

    print(f"[{data.get('prize', '?')}] {data.get('color', '?')} → {len(clients)} clients")
    return jsonify({"ok": True, "clients": len(clients)})


@app.route("/events")
def events():
    """SSE 端点，前端 connect 此 URL 接收推送"""
    def stream():
        q = queue.Queue(maxsize=10)
        with clients_lock:
            clients.append(q)

        try:
            # 如果有历史事件，先发送
            if last_event:
                yield f"data: {json.dumps(last_event, ensure_ascii=False)}\n\n"

            while True:
                data = q.get()  # 阻塞等待
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        except GeneratorExit:
            with clients_lock:
                if q in clients:
                    clients.remove(q)

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/")
def index():
    return app.send_static_file("reveal.html")


if __name__ == "__main__":
    print("=== 摇奖系统服务器启动 ===")
    print("  前端页面: http://localhost:5000")
    print("  SSE端点:  http://localhost:5000/events")
    print("  检测接口: POST http://localhost:5000/api/detect")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
```

- [ ] **Step 2: 安装依赖并启动验证**

```bash
cd lottery_system/server
pip install flask flask-cors
python app.py
# 确认: http://localhost:5000 返回 200
# curl -X POST http://localhost:5000/api/detect -H "Content-Type: application/json" -d '{"color":"gold","prize":"SSR·至尊传说","emoji":"⭐","type":"ssr","confidence":0.95}'
# 确认返回 {"ok":true,"clients":0}
```

- [ ] **Step 3: Commit**

```bash
git add lottery_system/server/app.py
git commit -m "feat: add Flask server with SSE event broadcasting"
```

---

## Task 4: 前端揭示动画页面

**Files:**
- Create: `lottery_system/server/static/reveal.html`

- [ ] **Step 1: 编写 reveal.html — 抽卡风格揭示动画**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>抽奖结果</title>
<script src="https://cdn.jsdelivr.net/npm/tsparticles-confetti@2.12.0/tsparticles.confetti.bundle.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    width: 100vw; height: 100vh; overflow: hidden;
    background: radial-gradient(ellipse at center, #1a0a2e 0%, #0d0015 100%);
    display: flex; align-items: center; justify-content: center;
    font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
  }

  /* 初始星空背景 */
  .stars {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none;
  }
  .star {
    position: absolute;
    background: white;
    border-radius: 50%;
    animation: twinkle var(--dur) ease-in-out infinite;
    animation-delay: var(--delay);
    opacity: 0;
  }
  @keyframes twinkle {
    0%, 100% { opacity: 0.2; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.5); }
  }

  /* 流星坠落 (SSR 专属) */
  .shooting-star {
    position: fixed;
    width: 3px;
    height: 80px;
    background: linear-gradient(to bottom, rgba(255,215,0,0), rgba(255,215,0,1));
    border-radius: 2px;
    animation: shoot 0.8s ease-out forwards;
    opacity: 0;
    z-index: 10;
  }
  @keyframes shoot {
    0% { transform: translate(0, -100vh) rotate(-35deg); opacity: 1; }
    70% { opacity: 1; }
    100% { transform: translate(200px, 100vh) rotate(-35deg); opacity: 0; }
  }

  /* 光芒爆发 */
  .burst {
    position: fixed;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 10px; height: 10px;
    background: radial-gradient(circle, var(--glow) 0%, transparent 70%);
    border-radius: 50%;
    animation: burst 1.2s ease-out forwards;
    pointer-events: none;
    z-index: 5;
  }
  @keyframes burst {
    0% { width: 10px; height: 10px; opacity: 0; }
    30% { width: 600px; height: 600px; opacity: 0.8; }
    100% { width: 1200px; height: 1200px; opacity: 0; }
  }

  /* 结果卡片 */
  .result-card {
    position: relative; z-index: 20;
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 48px 64px;
    text-align: center;
    animation: cardIn 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
    opacity: 0;
    transform: scale(0.5);
  }
  @keyframes cardIn {
    to { opacity: 1; transform: scale(1); }
  }
  .result-card.ssr { border-color: rgba(255,215,0,0.5); box-shadow: 0 0 60px rgba(255,215,0,0.2); }
  .result-card.sr  { border-color: rgba(147,112,219,0.5); box-shadow: 0 0 40px rgba(147,112,219,0.2); }
  .result-card.r   { border-color: rgba(100,149,237,0.4); }
  .result-card.n   { border-color: rgba(255,255,255,0.1); }

  .emoji {
    font-size: 72px;
    display: block;
    animation: emojiBounce 0.8s ease-out;
  }
  @keyframes emojiBounce {
    0% { transform: translateY(-60px); opacity: 0; }
    60% { transform: translateY(10px); }
    100% { transform: translateY(0); opacity: 1; }
  }

  .prize-name {
    font-size: 36px; font-weight: 700;
    color: white; margin: 16px 0 8px;
    text-shadow: 0 0 20px rgba(255,255,255,0.3);
  }
  .prize-detail {
    font-size: 16px; color: rgba(255,255,255,0.5);
  }

  .idle-hint {
    position: fixed; bottom: 32px;
    color: rgba(255,255,255,0.3);
    font-size: 14px;
  }
</style>
</head>
<body>

<div id="starfield" class="stars"></div>
<div id="card-container"></div>
<div class="idle-hint">等待摇奖结果...</div>

<script>
// -- 星空背景 --
function createStarfield() {
  const container = document.getElementById('starfield');
  for (let i = 0; i < 80; i++) {
    const star = document.createElement('div');
    star.className = 'star';
    star.style.left = Math.random() * 100 + '%';
    star.style.top = Math.random() * 100 + '%';
    star.style.width = star.style.height = (Math.random() * 2 + 1) + 'px';
    star.style.setProperty('--dur', (Math.random() * 3 + 1) + 's');
    star.style.setProperty('--delay', Math.random() * 4 + 's');
    container.appendChild(star);
  }
}
createStarfield();

// -- 流星 --
function spawnShootingStar() {
  const star = document.createElement('div');
  star.className = 'shooting-star';
  star.style.left = (Math.random() * 80 + 10) + '%';
  star.style.top = -(Math.random() * 20) + '%';
  document.body.appendChild(star);
  star.addEventListener('animationend', () => star.remove());
}

// -- 光芒爆发 --
function spawnBurst(colorHex) {
  const burst = document.createElement('div');
  burst.className = 'burst';
  burst.style.setProperty('--glow', colorHex);
  document.body.appendChild(burst);
  burst.addEventListener('animationend', () => burst.remove());
}

// -- 礼花 --
async function launchConfetti(type) {
  const defaults = { spread: 360, ticks: 100, gravity: 1, decay: 0.94, startVelocity: 30, colors: ['#ffd700', '#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24'] };

  if (type === 'ssr') {
    await confetti({ ...defaults, particleCount: 200, spread: 100, origin: { x: 0.5, y: 0.4 }, colors: ['#FFD700', '#FFA500', '#FF6347', '#FFD700', '#FFF8DC'] });
    await confetti({ ...defaults, particleCount: 150, spread: 120, origin: { x: 0.3, y: 0.5 } });
    await confetti({ ...defaults, particleCount: 150, spread: 120, origin: { x: 0.7, y: 0.5 } });
  } else if (type === 'sr') {
    await confetti({ ...defaults, particleCount: 120, spread: 80, origin: { x: 0.5, y: 0.5 } });
  } else if (type === 'r') {
    await confetti({ ...defaults, particleCount: 60, spread: 60, origin: { x: 0.5, y: 0.6 } });
  }
  // type 'n' 无礼花
}

// -- 展示结果 --
function showResult(data) {
  const container = document.getElementById('card-container');
  const colorGlowMap = {
    gold: '#FFD700', red: '#9B59B6', blue: '#4488FF',
    green: '#44CC44', yellow: '#FFAA00', white: '#999999'
  };

  // 流星
  if (data.type === 'ssr') {
    for (let i = 0; i < 3; i++) {
      setTimeout(() => spawnShootingStar(), i * 150);
    }
  }

  // 光芒
  setTimeout(() => {
    spawnBurst(colorGlowMap[data.color] || '#ffffff');
  }, 400);

  // 卡片
  const card = document.createElement('div');
  card.className = `result-card ${data.type || 'r'}`;

  card.innerHTML = `
    <span class="emoji">${data.emoji || '🎉'}</span>
    <div class="prize-name">${data.prize || '未知奖项'}</div>
    <div class="prize-detail">小球颜色: ${data.color || '?'}</div>
  `;

  container.innerHTML = '';
  container.appendChild(card);

  // 礼花
  launchConfetti(data.type || 'r');

  // SSR 额外流星
  if (data.type === 'ssr') {
    setTimeout(() => spawnShootingStar(), 1200);
  }
}

// -- SSE 连接 --
function connectSSE() {
  const es = new EventSource('/events');
  es.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      showResult(data);
    } catch (e) {
      console.error('SSE parse error:', e);
    }
  };
  es.onerror = () => {
    console.warn('SSE connection lost, retrying in 2s...');
    es.close();
    setTimeout(connectSSE, 2000);
  };
}
connectSSE();
</script>
</body>
</html>
```

- [ ] **Step 2: 浏览器打开验证**

```bash
cd lottery_system/server
python app.py
# 浏览器打开 http://localhost:5000
# 确认星空背景 + "等待摇奖结果" 显示
# 另开终端 curl POST 测试事件触发
```

- [ ] **Step 3: Commit**

```bash
git add lottery_system/server/static/reveal.html
git commit -m "feat: add prize reveal animation page with confetti, shooting stars, and SSR effects"
```

---

## Task 5: 联调测试

**Files:**
- Create: `lottery_system/test_integration.py`

- [ ] **Step 1: 编写集成测试脚本**

```python
"""模拟颜色检测端发送事件，验证前端动画触发"""
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

print("=== 集成测试 ===")
print("前提: 服务器已启动 (python app.py)")
print("前提: 浏览器已打开 http://localhost:5000")
print()

for i, event in enumerate(TEST_EVENTS, 1):
    print(f"[{i}/{len(TEST_EVENTS)}] 发送: {event['prize']} ({event['color']})", end=" ... ")
    try:
        resp = requests.post(SERVER, json=event, timeout=2)
        if resp.ok:
            print(f"OK (clients: {resp.json().get('clients', '?')})")
        else:
            print(f"FAIL: {resp.status_code}")
    except requests.exceptions.ConnectionError:
        print("FAIL: 服务器未启动")
        sys.exit(1)
    time.sleep(3)  # 间隔 3 秒观察每个动画

print("\n✅ 全部事件发送完成，确认浏览器端每次都有动画响应")
```

- [ ] **Step 2: 运行集成测试**

```bash
# 终端 1
cd lottery_system/server && python app.py

# 终端 2
cd lottery_system && python test_integration.py

# 浏览器: http://localhost:5000
# 预期: 依次看到 6 个奖项的揭示动画，SSR 有流星+金色爆炸+200粒子礼花
```

- [ ] **Step 3: Commit**

```bash
git add lottery_system/test_integration.py
git commit -m "test: add integration test for end-to-end event pipeline"
```

---

## Task 6: README 环境文档

**Files:**
- Create: `lottery_system/README.md`

- [ ] **Step 1: 编写 README.md**

```markdown
# 智能摇奖系统

3D 打印摇奖机 + 摄像头颜色识别 + 屏幕揭示动画

## 硬件清单

| 物料 | 数量 | 用途 |
|------|------|------|
| 3D 打印摇奖机 (Printables #622130) | 1 | 物理摇奖装置 |
| 6 色小球 (金/红/蓝/绿/黄/白) | 各1 | 代表不同奖项 |
| USB 摄像头 (1080p) | 1 | 对准出球口采集颜色 |
| 电脑 (Windows/macOS) | 1 | 运行识别+动画 |
| 显示器/投影 | 1 | 展示抽奖动画 |

## 环境搭建

```bash
# 安装依赖
pip install opencv-python scikit-learn flask flask-cors numpy requests

# 训练颜色分类器 (首次使用必须)
cd color_detector
python train.py
# 依次放入 6 种彩色小球，各采集 30-50 个样本
# 确保训练集准确率 > 85%
```

## 运行

```bash
# 终端 1: 启动服务器 + 前端页面
cd server
python app.py
# 浏览器打开 http://localhost:5000

# 终端 2: 启动摄像头检测
cd color_detector
python detect.py

# 现在转动摇奖机，小球落入出球口即可自动识别并触发动画
```

## 系统架构

```
摇奖机出球 → 摄像头 → detect.py (KNN 分类) → POST /api/detect
                                            ↓
浏览器 ← SSE /events ← app.py (Flask + SocketIO)
  ↓
揭示动画 (流星/光芒/礼花/卡片)
```

## 自定义

- 奖项配置: 修改 `color_detector/color_mapping.json`
- 动画效果: 修改 `server/static/reveal.html` 中的 CSS animation
- 检测灵敏度: 修改 `detect.py` 中的 `DEBOUNCE_FRAMES` 和 `COOLDOWN_SEC`
```

- [ ] **Step 2: Commit**

```bash
git add lottery_system/README.md
git commit -m "docs: add project README with setup and usage instructions"
```

---

## 3D 打印参考

| 模型 | URL |
|------|-----|
| Portable Lottery Machine (推荐) | https://www.printables.com/model/622130 |
| Mini Lottery Bingo Ball Tumbler (轻量) | https://www.printables.com/model/449905 |
| Bingo lottery drum (电动改装) | https://www.printables.com/model/343536 |
| Hand Crank Lottery Drum | https://www.thingiverse.com/thing:4516807 |

**推荐 Printables 622130**: 出球滑道设计清晰，旋桨拨球可靠性高，摄像头对准出球口即可稳定识别。建议打印时出球口上方加装一个遮光罩（减少环境光干扰），提升颜色识别准确率。

---

## Self-Review

**1. Spec coverage:**
- 3D 打印摇奖机 → Task 末尾提供模型推荐链接和选型建议 ✓
- 摄像头识别球色 → Task 1 (训练) + Task 2 (检测) ✓
- 播放出奖动画 → Task 3 (服务端推送) + Task 4 (前端动画) ✓
- 联调测试 → Task 5 (集成测试) ✓
- 文档 → Task 6 (README) ✓

**2. Placeholder scan:** 无 TBD/TODO/占位符，所有代码步骤均为完整可运行代码。

**3. Type consistency:**
- `color_mapping.json` 的字段 (`prize`, `emoji`, `type`) 在 `detect.py` 发送、`app.py` 透传、`reveal.html` 消费，三方一致 ✓
- `extract_features()` 在 `train.py` 定义，`detect.py` 导入，接口一致 ✓
- SSE 事件格式: `{color, prize, emoji, type, confidence, timestamp}` — `detect.py` 发送、`app.py` 补充时间戳、`reveal.html` 解析，字段名全链路统一 ✓

