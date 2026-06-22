# 智能摇奖系统

3D 打印物理摇奖机 + 摄像头颜色识别 + 屏幕揭示动画，完整抽卡风格抽奖流水线。

## 抽卡稀有度映射

按照主流抽卡游戏的稀有度配色体系：

| 小球颜色 | 奖项 | 稀有度 | 视觉色 |
|----------|------|--------|--------|
| 🟡 金色 | SSR·至尊传说 | SSR (5★) | #FFD700 金 |
| 🔴 红色 | SR·稀有奖励 | SR (4★) | #9B59B6 紫 |
| 🔵 蓝色 | R·精良奖励 | R (3★) | #4488FF 蓝 |
| 🟡 黄色 | R·幸运参与 | R (3★) | #FFAA00 金 |
| 🟢 绿色 | N·阳光普照 | N | #44CC44 绿 |
| ⚪ 白色 | N·谢谢参与 | N | #999999 灰 |

## 硬件清单

| 物料 | 数量 | 用途 |
|------|------|------|
| 3D 打印摇奖机 (Printables #622130) | 1 | 物理摇奖装置 |
| 6 色彩球 (金/红/蓝/绿/黄/白) | 各1 | 代表不同稀有度奖项 |
| USB 摄像头 (1080p) | 1 | 对准出球口采集颜色 |
| 电脑 (Windows/macOS) | 1 | 运行识别 + 动画 |
| 显示器 / 投影 | 1 | 展示抽卡揭示动画 |

## 快速开始（推荐）

首次使用无需手动训练，自动生成模型：

```bash
cd color_detector
python generate_model.py    # 一行命令，自动生成 model.pkl
```

如果实际使用中颜色识别不准，再用真实小球跑训练：

```bash
python train.py             # 交互式采集真实样本，校准模型
```

## 环境搭建

```bash
# 安装依赖
pip install opencv-python scikit-learn flask flask-cors numpy requests

# 训练颜色分类器（首次使用必须）
cd color_detector
python train.py
# 依次放入 6 种彩色小球，各采集 30-50 个样本
# 确保训练集准确率 > 85%
```

## 摄像头配置

编辑 `color_detector/camera_config.json`：

```json
{
  "source": "0"
}
```

`"0"` 为 USB 摄像头（默认）。使用手机摄像头时改为 IP 地址：

```json
{
  "source": "http://192.168.1.100:8080/video"
}
```

**手机端设置：**

| 平台 | 推荐 App | 视频流地址 |
|------|---------|-----------|
| Android | [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) | `http://<手机IP>:8080/video` |
| iOS / Android | [Iriun Webcam](https://iriun.com/) | `http://<手机IP>:4747/video` |
| iOS | [IP Camera Lite](https://apps.apple.com/app/ip-camera-lite/id578747580) | `http://<手机IP>:8080/video` |

步骤：
1. 手机和电脑连接同一 WiFi
2. 打开 IP 摄像头 App，点击"启动服务器"
3. 记下 App 显示的 IP 地址（如 `192.168.1.100:8080`）
4. 将 `camera_config.json` 中的 `source` 改为完整 URL
5. 将手机摄像头对准摇奖机出球口

## 运行

```bash
# 终端 1: 启动服务器
cd server
python app.py
```

然后：

**手机**（推荐，无需 Python）：

浏览器打开 `http://<电脑IP>:5000/detect`，摄像头对准摇奖机出球口，颜色识别在手机上完成并自动发送结果到服务器。

**电脑**（USB 摄像头）：

```bash
cd color_detector
python generate_model.py   # 首次
python detect.py
```

**展示屏**（电脑浏览器或大屏）：

打开 `http://localhost:5000`，等待揭示动画。

转动摇奖机，小球落入出球口，手机/电脑识别颜色，展示屏自动弹出抽卡动画。

```bash
# 终端 1: 启动服务器 + 前端页面
cd server
python app.py
# 浏览器打开 http://localhost:5000

# 终端 2: 启动摄像头检测
cd color_detector
python detect.py

# 转动摇奖机，小球落入出球口即可自动识别并触发揭示动画
```

## 系统架构

```
摇奖机出球 → 摄像头 → detect.py (KNN 分类) → POST /api/detect
                                            ↓
浏览器 ← SSE /events ← app.py (Flask) → SSE 广播
  ↓
揭示动画 (流星 / 光芒爆发 / 礼花 / 卡片)
```

## 自定义

- 奖项配置: 修改 `color_detector/color_mapping.json`
- 动画效果: 修改 `server/static/reveal.html` 中的 CSS animation
- 检测灵敏度: 修改 `detect.py` 中的 `DEBOUNCE_FRAMES` 和 `COOLDOWN_SEC`

## 3D 打印模型推荐

| 模型 | URL |
|------|-----|
| Portable Lottery Machine (推荐) | https://www.printables.com/model/622130 |
| Mini Lottery Bingo Ball Tumbler | https://www.printables.com/model/449905 |
| Bingo Lottery Drum (电动改装) | https://www.printables.com/model/343536 |
| Hand Crank Lottery Drum | https://www.thingiverse.com/thing:4516807 |

推荐 Printables 622130：出球滑道清晰，旋桨拨球可靠。建议在出球口上方加装遮光罩减少环境光干扰。

