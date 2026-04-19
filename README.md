# 👁️ VocalEyes v2.0

**AI-Powered Assistive Vision System for the Visually Impaired**

VocalEyes is an intelligent assistive technology that empowers visually impaired users to navigate their environment safely. Using real-time object detection, OCR text recognition, and natural voice alerts, VocalEyes provides contextual awareness through both a web application and desktop interface.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-green.svg)](https://github.com/ultralytics/ultralytics)
[![Flask](https://img.shields.io/badge/Flask-3.1%2B-lightgrey.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌟 Features

### Core Capabilities
- **Real-Time Object Detection** — Identifies objects using YOLOv8 with custom-trained models
- **Distance & Direction Estimation** — Calculates proximity and spatial positioning
- **Priority-Based Alerts** — Categorizes objects by danger level (HIGH/MEDIUM/LOW)
- **OCR Text Recognition** — Reads signs, labels, and text in the environment
- **Natural Voice Alerts** — Speaks contextual warnings and information
- **Dual Interface** — Web app with live camera feed + standalone desktop application

### Web Application
- 📹 Live camera feed with real-time detection overlay
- 🖼️ Image upload and batch analysis
- 🔊 Browser-based Text-to-Speech with adjustable voice
- ⚠️ Visual danger indicators with audio alerts
- 📊 Session statistics and detection analytics
- 📈 Top objects tracker with interactive charts
- ⚙️ Adjustable confidence threshold and detection speed
- 🎤 Voice command support ("start", "stop", "analyze")
- ⌨️ Keyboard shortcuts (S=start, X=stop, V=toggle voice)
- 📸 Snapshot capture and export
- 🌓 High contrast accessibility mode
- 📝 Exportable event logs with timestamps

### Desktop Application
- 🎥 Webcam live detection mode
- 🖼️ Single image analysis with GUI file picker
- 🔊 Desktop TTS (pyttsx3) with adjustable speech rate
- 📋 Console output for all detections

---

## 📁 Project Structure

```
VocalEyes/
├── app.py                     # Flask web server
├── main.py                    # Desktop application entry point
├── context_logic.py           # Priority classification system
├── sentence_generator.py      # Natural language generation
├── ocr_module.py              # OCR text reading module
├── speech.py                  # PowerShell TTS wrapper
├── yolo_live.py               # Basic YOLO webcam demo
├── yolo_to_vocaleyes.py       # Integrated detection script
├── simulator.py               # Testing simulator
├── requirements.txt           # Python dependencies
├── custom.yaml                # Custom YOLO training config
│
├── templates/
│   └── index.html             # Web frontend UI
│
├── static/                    # Web assets (CSS, JS)
│
├── dataset/                   # Training data (not in repo)
│   ├── train/images/
│   └── valid/images/
│
└── runs/detect/train26/       # Model outputs (not in repo)
    └── weights/
        └── best.pt            # Custom trained model
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam (for live detection)
- Tesseract OCR installed

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/VocalEyes.git
cd VocalEyes
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Install Tesseract OCR

**Windows:**
```
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
Add to PATH: C:\Program Files\Tesseract-OCR
```

**Linux/Ubuntu:**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### 4️⃣ Download YOLO Models

Place the following pre-trained models in the project root:
- `yolov8n.pt` — Nano model (fastest)
- `yolov8s.pt` — Small model (balanced)

**Download from:** https://github.com/ultralytics/assets/releases

### 5️⃣ Add Custom Model (Optional)

If you have a custom-trained model:
```
runs/detect/train26/weights/best.pt
```

If not, VocalEyes will use the base YOLOv8 model automatically.

---

## 🎯 Usage

### Web Application (Recommended)

```bash
python app.py
```

Then open your browser to:
```
http://localhost:5000
```

**Features:**
- Click **"Start Detection"** to begin live camera analysis
- Use **"Upload Image"** to analyze a single photo
- Adjust settings in the **Settings Panel**
- Enable **High Contrast Mode** for better visibility
- Export detection logs as `.txt` files

### Desktop Application

```bash
python main.py
```

**Options:**
- `1` → Webcam live detection with voice alerts
- `2` → Upload and analyze a single image

**Controls:**
- Press `Q` to quit webcam mode
- Voice alerts play automatically

---

## 🧠 How It Works

### Detection Pipeline

```
Camera Feed → YOLOv8 Detection → Distance Estimation → Priority Classification
     ↓
OCR Text Reading (when safe) → Natural Language Generation → Voice Alert
```

### Priority System

| Priority | Objects | Distance Threshold | Action |
|----------|---------|-------------------|--------|
| **HIGH** | car, stairs, truck, bus, motorcycle, bicycle, fire hydrant, stop sign, traffic light | Any | Immediate alert + beep |
| **MEDIUM** | person, doors, dog, cat, bird | very near / near | Alert with warning |
| **LOW** | chair, table, couch, bottle, cup, text-sign, book, laptop | Any | Log only (no alert) |

### Distance Estimation

Based on bounding box width relative to frame:
- **Very Near:** >55% of frame width
- **Near:** 30-55% of frame width
- **Medium:** 15-30% of frame width
- **Far:** <15% of frame width

### Direction Detection

Based on object center position:
- **Left:** <33% from left edge
- **Ahead/Center:** 33-66% from left edge
- **Right:** >66% from left edge

---

## 🔧 API Documentation

### REST Endpoints

| Endpoint | Method | Description | Request Body |
|----------|--------|-------------|--------------|
| `/` | GET | Serve web frontend | — |
| `/health` | GET | Health check | — |
| `/upload` | POST | Analyze uploaded image | `multipart/form-data` with image file |
| `/detect` | POST | Detect from base64 frame | `{"image": "base64_string"}` |
| `/stats` | GET | Get session statistics | — |
| `/reset_stats` | POST | Reset session counters | — |

### Response Format

```json
{
  "detections": [
    {
      "label": "car",
      "distance": "near",
      "direction": "right",
      "priority": "HIGH",
      "message": "Warning! Car nearby on your right.",
      "danger": true
    }
  ],
  "ocr_text": "STOP",
  "timestamp": "2026-04-19T10:30:00"
}
```

---

## 🏋️ Training Custom Models

### 1. Prepare Dataset

Organize your data:
```
dataset/
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

### 2. Configure `custom.yaml`

```yaml
path: /path/to/dataset
train: train/images
val: valid/images

nc: 3  # number of classes

names:
  0: stairs
  1: doors
  2: text-sign
```

### 3. Train

```bash
yolo detect train model=yolov8s.pt data=custom.yaml epochs=50 imgsz=512 batch=16
```

### 4. Export Best Weights

```bash
cp runs/detect/trainN/weights/best.pt runs/detect/train26/weights/best.pt
```

---

## ⚙️ Configuration

### Adjustable Parameters

**In `main.py` and `app.py`:**
```python
CONF = 0.45  # Confidence threshold (0.0 - 1.0)
```

**In `sentence_generator.py`:**
```python
COOLDOWN = {
    "HIGH": 2,     # seconds between HIGH alerts
    "MEDIUM": 5,   # seconds between MEDIUM alerts
    "LOW": 8,      # seconds between LOW alerts
}
```

**In `ocr_module.py`:**
```python
_COOLDOWN = 4.0   # seconds between OCR reads
_MIN_LEN = 4      # minimum text length to speak
```

---

## 🐛 Troubleshooting

### Common Issues

**❌ "Cannot access webcam"**
- Check camera permissions in system settings
- Ensure no other application is using the camera
- Try a different camera index: `cv2.VideoCapture(1)`

**❌ "Tesseract not found"**
- Verify Tesseract is installed: `tesseract --version`
- Add Tesseract to system PATH
- On Windows, set: `pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'`

**❌ "Model file not found"**
- Ensure YOLO models are in the project root
- Download from: https://github.com/ultralytics/assets/releases
- Check file names match exactly: `yolov8n.pt`, `yolov8s.pt`

**❌ Low FPS / Slow detection**
- Use a smaller model: `yolov8n.pt` instead of `yolov8s.pt`
- Reduce confidence threshold
- Lower camera resolution in `main.py`
- Use GPU acceleration if available

**❌ TTS not working on Linux**
- Install espeak: `sudo apt install espeak`
- Or install festival: `sudo apt install festival`

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/amazing-feature`
3. **Commit your changes:** `git commit -m 'Add amazing feature'`
4. **Push to the branch:** `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Areas for Contribution
- 🎯 Additional priority objects
- 🌍 Multi-language support
- 📱 Mobile app development
- 🧪 Unit tests and integration tests
- 📖 Documentation improvements
- 🎨 UI/UX enhancements
- ♿ Additional accessibility features

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Ultralytics YOLOv8** — Object detection framework
- **Tesseract OCR** — Text recognition engine
- **Flask** — Web framework
- **OpenCV** — Computer vision library
- **Roboflow** — Dataset management and hosting

### Dataset
- **VocalEyes_Final** — Custom dataset for stairs, doors, and text-sign detection
- License: CC BY 4.0

---

## 📞 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/VocalEyes/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/VocalEyes/discussions)
- **Email:** your.email@example.com

---

## 🗺️ Roadmap

- [ ] Mobile app (iOS/Android)
- [ ] Cloud-based API service
- [ ] Multi-language TTS support
- [ ] Improved depth estimation using stereo vision
- [ ] Integration with smart glasses
- [ ] Offline mode optimization
- [ ] GPS navigation integration
- [ ] Community-contributed object models
- [ ] Real-time obstacle avoidance guidance

---

## ⭐ Show Your Support

If VocalEyes helps you or someone you know, please consider:
- ⭐ Starring this repository
- 🐛 Reporting bugs
- 💡 Suggesting new features
- 📢 Sharing with others who might benefit

---

<div align="center">

**Made with ❤️ for accessibility**

[Report Bug](https://github.com/yourusername/VocalEyes/issues) · [Request Feature](https://github.com/yourusername/VocalEyes/issues) · [Documentation](https://github.com/yourusername/VocalEyes/wiki)

</div>