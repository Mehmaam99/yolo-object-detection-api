# 👁️ YOLO Object Detection API

> Upload any image and detect objects instantly using YOLOv8 + OpenCV. Returns bounding boxes, labels, confidence scores, and an annotated image.

Built by **Syed Muhammad Mehmam** — AI Engineer | [LinkedIn](https://linkedin.com/in/muhammad-mehmam) | [GitHub](https://github.com/Mehmaam99)

---

## 🎯 What This Does

Upload an image → get back every detected object with:
- **Label** (person, car, phone, etc. — 80 COCO classes)
- **Confidence score** (0–100%)
- **Bounding box coordinates** (pixel + normalized)
- **Annotated image** with boxes drawn (base64)

## 🏗️ Architecture

```
Input Image
    │
    ▼
[Preprocessing]  ←── Resize to 640x640, normalize
    │
    ▼
[YOLOv8 CNN Backbone]  ←── Feature extraction
    │
    ▼
[Detection Head + NMS]  ←── Non-Maximum Suppression removes duplicates
    │
    ▼
[Filter by Confidence]  ←── Configurable threshold (default 0.5)
    │
    ▼
[Draw Bounding Boxes]  ←── OpenCV annotation
    │
    ▼
JSON Response + Annotated Image (base64)
```

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| API | FastAPI | Async, fast, auto docs |
| Detection | YOLOv8n (Ultralytics) | State-of-art, real-time speed |
| Image Processing | OpenCV | Industry standard CV library |
| Frontend | Vanilla HTML/JS | No build step needed |

## 🚀 How to Run

```bash
# 1. Clone repo
git clone https://github.com/Mehmaam99/yolo-detection-api
cd yolo-detection-api

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run (YOLOv8 model downloads automatically ~6MB)
uvicorn app.main:app --reload --port 8001

# 4. Open browser
# http://localhost:8001
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Frontend UI |
| POST | `/detect?confidence=0.5` | Detect objects in image |
| GET | `/classes` | List all 80 detectable classes |
| GET | `/health` | System status |

## 💡 Real-World Application

This project is inspired by a **production surveillance system** I built at Xloop Digital Services for a textile factory client — using YOLOv8 for real-time person detection and MeanShift clustering for crowd density analysis and violence detection. This demo shows the detection core of that system.

## 📁 Project Structure

```
project2_yolo_detection/
├── app/
│   └── main.py         # FastAPI + YOLO inference pipeline
├── static/
│   └── index.html      # Frontend with drag-drop upload
├── requirements.txt
└── README.md
```
