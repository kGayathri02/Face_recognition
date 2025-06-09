# 🧠 Face Recognition with Pose Variation: Multi-Angle Matching in Real Time

This project demonstrates how to build a robust face recognition system that handles **pose variations** — like side profiles — using yaw-augmented face embeddings. Even low-quality images (like Aadhaar-style IDs) can be recognized in real-time webcam feeds.

<div align="center">
  <img src="workflow_image.png" alt="Workflow Diagram" width="600"/>
</div>

---

## 🚀 Features

- ✅ Real-time face recognition using webcam
- ✅ Handles pose variations with simulated yaw augmentation
- ✅ No deep learning experience required
- ✅ Works on CPU (no GPU needed)
- ✅ Uses [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l` model for high-quality embeddings

---

## 🧱 Project Structure
.
├── main_recognizer.py # Real-time recognition logic
├── register_aadhaar_face.py # Embedding generation from yaw-augmented images
├── augmented_faces/ # Stores yaw-simulated images
├── aadhaar_embeddings.pkl # Saved facial embeddings
├── gayu_a_ch.jpeg # Original Aadhaar-style input image
├── workflow_image.png # Visual workflow
└── README.md


---

## 📸 Workflow Overview

1. **Register Face Image** (Aadhaar-style)
2. **Simulate Yaw Angles** (Left/Right head tilt)
3. **Extract Embeddings** from each variation
4. **Save Embeddings** for reference
5. **Real-Time Webcam Feed** captures live faces
6. **Compare Live Embedding** with stored ones
7. **Display Name or "Unknown"** based on cosine similarity

---

## 🛠️ How to Run

### 1. 🔧 Install Dependencies
pip install insightface opencv-python numpy
Ensure you have onnxruntime or a compatible CPU backend (InsightFace will handle most setups automatically).

2. 📁 Step 1: Register Aadhaar Image

python register_aadhaar_face.py

This will:

    Simulate multiple angles of your Aadhaar image

    Generate embeddings for each pose

    Save them in aadhaar_embeddings.pkl

3. 🎥 Step 2: Run Live Recognition

python main_recognizer.py

    Opens webcam

    Detects and identifies matching face using cosine similarity

    Tolerant to head pose changes due to pre-augmented angles

⚙️ Configurable Settings

In both scripts, you can modify:

SIMULATED_YAW_ANGLES = [-45, -30, -15, 0, 15, 30, 45]
THRESHOLD = 0.35  # Lower for more tolerance, higher for stricter matching

🔬 Model Used: buffalo_l

    Pretrained InsightFace model

    Balanced for accuracy and speed

    Robust to lighting, pose, and expression variations

    No training required – plug and play!

🧠 Why This Works

Instead of training complex models, we pre-augment the face with simulated yaw rotations, making our system recognize side views. Combined with strong embeddings from buffalo_l, it performs well even on low-resolution inputs.
