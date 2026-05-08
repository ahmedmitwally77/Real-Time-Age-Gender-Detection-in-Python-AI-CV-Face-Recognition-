# AI Surveillance Pro - Advanced Real-Time Face Analytics

An advanced, high-performance computer vision system for real-time face detection, recognition, and comprehensive biometric analysis (Age, Gender, and Emotion) using state-of-the-art AI models.

## 🚀 Key Features

- **Real-time Biometric Analysis**: Instantly detects and analyzes Age, Gender, and Dominant Emotions.
- **Face Recognition System**: Identifies known individuals by comparing live feeds against images in the `known_faces` directory.
- **Professional HUD Interface**: Features a futuristic "Heads-Up Display" (HUD) showing system status, detection logs, FPS counter, and interactive panels.
- **Multi-Threaded Processing**: Offloads heavy AI computations to background threads to maintain a high capture frame rate (up to 90 FPS).
- **Smart Face Filtering**: Specialized logic to filter out small or distorted face detections, ensuring high accuracy.
- **Instant Surveillance Tools**: Capture high-quality screenshots of detections with a single keystroke.

## 🛠 System Architecture

The project leverages a modern AI stack for maximum reliability and speed:

- **DeepFace**: Core framework for deep learning-based face analysis.
- **RetinaFace/OpenCV Integration**: High-precision face detection backend.
- **Caffe/TensorFlow Wrappers**: Optimized model execution for real-time performance.
- **Threading Engine**: Ensures the UI remains responsive while the AI engine processes frames at 480p/270p for optimized throughput.

## 📂 Project Structure

- `main.py`: The entry point script containing the GUI logic and AI pipeline.
- `models/`: Centralized storage for pre-trained weights and AI configuration files.
- `known_faces/`: Database for face recognition. **Add photos here (e.g., `ahmed.jpg`) to recognize people.**
- `screenshots/`: Automatic storage for any images captured during the session.
- `README.md`: System documentation.

## 📋 Requirements

- **Python**: 3.9 or newer recommended.
- **Hardware**: A standard USB/Integrated Webcam.
- **OS**: Windows, Linux, or macOS.

## ⚙️ Installation

Install the necessary dependencies using pip:

```bash
pip install opencv-python deepface tf-keras retina-face tensorflow
```

## 🎮 How to Use

1.  **(Optional) Registration**: Place clear photos of people you want to recognize in the `known_faces` folder.
2.  **Launch**: Run the main script:
    ```bash
    python main.py
    ```
3.  **Interaction**:
    - **View**: Observe the live telemetry on the HUD.
    - **Capture**: Press **'S'** to save a timestamped screenshot of a detection.
    - **Terminate**: Press **'Q'** to safely close the camera and exit.

## 🔍 Technical Details

- **Input Optimization**: The system captures at high resolution (`1380x780`) but analyzes a downscaled version (`480x270`) to reduce latency without sacrificing detection quality.
- **Emoji Overlay**: Automatically maps detected emotions (Happy, Sad, Angry, etc.) to visual identifiers.
- **Duplicate Prevention**: Implements center-distance algorithms to prevent multiple detections of the same face across analysis cycles.

---

Developed by [Ahmed Mitwally](https://github.com/ahmedmitwally77)
