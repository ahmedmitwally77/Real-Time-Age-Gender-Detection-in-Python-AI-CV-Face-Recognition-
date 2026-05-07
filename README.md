# Real-Time Age and Gender Detection

This project implements real-time age and gender detection using OpenCV and pre-trained deep learning models.

## Project Structure

- `main.py`: The main script to run the detection using your webcam.
- `models/`: Contains the pre-trained models and configuration files.
  - `age_deploy.prototxt`: Model architecture for age detection.
  - `age_net.caffemodel`: Pre-trained weights for age detection.
  - `gender_deploy.prototxt`: Model architecture for gender detection.
  - `gender_net.caffemodel`: Pre-trained weights for gender detection.
  - `opencv_face_detector_uint8.pb`: Pre-trained TensorFlow model for face detection.
  - `opencv_face_detector.pbtxt`: Configuration for the face detector.

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)

## How to Run

1. Install the required dependencies:

   ```bash
   pip install opencv-python
   ```

2. Run the detection script:

   ```bash
   python main.py
   ```

3. Press `q` (or close the window) to stop the video stream.

## How it Works

The system uses:

1. A Single Shot Detector (SSD) for face detection.
2. A pre-trained Caffe model to predict gender (Male/Female).
3. A pre-trained Caffe model to predict age group among several categories.
