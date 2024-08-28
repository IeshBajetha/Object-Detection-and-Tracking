# Real-Time Object Detection with OpenCV

## Project Overview

This project implements real-time object detection using a machine learning (ML) model and OpenCV. The code captures live video from a mobile phone or video file, processes each frame using the ML model to detect objects, and displays the results with bounding boxes and labels. This project can be used for applications such as surveillance, autonomous driving, and real-time video analysis.

## Features

- **Real-Time Object Detection**: Utilizes a machine learning model for detecting multiple objects in real time.
- **Live Video Streaming**: Supports video input from a mobile device or a video file.
- **Customizable Output**: Easily modify the code to detect specific object classes and adjust display settings.
- **FPS Display**: Shows the frames per second (FPS) to help monitor performance.

## How It Works

1. **Video Capture**: The script captures a video stream from a specified source, such as a mobile device or a video file.
2. **Object Detection**: Each frame of the video is processed using the machine learning model to detect objects.
3. **Display Output**: The detected objects are highlighted with bounding boxes and labels, and the video is displayed on the screen.

## Requirements

To run this project, you need:

- Python
- OpenCV (`opencv-python`)
- Machine Learning Model (e.g., a pre-trained model file)
  

## Getting Started

1. Clone the repository.
    ```bash
    git clone https://github.com/IeshBajetha/Object-Detection-and-Tracking
    ```
2. Install the required dependencies.
    ```bash
    pip install -r requirements.txt
    ```
3. Run the script and view the live object detection.
    ```bash
    python detect.py
    ```

## Customization

- Modify the `classNames` list to detect specific objects.
- Adjust the display settings such as resolution or label colors in the script.

## License

This project is licensed under the MIT License.
