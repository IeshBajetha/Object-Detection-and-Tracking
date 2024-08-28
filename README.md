# Object-Detection-and-Tracking


YOLO Object Detection with OpenCV
Project Overview
This project implements real-time object detection using the YOLOv8 model and OpenCV. The code captures live video from a mobile phone or video file, processes each frame using YOLOv8 to detect objects, and displays the results with bounding boxes and labels. This project can be used for applications like surveillance, autonomous driving, and real-time video analysis.

Features
Real-Time Object Detection: Utilizes the YOLOv8 model for detecting multiple objects in real time.
Live Video Streaming: Supports video input from a mobile device or a video file.
Customizable Output: Easily modify the code to detect specific object classes and adjust display settings.
FPS Display: Shows the frames per second (FPS) to help monitor performance.
How It Works
Video Capture: The script captures a video stream from a specified source, such as a mobile device or a video file.
Object Detection: Each frame of the video is processed using the YOLOv8 model to detect objects.
Display Output: The detected objects are highlighted with bounding boxes and labels, and the video is displayed on the screen.
Requirements
To run this project, you need:

Python
OpenCV (opencv-python)
UltraLytics YOLO (ultralytics)
CVZone (cvzone)
Getting Started
Clone the repository.
Install the required dependencies.
Run the script and view the live object detection.
Customization
Modify the classNames list to detect specific objects.
Adjust the display settings such as resolution or label colors in the script.
License
This project is licensed under the MIT License.
