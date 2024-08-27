from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture("http://192.168.0.102:8080/video")  # For Webcam
cap.set(3, 1920)
cap.set(4, 1080)


# cap = cv2.VideoCapture("../Videos/bikes.mp4")  # For Video


model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

label_colors = {'person': (128,0,128), 'bicycle': (0, 192, 0), 'car': (0, 0, 255), 'motorbike': (255, 255, 0),
             'aeroplane': (255, 0, 255), 'bus': (0, 255, 255), 'train': (255, 255, 255), 'truck': (128, 0, 0),
             'boat': (0, 128, 0), 'traffic light': (0, 0, 128), 'fire hydrant': (128, 128, 0), 'stop sign': (128, 0, 128),
             'parking meter': (0, 128, 128), 'bench': (128, 128, 128), 'bird': (64, 0, 0), 'cat': (192, 0, 0),
             'dog': (64, 128, 0), 'horse': (192, 128, 0), 'sheep': (64, 0, 128), 'cow': (192, 0, 128),
             'elephant': (64, 128, 128), 'bear': (192, 128, 128), 'zebra': (0, 64, 0), 'giraffe': (128, 64, 0),
             'backpack': (0, 255, 0), 'umbrella': (128, 192, 0), 'handbag': (0, 64, 128), 'tie': (128, 64, 128),
             'suitcase': (0, 192, 128), 'frisbee': (128, 192, 128), 'skis': (64, 64, 0), 'snowboard': (192, 64, 0),
             'sports ball': (64, 192, 0), 'kite': (192, 192, 0), 'baseball bat': (64, 64, 128),
             'baseball glove': (192, 64, 128), 'skateboard': (64, 192, 128), 'surfboard': (192, 192, 128),
             'tennis racket': (0, 64, 64), 'bottle': (128, 64, 64), 'wine glass': (0, 192, 64), 'cup': (128, 192, 64),
             'fork': (0, 64, 192),'knife': (128, 64, 192), 'spoon': (0, 192, 192), 'bowl': (128, 192, 192),
                 'banana': (64, 0, 64), 'apple': (192, 0, 64), 'sandwich': (64, 128, 64),
                 'orange': (192, 128, 64), 'broccoli': (64, 0, 192), 'carrot': (192, 0, 192),
                 'hot dog': (64, 128, 192), 'pizza': (192, 128, 192), 'donut': (0, 64, 64),
                 'cake': (128, 64, 64), 'chair': (0, 192, 64),'sofa': (255, 128, 0), 'pottedplant': (128, 255, 0), 'bed': (0, 255, 128), 'diningtable': (0, 128, 255),
                'toilet': (255, 0, 128), 'tvmonitor': (128, 0, 255), 'laptop': (255, 128, 128), 'mouse': (128, 255, 128),
                'remote': (128, 128, 255), 'keyboard': (255, 128, 255), 'cell phone': (128, 255, 255),
                'microwave': (255, 0, 64), 'oven': (64, 255, 0), 'toaster': (0, 64, 255), 'sink': (255, 128, 64),
                'refrigerator': (64, 255, 128), 'book': (128, 64, 255), 'clock': (255, 0, 192), 'vase': (192, 0, 255),
                'scissors': (128, 128, 0), 'teddy bear': (0, 128, 128), 'hair drier': (128, 0, 128),
                'toothbrush': (128, 0, 64)}


prev_frame_time = 0
new_frame_time = 0


while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = classNames[int(box.cls[0])]
            color = label_colors.get(label, (0, 0, 0))  # default color is black
            thickness = 1

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            label = classNames[cls]
            color = label_colors.get(label, (0, 0, 0))  # default color is black
            cvzone.putTextRect(img, f'{label} {conf}', (max(0, x1+8), max(35, y1-12)), scale=0.5, thickness=1,colorR=color, font=cv2.FONT_HERSHEY_SIMPLEX)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
