# Simple face detector using Mediapipe

A simple face detector class which can be used in your own projects, or run by itself. 

## Class structure
```sh
Class name: FaceDetector
Arguments: minConfidence (default = 0.5)
Method: find_face
Arguments: img, draw (default = True)
```

## How to run
As a module:
```sh
import Face_Detection_module.py
detector = FaceDetector(confidence)
img, bboxes = detector.find_face(img, draw=True)
```

Directly:
```sh
Face_Detection_module.py -l [video_input] -c [confidence]
```

### Author: Amogh Dhaliwal


