import cv2
import mediapipe as mp
import time
import argparse

class FaceDetector():
    def __init__(self, minConfidence = 0.5):
        self.minConfidence = minConfidence
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minConfidence)

    def find_face(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxes = []

        if self.results.detections:
            for detection in self.results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                bbox = (int(bbox.xmin * w), int(bbox.ymin * h), 
                        int(bbox.width* w ), int(bbox.height * h))
                bboxes.append([bbox, detection.score])

                if draw:
                    bbox_title = (bbox[0], bbox[1]-20)
                    img = self.fancy_draw(img,  bbox)
                    cv2.putText(img, "{:.3f}".format(float(detection.score[0])), bbox_title, cv2.FONT_HERSHEY_PLAIN,
                                2, (0, 255, 0), 2)

        return img, bboxes

    def fancy_draw(self, img, bbox):
        cv2.rectangle(img, bbox, (255, 255, 255), 2)

        x, y, x1, y1 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]

        l = 40
        t = 10
        edge_color = (0, 0, 255)

        #top left
        cv2.line(img, (x, y), (x+l, y),edge_color, t)
        cv2.line(img, (x, y), (x, y+l),edge_color, t)

        #top right
        cv2.line(img, (x1, y), (x1-l, y),edge_color, t)
        cv2.line(img, (x1, y), (x1, y+l),edge_color, t)

        #bottom left
        cv2.line(img, (x, y1), (x+l, y1),edge_color, t)
        cv2.line(img, (x, y1), (x, y1-l),edge_color, t)

        #bottom right
        cv2.line(img, (x1, y1), (x1-l, y1),edge_color, t)
        cv2.line(img, (x1, y1), (x1, y1-l),edge_color, t)
        
        return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', type=str, default="Videos/Test1.mp4", help='video input location')
    parser.add_argument('-c', type=float, default=0.5, help='Confidence')
    args = parser.parse_args()


    cap = cv2.VideoCapture(args.l)
    ptime = 0
    detector = FaceDetector(args.c)


    while True:
        success, img = cap.read()
        img, bboxes = detector.find_face(img, draw=True)
        ctime = time.time()
        fps = 1 / (ctime-ptime)
        ptime = ctime 
        cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()