# python version used 3.9
import cv2
from cvzone.FaceDetectionModule import FaceDetector

video_capture = cv2.VideoCapture(0)
# set width
video_capture.set(3, 640)
# set height
video_capture.set(4, 480)

detector = FaceDetector(minDetectionCon=0.75)

while True:
    success, cap = video_capture.read()
    if success:
        # it will return image and bounding box means x,y coordinates and height and width
        img, bboxs = detector.findFaces(img=cap, draw=True)
        # check bboxs if not empty
        if bboxs:
            for bbox in bboxs:
                x, y, w, h = bbox['bbox']
                crop_img = img[y: y + h, x: x + w]
                # make particular portion blur
                blur_img = cv2.blur(crop_img, (35, 35))
                # again assign the blur portion in main image
                img[y: y + h, x: x + w] = blur_img

        cv2.imshow('video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        exit()

cap.release()
cv2.destroyAllWindows()
