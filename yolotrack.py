from imutils.video import VideoStream
from imutils.video import FPS
from ultralytics import YOLO
import argparse
import imutils
import time
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="path to input video file")
ap.add_argument(
    "-t", "--tracker", type=str, default="kcf", help="OpenCV object tracker type"
)
args = vars(ap.parse_args())

tracker = cv2.legacy.TrackerKCF.create()
initBB = None

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
fps = None

while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    if frame is None:
        break
    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]

    if initBB is not None:
        (success, box) = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        fps.update()
        fps.stop()
        info = [
            ("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]
        for i, (k, v) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(
                frame,
                text,
                (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        print("[INFO] Starting detection...")
        look = vs.read()
        model = YOLO("best11.pt")
        yolo_outputs = model(look)

        for r in yolo_outputs:
            for box in r.boxes:
                left, top, right, bottom = np.array(
                    box.xyxy.cpu(), dtype=np.int32
                ).squeeze()
                width = right - left
                height = bottom - top
                center = (left + int((right - left) / 2), top + int((bottom - top) / 2))
                label = yolo_outputs[0].names[int(box.cls)]

                initBB = left, top, right, bottom
                tracker.init(look, initBB)

                fps = FPS().start()


cv2.destroyAllWindows()
vs.stop()
