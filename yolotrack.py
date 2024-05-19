from imutils.video import VideoStream
from imutils.video import FPS
from ultralytics import YOLO
import argparse
import imutils
import time
import cv2
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())


tracker = cv2.legacy.TrackerKCF.create()

initBB = None

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)
fps = None

# loop over frames from the video stream
while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame
	# check to see if we have reached the end of the stream
	if frame is None:
		break
	# resize the frame (so we can process it faster) and grab the
	# frame dimensions
	frame = imutils.resize(frame, width=500)
	(H, W) = frame.shape[:2]
	
# check to see if we are currently tracking an object
	if initBB is not None:
		# grab the new bounding box coordinates of the object
		(success, box) = tracker.update(frame)
		# check to see if the tracking was a success
		if success:
			(x, y, w, h) = [int(v) for v in box]
			cv2.rectangle(frame, (x, y), (x + w, y + h),
				(0, 255, 0), 2)
		# update the FPS counter
		fps.update()
		fps.stop()
		# initialize the set of information we'll be displaying on
		# the frame
		info = [
			("Tracker", args["tracker"]),
			("Success", "Yes" if success else "No"),
			("FPS", "{:.2f}".format(fps.fps())),
		]
		# loop over the info tuples and draw them on our frame
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
			
# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the 's' key is selected, we are going to "select" a bounding
	# box to track


	if key == ord("s"):
		
		print("[INFO] Starting detection...")
		look=vs.read()
		model=YOLO("best11.pt")
		yolo_outputs = model(look)

		for r in yolo_outputs:
			boxes= r.boxes
			for box in boxes:
				x1, y1, x2, y2 = box.xyxy[0]
				rects = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.int32)
				bbox=rects[x1,y1,x2,y2]
				tracker.init(frame, (x1, y1, (x2 - x1), (y2 - y1)))
				print(x1,y1,x2,y2)
				fps = FPS().start()
                
    			
	
		

			


		
			
		

