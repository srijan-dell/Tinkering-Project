# Import the necessary packages

from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-b", "--min-area", type=int, default=500, help="max buffer size")
args = vars(ap.parse_args())

# If we don't provide recorded video, then we read from the WebCam/Raspberry Pi Camera Module
if args.get("video", None) is None:
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

# When reading from a video file
else:
    vs = cv2.VideoCapture(args["video"])

firstFrame = None

# Looping over the frames of the video
while True:
    # Grab the initial frame
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]
    text = "No Movement"

    # If frame is not grabbed/ End of the video
    if frame is None:
        break

        # Resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Dilate thresh to fill in holes, then find contours on the thresholded image
    faces = faceCascade.detectMultiScale(grayFrame, 1.1, 4)
    # Looping over the contours
    # for c in cnts:
    #     # If the contour is too small, ignore it
    #     if cv2.contourArea(c) < args["min_area"]:
    #         continue
    #
    #     # Compute the bounding box for the contour, draw it on the frame, and update the text
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     text = "Motion Detected"

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        text = "Human Detected"

    # Show text and timestamp on screen
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # Show the frame
    cv2.imshow("Camera Feed", frame)

    # Stop the stream if Q is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Ending the stream

vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
