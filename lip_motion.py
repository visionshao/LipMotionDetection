from imutils.video import FileVideoStream
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2

def lip_aspect_ratio(lip):
    A = np.linalg.norm(lip[2] - lip[9])  # 51, 59
    B = np.linalg.norm(lip[4] - lip[7])  # 53, 57
    C = np.linalg.norm(lip[0] - lip[6])  # 49, 55
    lar = (A + B) / (2.0 * C)

    return lar

# use 68 key points face model
shape_predictor = "shape_predictor_68_face_landmarks.dat"
# define lip region
(lipFrom, lipTo) = (49, 68)
# define video path
videoPath = "video/test.mp4"
# define threshold for lip motion
threshold = 0.6

# define the face detector
detector = dlib.get_frontal_face_detector()
# define a shape predictor
predictor = dlib.shape_predictor(shape_predictor)

# read original video
fvs = FileVideoStream(path=videoPath).start()
# define output video
frame_width = 1080
frame_height = 1920
out = cv2.VideoWriter('video/out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

while True:
    # read frames
    frame = fvs.read()
    if frame is not None:
        # preprocess
        frame = imutils.resize(frame, width=640)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect face rect
        rects = detector(frame_gray, 0)

        for rect in rects:
            # find key points inside the face rect
            shape = predictor(frame_gray, rect)
            shape = face_utils.shape_to_np(shape)

            # locate lip region
            lip = shape[lipFrom:lipTo]
            # get lip aspect ratio
            lar = lip_aspect_ratio(lip)

            # get the shape of lip
            lip_shape = cv2.convexHull(lip)
            cv2.drawContours(frame, [lip_shape], -1, (0, 255, 0), 1)
            cv2.putText(frame, "LAR: {:.2f}".format(lar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

            # if open
            if lar > threshold:
                cv2.putText(frame, "Mouth is Open!", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

        # write into output video
        out.write(frame)
        # show the frame
        cv2.imshow("Frame", frame)
        # control imshow lasting time  Explaination: https://juejin.cn/post/6870776834926051342
        key = cv2.waitKey(1) & 0xFF

        # quit
        if key == ord("q"):
            break

# cleanup
cv2.destroyAllWindows()
fvs.stop()
