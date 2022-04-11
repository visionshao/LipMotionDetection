from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import os
from datetime import datetime
import gdown


# use 68 key points face model
model_path = 'model/shape_predictor_68_face_landmarks.dat'
if os.path.exists('Dlib/'):
    model_path = 'Dlib/' + model_path
if not os.path.exists(model_path):
    # You canalso click this link to download 'https://drive.google.com/file/d/1AwHKa2-QpcqkFgqTbOLoRoNBDVXfTZ05/view?usp=sharing'
    url = 'https://drive.google.com/uc?id=1AwHKa2-QpcqkFgqTbOLoRoNBDVXfTZ05'
    gdown.download(url, model_path, quiet=False)
SHAPE_PREDICTOR = model_path
# define lip region
(LIPFROM, LIPTO) = (48, 68)
# define threshold for lip motion
HIGH_THRESHOLD = 0.49
LOW_THRESHOLD = 0.4

# define the face detector
DETECTOR = dlib.get_frontal_face_detector()
# define a shape predictor
PREDICTOR = dlib.shape_predictor(SHAPE_PREDICTOR)


# arguments
class Dlib_Args():
    def __init__(self, input_type, input, save_path=None):
        self.input_type = input_type.upper()
        self.input = input
        self.save_path = save_path


# calculate lip aspect ratio
def lip_aspect_ratio(lip):

    # left top to left bottom
    A = np.linalg.norm(lip[2] - lip[9])  # 51, 59
    # right top to right bottom
    B = np.linalg.norm(lip[4] - lip[7])  # 53, 57
    # leftest to rightest
    C = np.linalg.norm(lip[0] - lip[6])  # 49, 55
    lar = (A + B) / (2.0 * C)

    return lar


# process one image
def process_frame(frame):

    # preprocess
    frame = imutils.resize(frame, width=640)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect face rect
    rects = DETECTOR(frame_gray, 0)
    lar = 0.45

    if len(rects):
        rect = rects[0]
        # find key points inside the face rect
        shape = PREDICTOR(frame_gray, rect)
        shape = face_utils.shape_to_np(shape)

        # locate lip region
        lip = shape[LIPFROM:LIPTO]
        # get lip aspect ratio
        lar = lip_aspect_ratio(lip)

        # get the shape of lip
        lip_shape = cv2.convexHull(lip)
        # print(lar)
        cv2.drawContours(frame, [lip_shape], -1, (0, 255, 0), 1)
        # left = rect.tl_corner().x
        # top = rect.tl_corner().y
        # right = rect.br_corner().x
        # bottom = rect.br_corner().y
        # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  

        cv2.putText(frame, "LAR: {:.2f}".format(lar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        # if open
        if lar > HIGH_THRESHOLD or lar < LOW_THRESHOLD:
            cv2.putText(frame, "Lip Motion Detected!", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)                
    else:
        print('No face found!')

    return lar, frame


# process input
def execute(args):
    (_, tempfilename) = os.path.split(args.input)
    (filename, _) = os.path.splitext(tempfilename)
    # image input
    if args.input_type.upper() == 'IMAGE':
        img = cv2.imread(args.input)
        _, img = process_frame(img)
        now = datetime.now()
        filename = filename + now.strftime("_%Y%m%d_%H%M%S_") + 'Dlib'
        cv2.imwrite(args.save_path + filename + '.jpg', img)
        cv2.imshow("Image", img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    # video input
    elif args.input_type.upper() == 'VIDEO':
        # read original video
        VC = cv2.VideoCapture(args.input)
        FRAME_RATE = VC.get(cv2.CAP_PROP_FPS)
        # define output video
        FRAME_WIDTH = int(VC.get(cv2.CAP_PROP_FRAME_WIDTH))
        FRAME_HEIGHT = int(VC.get(cv2.CAP_PROP_FRAME_HEIGHT))
        now = datetime.now()
        filename = filename + now.strftime("_%Y%m%d_%H%M%S_") + 'Dlib'
        out = cv2.VideoWriter(args.save_path + filename + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE, (FRAME_WIDTH, FRAME_HEIGHT))
        f = open(args.save_path + filename + "_LARs.txt","w")
        # process video
        while (VC.isOpened()):
            # read frames
            rval, frame = VC.read()
            if rval:
                lar, frame = process_frame(frame)
                # record lar
                f.write(str(lar)+'\n')
                # write into output video
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation = cv2.INTER_AREA)
                out.write(frame)
                # show the frame
                cv2.imshow("Frame", frame)
                # control imshow lasting time
                key = cv2.waitKey(1) & 0xFF
                # quit
                if key == ord("q"):
                    break
            else: 
                break

        # cleanup
        cv2.destroyAllWindows()
        VC.release()
        out.release()
        f.close()
    # camera input
    else:
        # activate the camera
        VC = cv2.VideoCapture(0)
        FRAME_RATE = 30
        FRAME_WIDTH = 640
        FRAME_HEIGHT = 380
        now = datetime.now()
        filename = 'Camera' + now.strftime("_%Y%m%d_%H%M%S_") + 'Dlib'
        out = cv2.VideoWriter(args.save_path + filename + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE, (FRAME_WIDTH, FRAME_HEIGHT))
        # process video
        while (VC.isOpened()):
            # read frames
            rval, frame = VC.read()
            if rval:
                _, frame = process_frame(frame)
                # write into output video
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation = cv2.INTER_AREA)
                out.write(frame)
                # show the frame
                cv2.imshow("Frame", frame)
                # control imshow lasting time
                key = cv2.waitKey(1) & 0xFF
                # quit
                if key == ord("q"):
                    break
            else: 
                break

        # cleanup
        cv2.destroyAllWindows()
        VC.release()
        out.release()
