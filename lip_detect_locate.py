import cv2

def readvedio(videoname):
    capture = cv2.VideoCapture(videoname)
    if capture.isOpened():
        while True:
            ret,img = capture.read() # ret (t/f) read picture or not
            if not ret:
                break
    else:
        print('Fail to read!')