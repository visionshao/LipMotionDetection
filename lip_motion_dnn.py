from imutils.video import FileVideoStream
import time
import cv2

# define video path
videoPath = "video/test.mp4"

# import model
model_bin = "model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
config_text = "model/deploy.prototxt"

# load tensorflow model
net = cv2.dnn.readNetFromCaffe(config_text, model_bin)

# set back-end
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# read original video
fvs = FileVideoStream(path=videoPath).start()
# define output video
frame_width = 1080
frame_height = 1920
out = cv2.VideoWriter('video/out_dnn.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

start = time.time()

while True:
    # read frames
    frame = fvs.read()
    if frame is not None:
        height, width = frame.shape[:2]
        # preprocess
        blobImage = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        # input
        net.setInput(blobImage)
        output = net.forward()

        # Put efficiency information.
        t, _ = net.getPerfProfile()
        fps = 1000 / (t * 1000.0 / cv2.getTickFrequency())
        label = 'FPS: %.2f' % fps
        cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        # write rectangle
        for detection in output[0,0,:,:]:
            score = float(detection[2])
            objIndex = int(detection[1])
            if score > 0.5:
                left = detection[3] * width
                top = detection[4] * height
                right = detection[5] * width
                bottom = detection[6] * height

                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)
                cv2.putText(frame, "score:%.2f"%score, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        out.wirte(frame)

    else: 
        break

# # cleanup
cv2.destroyAllWindows()
fvs.stop()
print(time.time()-start)
