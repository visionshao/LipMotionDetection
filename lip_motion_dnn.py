import time
import cv2
from openvino.inference_engine import IENetwork, IECore

# define video path
videoPath = "video/G0001/V11_20150707142849_52654320.mp4"

# import model
model_bin = "model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
config_text = "model/deploy.prototxt"

xmlfile = "model/facial-landmarks-35-adas-0002.xml"
binfile = "model/facial-landmarks-35-adas-0002.bin"

# load caffe model
net = cv2.dnn.readNetFromCaffe(config_text, model_bin)
marknet = IENetwork(model = xmlfile, weights = binfile) 

# set back-end
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# read original video
theVedio = cv2.VideoCapture(videoPath)
start = time.time()

while True:
    # read frames
    success, frame = theVedio.read()
    if success:
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

                face = frame[int(top):int(bottom), int(left):int(right)]
                faceblobImage = cv2.dnn.blobFromImage(face, 1.0, (60, 60), Scalar(), False, False)




        cv2.imshow("Frame", frame)
        # control imshow lasting time  Explaination: https://juejin.cn/post/6870776834926051342
        key = cv2.waitKey(1) & 0xFF

        # quit
        if key == ord("q"):
            break

    else: 
        break

# cleanup
cv2.destroyAllWindows()
theVedio.release()
print(time.time()-start)
