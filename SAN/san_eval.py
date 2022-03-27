##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
from __future__ import division

import os, sys, time, random, argparse
from pathlib import Path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # please use Pillow 4.0.0 or it may fail for some images
from os import path as osp
import numbers, numpy as np
import init_path
import torch
import dlib
import cv2
from datetime import datetime
import models
import datasets
from visualization import draw_image_by_points
from san_vision import transforms
from utils import time_string, time_for_file, get_model_infos

os.environ["CUDA_VISIBLE_DEVICES"]='0'
# define the face detector
DETECTOR = dlib.get_frontal_face_detector()
# define lip region
(LIPFROM, LIPTO) = (48, 68)
# define threshold for lip motion
HIGH_THRESHOLD = 0.65
LOW_THRESHOLD = 0.4


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

class SAN_Args():
    def __init__(self, input_type, input, save_path=None):
        self.input_type = input_type
        self.input = input
        self.save_path = save_path
    
    def execute(self):
        (_, tempfilename) = os.path.split(self.input)
        (filename, _) = os.path.splitext(tempfilename)
        # image input
        if self.input_type.upper() == 'IMAGE':
            args = Args(image=self.input, save_path=self.save_path)
            _, img = evaluate(args)
            now = datetime.now()
            filename = filename + '_SAN' + now.strftime("%Y%m%d_%H%M%S")
            cv2.imshow(args.save_path + filename + '.jpg', img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        # video input
        else:
            # read original video
            VC = cv2.VideoCapture(self.input)
            FRAME_RATE = VC.get(cv2.CAP_PROP_FPS)
            # define output video
            FRAME_WIDTH = int(VC.get(cv2.CAP_PROP_FRAME_WIDTH))
            FRAME_HEIGHT = int(VC.get(cv2.CAP_PROP_FRAME_HEIGHT))
            now = datetime.now()
            filename = filename + '_SAN' + now.strftime("%Y%m%d_%H%M%S")
            out = cv2.VideoWriter(args.save_path + filename + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE, (FRAME_WIDTH, FRAME_HEIGHT))
            f=open(args.save_path + "LARs_SAN.txt","w")
            # process video
            while (VC.isOpened()):
                # read frames
                rval, frame = VC.read()
                if rval:
                    cv2.imwrite('frame.jpg', frame)
                    args = Args(image='frame.jpg')
                    lar, frame = evaluate(args)
                    # record lar
                    f.write(str(lar)+'\n')
                    # write into output video
                    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation = cv2.INTER_AREA)
                    out.write(frame)
                    # # show the frame
                    # cv2.imshow("Frame", frame)
                    # # control imshow lasting time
                    # key = cv2.waitKey(1) & 0xFF
                    # # quit
                    # if key == ord("q"):
                    #     break
                else:
                    break
            # cleanup
            cv2.destroyAllWindows()
            os.remove('frame.jpg')
            VC.release()
            out.release()
            f.close()



class Args():
    def __init__(self, image, face=None, save_path=None, cpu=False):
        self.image = image
        self.model = 'SAN/snapshots/SAN_300W_GTB_itn_cpm_3_50_sigma4_128x128x8/checkpoint_49.pth.tar'
        self.face = face
        self.locate_face()
        self.save_path = save_path
        self.cpu = cpu
    
    def locate_face(self):
        if self.face == None:
            img = cv2.imread(self.image)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rects = DETECTOR(img_rgb, 0)
            if len(rects) == 0:
                print('Fail to find a face!')
            else:
                rect = rects[0]
                left = rect.tl_corner().x
                top = rect.tl_corner().y
                right = rect.br_corner().x
                bottom = rect.br_corner().y
                self.face = [left, top, right, bottom]

def evaluate(args):
    if not args.cpu:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        torch.backends.cudnn.enabled   = True
        torch.backends.cudnn.benchmark = True

    print ('The image is {:}'.format(args.image))
    print ('The model is {:}'.format(args.model))
    snapshot = Path(args.model)
    assert snapshot.exists(), 'The model path {:} does not exist'
    print ('The face bounding box is {:}'.format(args.face))
    assert len(args.face) == 4, 'Invalid face input : {:}'.format(args.face)
    
    if args.cpu: snapshot = torch.load(snapshot, map_location='cpu')
    else       : snapshot = torch.load(snapshot)

    mean_fill   = tuple( [int(x*255) for x in [0.5, 0.5, 0.5] ] )
    normalize   = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
    param = snapshot['args']
    eval_transform  = transforms.Compose([transforms.PreCrop(param.pre_crop_expand), transforms.TrainScale2WH((param.crop_width, param.crop_height)),  transforms.ToTensor(), normalize])

    net = models.__dict__[param.arch](param.modelconfig, None)

    if not args.cpu: net = net.cuda()
    weights = models.remove_module_dict(snapshot['state_dict'])
    net.load_state_dict(weights)

    dataset = datasets.GeneralDataset(eval_transform, param.sigma, param.downsample, param.heatmap_type, param.dataset_name)
    dataset.reset(param.num_pts)

    print ('[{:}] prepare the input data'.format(time_string()))
    [image, _, _, _, _, _, cropped_size], meta = dataset.prepare_input(args.image, args.face)
    print ('[{:}] prepare the input data done'.format(time_string()))
    print ('Net : \n{:}'.format(net))
    # network forward
    with torch.no_grad():
        if args.cpu: inputs = image.unsqueeze(0)
        else       : inputs = image.unsqueeze(0).cuda()
        batch_heatmaps, batch_locs, batch_scos, _ = net(inputs)
        #print ('input-shape : {:}'.format(inputs.shape))
        flops, params = get_model_infos(net, inputs.shape, None)
        print ('\nIN-shape : {:}, FLOPs : {:} MB, Params : {:}.'.format(list(inputs.shape), flops, params))
        flops, params = get_model_infos(net, None, inputs)
        print ('\nIN-shape : {:}, FLOPs : {:} MB, Params : {:}.'.format(list(inputs.shape), flops, params))
    print ('[{:}] the network forward done'.format(time_string()))

    # obtain the locations on the image in the orignial size
    cpu = torch.device('cpu')
    np_batch_locs, np_batch_scos, cropped_size = batch_locs.to(cpu).numpy(), batch_scos.to(cpu).numpy(), cropped_size.numpy()
    locations, scores = np_batch_locs[0,:-1,:], np.expand_dims(np_batch_scos[0,:-1], -1)

    scale_h, scale_w = cropped_size[0] * 1. / inputs.size(-2) , cropped_size[1] * 1. / inputs.size(-1)

    locations[:, 0], locations[:, 1] = locations[:, 0] * scale_w + cropped_size[2], locations[:, 1] * scale_h + cropped_size[3]
    prediction = np.concatenate((locations, scores), axis=1).transpose(1,0)
    shape = []
    for i in range(param.num_pts):
        point = prediction[:, i]
        shape.append([round(point[0]), round(point[1])])
        print ('The coordinate of {:02d}/{:02d}-th points : ({:.1f}, {:.1f}), score = {:.3f}'.format(i, param.num_pts, float(point[0]), float(point[1]), float(point[2])))
    shape = np.array(shape)
    # locate lip region
    lip = shape[LIPFROM:LIPTO]
    # get lip aspect ratio
    lar = lip_aspect_ratio(lip)
    # image = draw_image_by_points(args.image, prediction, 1, (255,0,0), False, False)
    img = cv2.imread(args.image)
    lip_shape = cv2.convexHull(lip)
    cv2.drawContours(img, [lip_shape], -1, (0, 255, 0), 1)
    if lar > HIGH_THRESHOLD or lar < LOW_THRESHOLD:
        cv2.putText(img, "Lip Motion Detected!", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)                
    if args.save_path:
        cv2.imwrite(args.save_path + os.path.basename(args.image), img)
        # print ('save image with landmarks into {:}'.format(args.save_path + os.path.basename(args.input)))
    print(lar)
    print('finish san evaluation on a single image : {:}'.format(args.image))

    return lar, img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a single image by the trained model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image',            type=str,   help='The evaluation image path.')
    parser.add_argument('--model',            type=str,   help='The snapshot to the saved detector.')
    parser.add_argument('--face',  nargs='+', type=float, help='The coordinate [x1,y1,x2,y2] of a face')
    parser.add_argument('--save_path',        type=str,   help='The path to save the visualization results')
    parser.add_argument('--cpu',     action='store_true', help='Use CPU or not.')
    args = parser.parse_args()
    evaluate(args)
