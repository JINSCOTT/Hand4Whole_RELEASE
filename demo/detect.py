import cv2
import numpy as np
import onnxruntime
import sys
import os
import os.path as osp
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
import datetime

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import load_img, process_bbox, generate_patch_image
from utils.human_models import smpl_x
from utils.vis import render_mesh, save_obj
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args

class Tracks:
    def __init__(self, name, x,y,xx,yy,label,prob):
        self.batch = name
        self.x1 = int(x)
        self.y1 = int(y)
        self.x2 = int(xx)
        self.y2 = int(yy)
        self.Label = label
        self.class_prob = prob



args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True
# load modes
# snapshot load
model_path = './snapshot_6.pth.tar'
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model('test')
#model = model.to('cuda')
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

model_onnx = 'yolov7-tiny.onnx'
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = onnxruntime.InferenceSession(model_onnx, sess_options, providers=[ 'CUDAExecutionProvider'])

# prepare input image
transform = transforms.ToTensor()


cam = cv2.VideoCapture(0)

if cam.isOpened():
    CamWidth = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    CamHeight = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    Detection = []
    while True:
        check, frame = cam.read()
        DetectFrame = frame.copy()
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=False, crop=False)
        
        input_name = sess.get_inputs()[0].name  
        model_outputs = sess.get_outputs()
        output_names = [model_outputs[i].name for i in range(len(model_outputs))] 
        dt1 = datetime.datetime.now()
        outputs = sess.run(output_names, {input_name: blob})
        dt2 = datetime.datetime.now()
        deltaT = dt2-dt1;
        print(f'detection time {deltaT.microseconds/1000}')
        if len(outputs[0])>0:
            Detection.clear()
            for i in range(0, len(outputs[0]), 1):
                if int(outputs[0][i][5]) == 0:  #We only need Person 
                    Detection.append(Tracks(int(outputs[0][i][0]),outputs[0][i][1] / 640 * CamWidth, outputs[0][i][2] / 640 * CamHeight, 
                    outputs[0][i][ 3] / 640 * CamWidth, outputs[0][i][ 4] / 640 * CamHeight, int(outputs[0][i][5]), outputs[0][i][6]) )    
            for item in Detection:
                cv2.rectangle(DetectFrame, (item.x1,item.y1), (item.x2,item.y2),(0, 0, 0), 1)
            #cv2.imshow('result', DetectFrame)

        # prepare bbox
        MeshImage = frame.copy()
        for item in Detection:
            original_img = frame[:,:,::-1].copy()
            original_img = original_img.astype(np.float32)
            original_img_height, original_img_width = original_img.shape[:2]
        #if True:
            bbox = [item.x1,item.y1, item.x2- item.x1,item.y2-item.y1] # xmin, ymin, width, height
           
           
            bbox = process_bbox(bbox, original_img_width, original_img_height)

            img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape) 
            img = transform(img.astype(np.float32))/255.0
            img = img.cuda()[None,:,:,:]

            # forward
            inputs = {'img': img}
            targets = {}
            meta_info = {}
            dt1 = datetime.datetime.now()
            with torch.no_grad():
                out = model(inputs, targets, meta_info, 'test')
            dt2 = datetime.datetime.now()
            deltaT = dt2-dt1;
            print(f'Pose time {deltaT.microseconds/1000}')
            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

            vis_img = MeshImage.copy()
            focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
            princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
            MeshImage = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})
            MeshImage = MeshImage.astype(np.uint8)
        cv2.imshow('video', MeshImage)
        key = cv2.waitKey(1)
        if key == 27:
            break
    
    cam.release()
    cv2.destroyAllWindows()