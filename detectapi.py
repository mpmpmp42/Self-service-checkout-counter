# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


class DetectAPI:
    def __init__(self, weights='best.pt', data='dataset.yaml',
                 source="static/images", conf_thres=0.25,
                 iou_thres=0.45, max_det=1000, device='', view_img=False, save_txt=False,
                 save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False,
                 visualize=False, update=False, project='runs/detect', name='myexp', exist_ok=False, line_thickness=3,
                 hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1):

        self.weights = weights
        self.data = data
        self.source = source
        self.imgsz = [640, 640]
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.view_img = view_img
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.save_crop = save_crop
        self.nosave = nosave
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.visualize = visualize
        self.update = update
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.half = half
        self.dnn = dnn
        self.vid_stride = vid_stride
        self.dataset = None
        self.dt = None
        self.model = None
        self.save_dir = None
        self.seen = None
        self.webcam = None
        self.name2 = None
        self.imgsz2 = None
        self.stride = None
        self.pt = None

    def loadmodel(self):

        # Load model
        device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.name2, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz2 = check_img_size(self.imgsz, s=self.stride)  # check image size

    def loaddata(self):
        source = str(self.source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download

        bs = 1  # batch_size
        if self.webcam:
            self.dataset = LoadStreams(source, img_size=self.imgsz2, stride=self.stride, auto=self.pt,
                                       vid_stride=self.vid_stride)
            bs = len(self.dataset)
        elif screenshot:
            self.dataset = LoadScreenshots(source, img_size=self.imgsz2, stride=self.stride, auto=self.pt)
        else:
            self.dataset = LoadImages(source, img_size=self.imgsz2, stride=self.stride, auto=self.pt,
                                      vid_stride=self.vid_stride)

        # Run inference
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *self.imgsz2))  # warmup
        self.seen, windows, self.dt = 0, [], (Profile(), Profile(), Profile())

    def run(self):
        all_list = []
        for path, im, im0s, vid_cap, s in self.dataset:
            www = {}
            with self.dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with self.dt[1]:
                self.visualize = increment_path(self.save_dir / Path(path).stem,
                                                mkdir=True) if self.visualize else False
                pred = self.model(im, augment=self.augment, visualize=self.visualize)

            # NMS
            with self.dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                           max_det=self.max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                self.seen += 1
                if self.webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), self.dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(self.dataset, 'frame', 0)

                s += '%gx%g ' % im.shape[2:]  # print string

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        w = f"{n} {self.name2[int(c)]}{'s' * (n > 1)} "
                        LOGGER.info(w)
                        # s += f"{n} {names[int(c)]}{'s' * (n > 1)} "  # add to string!!!!
                        www["label"] = self.name2[int(c)]
                        www["number"] = int(n)
                        all_list.append(www)

        return all_list
