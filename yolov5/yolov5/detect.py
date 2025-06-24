#!usr/bin/env python3

import numpy as np
import torch
import rclpy
import loguru
import sys

import os
from pathlib import Path
from ultralytics.utils.plotting import Annotator, colors
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (Profile, check_file, check_img_size, cv2, non_max_suppression, scale_boxes)
from yolov5.utils.torch_utils import select_device, smart_inference_mode

from rclpy.node import Node, Clock
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from yolo_msgs.msg import BoundingBox, BoundingBoxArray

@smart_inference_mode()

class Detect(Node):
    def __init__(self):
        super().__init__('inference_camera')

        loguru.logger.remove(0)
        loguru.logger.add(sys.stderr, format=
                          '<red>{level}</red> | <green>{time:YYYY-MM-DD HH:mm:ss}</green> | <yellow>{message}</yellow>', 
                          colorize=True
                          )
        loguru.logger.info('Initializing inference node')

        self.cv_bridge = CvBridge()
        self.image = Image()
        self.bbox_arr = BoundingBoxArray()
        self.clock = Clock()

        self.create_rate(30)
        self.img_pub = self.create_publisher(Image, '/inference/image_raw', 10)
        self.obj_pub = self.create_publisher(BoundingBoxArray, '/inference/object_raw', 10)

        # self.weights = '/home/ict/Wook/Fusion_ws/src/yolov5/yolov5/data/best.pt'
        self.weights = '/home/ict/Wook/Fusion_ws/src/yolov5/yolov5/yolov5n.pt'
        self.data = '/home/ict/Wook/yolov5_ws2/src/yolov5/yolov5/data/coco.yaml'  # 있어도 없어도 이상이 없는데 왜인지 모르겠음 ;;;
        self.source = 2 # 0 for webcam, path to video file, or image folder
        self.img_size = (640, 640) # (height, width)
        self.conf_thres = 0.25 # confidence threshold
        self.iou_thres = 0.45 # NMS IoU threshold
        self.max_det = 1000 # maximum detections per image
        self.device = 0# cuda device, i.e. 0 or 0,1,2,3 or cpu
        # self.device = 'cpu' # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img = True # display results
        self.nosave = False # do not save images or videos
        self.classes = None # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False # class-agnostic NMS
        self.augment = False # augmented inference
        self.half = False # use FP16 half-precision inference
        self.dnn = False # use DNN for ONNX models
        self.vid_stride = 1 # video frames stride
        self.visualize = False # visualize features
        self.line_thickness = 2 # bounding box thickness (pixels)
        self.hide_labels = False # hide labels
        self.hide_conf = False # hide confidences

    def prepareDetection(self):
        """
        Prepare the model for detection
        """

        source = str(self.source)
        save_img = not self.nosave and not source.endswith('.txt')
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS) # false
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) # false
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file) # true

        if is_url and is_file:
            source = check_file(source)

        device = select_device(self.device)
        
        model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)   # self.dnn: False, self.half: False
        loguru.logger.info(f"data: {self.data}")
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(self.img_size, s=stride)

        # source: 0, is_file: False, is_url: False, webcam: True, pt: True, stride: 32, names: {0: 'person', ...} 출력됨
        return source, save_img, is_file, is_url, webcam, device, model, stride, names, imgsz
    
    def streamSelect(self, source, webcam, imgsz, stride):
        """
        """

        # if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        # else:
        #     dataset = LoadImages(source, img_size=imgsz, stride=stride)

        return dataset
    
    def showFps(self, img, fps):
        """
        Show FPS on top left corner of the image

        Args:
            img (np.ndarray): image
            fps (float): fps value

        Returns:
            np.ndarray: image with fps
        """
        font = cv2.FONT_HERSHEY_PLAIN
        line = cv2.LINE_AA
        fps_text = 'FPS: {:.2f}'.format(fps)
        cv2.putText(img, fps_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
        cv2.putText(img, fps_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)

        return img

    def detect(self):
        """
        """

        source, save_img, _, _, webcam, device, model, stride, names, imgsz = self.prepareDetection()
        # loguru.logger.info(f"source: {source}, device: {device}, model.device: {model.device}, stride: {stride}")
        dataset = self.streamSelect(source, webcam, imgsz, stride)

        seen, dt = 0, (Profile(), Profile(), Profile()) # delta time (data, nms, tot)
        start_time = self.clock.now()
        for path, im, im0s, _ , s in dataset:
            with dt[0]: # data time
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()
                im /= 255

                if len(im.shape) == 3:
                    im = im[None]

            with dt[1]: # nms time
                pred = model(im, augment=self.augment, visualize=self.visualize)

            with dt[2]: # tot time
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

            for i, det in enumerate(pred): # det 은 인식된 객체 수
                seen += 1
                if webcam:
                    p, im0, _ = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: ' # 0: 뜸
                else:
                    p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

                s += '%gx%g ' % im.shape[2:] # 480x640 나옴
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(names)) # names에 객체 이름들 다 있음
                if len(det): # detections not empty
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round() # to original shape
                    for c in det[:, 5].unique(): # draw each class
                        n = (det[:, 5] == c).sum()
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                    self.bbox_arr.bounding_box_array.clear()
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        label = names[c] if self.hide_conf else f'{names[c]}' # 인식된 라벨 값
                        # if label == "person":
                        # print(xyxy)
                        x_min, y_min, x_max, y_max = xyxy # xyxy에 객체인식된 좌표값이 들어있음 torch타입으로 되어있음
                        h = float(y_max - y_min)
                        l = float(x_max - x_min)

                        if device == 'cpu':
                            center_x = float(np.mean([x_min, x_max]))
                            center_y = float(np.mean([y_min, y_max]))
                        else:
                            center_x = float(torch.mean(torch.tensor([x_min, x_max])))
                            center_y = float(torch.mean(torch.tensor([y_min, y_max])))

                        self.bbox = BoundingBox()
                        self.bbox.class_name = label
                        self.bbox.x_max = int(x_max)
                        self.bbox.x_min = int(x_min)
                        self.bbox.y_max = int(y_max)
                        self.bbox.y_min = int(y_min)
                        self.bbox.bounding_box.center.position.x = center_x
                        self.bbox.bounding_box.center.position.y = center_y
                        self.bbox.bounding_box.size_x = l
                        self.bbox.bounding_box.size_y = h
                        self.bbox_arr.bounding_box_array.append(self.bbox)

                        if save_img or self.view_img:
                            c = int(cls)
                            annotator.box_label(xyxy, label, color=colors(c, True))

                    self.bbox_arr.header.stamp = self.get_clock().now().to_msg()
                    self.bbox_arr.header.frame_id = 'camera'

                im0 = annotator.result()
                inference_time = dt[1].dt * 1E3
                elapsed_time = (self.clock.now() - start_time).nanoseconds / 1E9
                fps = 1 / elapsed_time
                self.image = self.cv_bridge.cv2_to_imgmsg(im0, 'bgr8')
                self.image.header.stamp = self.get_clock().now().to_msg()
                try:
                    self.obj_pub.publish(self.bbox_arr)
                    self.img_pub.publish(self.image)

                    # 시간 확인
                    # loguru.logger.info(f"self.bbox_arr {self.bbox_arr.header.stamp}    ,       im0 {self.image.header.stamp}")
                except CvBridgeError as e:
                    print(e)

                # 이미지 표시
                # if self.view_img:
                    # cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    # cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    # cv2.imshow(str(p), im0)
                    # if cv2.waitKey(1) == ord('q'):
                        # loguru.logger.warning('Exit via keyboard interrupt (^C) or "q"')
                        # raise StopIteration

            loguru.logger.info(f"{s}{'' if len(det) else '(no detections), '}{inference_time:.1f}ms, FPS: {fps:.2f}")

            start_time = self.clock.now()

        t = tuple(x.t / seen * 1E3 for x in dt)
        # loguru.logger.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)


def main(args=None):

    rclpy.init(args=args)
    detect = Detect()
    detect.detect()

    rclpy.spin(detect)

    detect.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
