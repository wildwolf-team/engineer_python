from pickle import FALSE
from time import time
from utils.general import check_img_size
from utils.torch_utils import select_device
import math
import cv2
import numpy as np
from sympy import false, true
import cv2
import torch
import numpy as np
from models.common import DetectMultiBackend
from utils.general import check_img_size,non_max_suppression,scale_coords, xyxy2xywh
from utils.torch_utils import select_device
import serial
import time
from typing import List

class Function:

    DEVIATION_X = -1
    DIRECTION = -1
    SEND_DATA_1 = -1
    SEND_DATA_0 = -1
    TARGET_X = 400
    def __init__(self,weights):
        self.ser = serial.Serial()
        self.ser.port = "/dev/ttyUSB0"
        # self.ser.baudrate = 115200
        self.ser.baudrate = 921600
        self.ser.bytesize = 8
        self.ser.parity = 'N'
        self.ser.stopbits = 1
        try:
            self.ser.open()
        except:
            print("Serial Open Error")

        # 加载模型
        self.device = select_device('cpu')
        self.model = DetectMultiBackend(weights, device=self.device)
        self.stride = self.model.stride 
        # stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        self.imgsz = check_img_size((320,320),s=self.stride)
        self.model.model.float()
    
    def radix_sort(arr:List[int]):
        n = len(str(max(arr)))  # 记录最大值的位数
        for k in range(n):#n轮排序
            # 每一轮生成10个列表
            bucket_list=[[] for i in range(10)]#因为每一位数字都是0~9，故建立10个桶
            for i in arr:
                # 按第k位放入到桶中
                bucket_list[i//(10**k)%10].append(i)
            # 按当前桶的顺序重排列表
            arr=[j for i in bucket_list for j in i]
        return arr

    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    # 进行推理 绘制图像 结算出最优 发送数据
    def to_inference(self, frame, device, model, imgsz, stride, conf_thres=0.45, iou_thres=0.45):
        img_size = frame.shape
        img0 = frame 
        img = Function.letterbox(img0,imgsz,stride=stride)[0]
        img = img.transpose((2,0,1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.

        if len(img.shape) == 3:
            img = img[None]

        pred = model(img)
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)
        aims = []
        self.direction = 0
        self.deviation_x = 0
        confs = []
        arr = []

        for i ,det in enumerate(pred):
            gn = torch.tensor(img0.shape)[[1,0,1,0]]
            if len(det):
                det[:,:4] = scale_coords(img.shape[2:], det[:, :4],img0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1,4)) / gn).view(-1).tolist()
                    line = (cls, *xywh)
                    aim = ('%g ' * len(line)).rstrip() % line 
                    aim = aim.split(' ')
                    if float(conf) > 0.4:
                        aims.append(aim)
                        confs.append(float(conf))

            if len(aims):
                for i,det in enumerate(aims):
                    tag, x_center, y_center, width, height = det
                    x_center, width = float(x_center) * img_size[1], float(width) * img_size[1]
                    y_center, height = float(y_center) * img_size[0], float(height) * img_size[0]
                    top_left = (int(x_center - width * 0.5), int(y_center - height * 0.5))
                    top_right = (int(x_center + width * 0.5), int(y_center - height * 0.5))
                    bottom_right = (int(x_center + width * 0.5), int(y_center + height * 0.5))

                    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 3, 8)
                    cv2.putText(frame,str(float(round(confs[i], 2))), top_right, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    cv2.putText(frame, tag, top_left, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 4)

                    arr.append(int(x_center - Function.TARGET_X))
               
                if abs(Function.radix_sort(arr)[0]) < abs(Function.radix_sort(arr)[len(arr)-1]):
                    self.deviation_x = Function.radix_sort(arr)[0]
                else:
                    self.deviation_x = Function.radix_sort(arr)[len(arr)-1]
                cv2.putText(frame, "x = " + str(self.deviation_x), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                if abs(self.deviation_x) < (bottom_right[0]- top_left[0] - 10)/4:
                    self.deviation_x = 0
                if self.deviation_x > 0:
                    self.direction = 1
                cv2.putText(frame, "x = " + str(self.deviation_x), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

            cv2.line(frame, (0, int(img_size[0] * 0.5)), (int(img_size[1]), int(img_size[0] * 0.5)), (255, 0, 255), 3)
            cv2.line(frame, (Function.TARGET_X, 0), (Function.TARGET_X, int(img_size[0])), (255, 0, 255), 3)

            self.send_data_0 = (self.deviation_x >> 8) & 0xff
            self.send_data_1 = self.deviation_x & 0xff
            
            Function.DEVIATION_X = self.deviation_x
            Function.DIRECTION = self.direction
            Function.SEND_DATA_0 = self.send_data_0
            Function.SEND_DATA_1 = self.send_data_1
            cv2.putText(frame, ('S' + str(self.direction) + str(self.send_data_0) + str(self.send_data_1) + 'E'), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            
    def send_data(self):
        while 1:
            time.sleep(0.005)
            if   Function.SEND_DATA_1 / 100 > 0:
                self.ser.write(('S' + str(Function.DIRECTION) + str(Function.SEND_DATA_0) + str(Function.SEND_DATA_1) + 'E').encode("utf-8"))

            elif Function.SEND_DATA_1 / 10 > 0:
                self.ser.write(('S' + str(Function.DIRECTION) + str(Function.SEND_DATA_0) + str(0) + str(Function.SEND_DATA_1) + 'E').encode("utf-8"))

            elif Function.SEND_DATA_1 / 1 > 0:
                self.ser.write(('S' + str(Function.DIRECTION) + str(Function.SEND_DATA_0) + str(0) + str(0) + str(Function.SEND_DATA_1) + 'E').encode("utf-8"))

            elif Function.DEVIATION_X == 0:
                self.ser.write(('S' + str(2) + str(0) + str(0) + str(0) + str(0) + 'E').encode("utf-8"))
                
            else:
                self.ser.write(('S' + str(2) + str(0) + str(0) + str(0) + str(0) + 'E').encode("utf-8"))
