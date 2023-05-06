from pickle import FALSE
from time import time
from utils.general import check_img_size
from utils.torch_utils import select_device
import cv2
import numpy as np
import torch
import numpy as np
from models.common import DetectMultiBackend
from utils.general import check_img_size,non_max_suppression,scale_coords, xyxy2xywh
from utils.torch_utils import select_device
import serial
import time
from typing import List

class Function:

    DEVIATION_X = 0
    DIRECTION = 0
    HIGH_EIGHT = 0
    LOW_EIGHT = 0
    TARGET_X = 0 #夹取机构相对相机位置

    def __init__(self,weights):
        self.ser = serial.Serial()
        self.ser.port = "/dev/ttyUSB0"
        self.ser.baudrate = 921600
        self.ser.bytesize = 8
        self.ser.parity = 'N'
        self.ser.stopbits = 1
        try:
            self.ser.open()
        except:
            self.ser.close()
            print("Serial Open Error")

        # 加载模型
        self.device = select_device('cpu')
        self.model = DetectMultiBackend(weights, device=self.device)
        self.stride = self.model.stride 
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
    def to_inference(self, frame, device, model, imgsz, stride,mode = 1, conf_thres=0.45, iou_thres=0.45):
        img_size = frame.shape
        img0 = frame 
        img = Function.letterbox(img0,imgsz,stride=stride)[0]
        img = img.transpose((2,0,1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.

        # 每次初始化防止数据未刷新自己走，可能会慢一些
        Function.DEVIATION_X = 0
        Function.DIRECTION = 0
        Function.HIGH_EIGHT = 0
        Function.LOW_EIGHT = 0

        if len(img.shape) == 3:
            img = img[None]

        pred = model(img)
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)
        aims = []
        confs = []
        arr = []

        # 可以加个矿石面积判断
        for i ,det in enumerate(pred): 
            gn = torch.tensor(img0.shape)[[1,0,1,0]]
            if len(det):
                det[:,:4] = scale_coords(img.shape[2:], det[:, :4],img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1,4)) / gn).view(-1).tolist()
                    line = (cls, *xywh)
                    aim = ('%g ' * len(line)).rstrip() % line 
                    aim = aim.split(' ')
                    if float(conf) > 0.7:
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

                    Function.draw_inference(frame, top_left, top_right, bottom_right, tag, confs, i, mode)

                    arr.append(int(x_center - Function.TARGET_X)) 

                if abs(Function.radix_sort(arr)[0]) < abs(Function.radix_sort(arr)[-1]):
                    Function.DEVIATION_X = Function.radix_sort(arr)[0]
                else:
                    Function.DEVIATION_X = Function.radix_sort(arr)[-1]

                if mode == 1:
                    cv2.putText(frame, "real_x = " + str(Function.DEVIATION_X), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

                Function.HIGH_EIGHT = (abs(Function.DEVIATION_X) >> 8) & 0xff
                Function.LOW_EIGHT = abs(Function.DEVIATION_X)  & 0xff

                if Function.FLAG == 1:
                    if abs(Function.DEVIATION_X ) < 24: #24
                        Function.DEVIATION_X  = 0
                else :
                    if abs(Function.DEVIATION_X ) < 24: #24
                        Function.DEVIATION_X  = 0
                if Function.DEVIATION_X > 0:
                    Function.DIRECTION = 1

            Function.draw_data(frame, img_size, mode)


    def serial_connection(self):
        self.ser.port = "/dev/ttyUSB0"
        self.ser.baudrate = 921600
        self.ser.bytesize = 8
        self.ser.parity = 'N'
        self.ser.stopbits = 1
        try:
            self.ser.open()
            print('Open')
        except:
            print("Serial Open Error")

    # 0 左 1右 2停
    def send_data(self):
        while 1:
            time.sleep(0.0005)
            try:
                if Function.DEVIATION_X == 0:
                    self.ser.write(('S' + str(2) + str(0) + str(0) + str(0) + str(0) + 'E').encode("utf-8"))
                    
                elif   Function.LOW_EIGHT / 100 >= 1:
                    self.ser.write(('S' + str(Function.DIRECTION) + str(Function.HIGH_EIGHT) + str(Function.LOW_EIGHT) + 'E').encode("utf-8"))

                elif Function.LOW_EIGHT / 10 >= 1:
                    self.ser.write(('S' + str(Function.DIRECTION) + str(Function.HIGH_EIGHT) + str(0) + str(Function.LOW_EIGHT) + 'E').encode("utf-8"))

                elif Function.LOW_EIGHT / 1 >= 1:
                    self.ser.write(('S' + str(Function.DIRECTION) + str(Function.HIGH_EIGHT) + str(0) + str(0) + str(Function.LOW_EIGHT) + 'E').encode("utf-8"))

                else:
                    self.ser.write(('S' + str(2) + str(0) + str(0) + str(0) + str(0) + 'E').encode("utf-8"))
            except:
                print('Serial Send Data Error')
                cv2.putText(frame, " Serial Send Data Error " , (0, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                self.ser.close()
                Function.serial_connection(self)                
                
    def draw_inference(frame, top_left, top_right, bottom_right, tag, confs, i, mode = 1):
        if mode == 1:
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 3, 8)
            cv2.putText(frame,str(float(round(confs[i], 2))), top_right, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(frame, tag, top_left, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 4)
 
    def draw_data(frame, img_size, mode = 1):
        if mode == 1:
            cv2.putText(frame, "judge_x = " + str(Function.DEVIATION_X), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.line(frame, (Function.TARGET_X, 0), (Function.TARGET_X, int(img_size[0])), (255, 0, 255), 3)
            cv2.putText(frame, 'direction: ' + str(Function.DIRECTION), (0, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(frame, 'high_eight: ' + str(Function.HIGH_EIGHT), (0, 210), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(frame, 'low_eight: ' + str(Function.LOW_EIGHT), (0, 260), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
