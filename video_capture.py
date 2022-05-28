import mvsdk
from pickle import FALSE
import cv2
import time
from utils.torch_utils import time_sync
import numpy as np

class Video_capture:
    COLS = 1280
    ROWS = 800
    ExposureTime = 30 * 1000
    IS_SAVE_VIDEO = 0
    CAMERA_OPEN = 0
    # 相机初始化配置
    def __init__(self,is_save_video = 0):

        Video_capture.IS_SAVE_VIDEO = is_save_video

        DevList = mvsdk.CameraEnumerateDevice()
        try:
            for i, DevInfo in enumerate(DevList):
                ("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
            i = 0 
            DevInfo = DevList[i]
            print(DevInfo)
            # 打开相机
            self.hCamera = 0
            try:
                self.hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
            except mvsdk.CameraException as e:
                print("CameraInit Failed({}): {}".format(e.error_code, e.message) )
            
            # 录制视频
            if Video_capture.IS_SAVE_VIDEO :
                try:
                    print(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
                    time_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                    # print(time_name)
                    sfourcc = cv2.VideoWriter_fourcc(*'XVID')#视频存储的格式
                    #视频的宽高
                    self.out = cv2.VideoWriter('./video/' + time_name + '.avi', sfourcc, 30, (Video_capture.COLS,Video_capture.ROWS))#视频存储
                except:
                    print("To Save Video Error")
                    
            # 获取相机特性描述
            cap = mvsdk.CameraGetCapability(self.hCamera)

            # 判断是黑白相机还是彩色相机
            monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

            # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
            if monoCamera:
                mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
            else:
                mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

            # 相机模式切换成连续采集
            mvsdk.CameraSetTriggerMode(self.hCamera, 0)

            pImageResolution = mvsdk.CameraGetImageResolution(self.hCamera)
            pImageResolution.iIndex       = 0xFF
            pImageResolution.iWidthFOV    = Video_capture.COLS
            pImageResolution.iHeightFOV   = Video_capture.ROWS
            pImageResolution.iWidth       = Video_capture.COLS
            pImageResolution.iHeight      = Video_capture.ROWS
            pImageResolution.iHOffsetFOV  = int((1280 - Video_capture.COLS) * 0.5)
            pImageResolution.iVOffsetFOV  = int((1024 - Video_capture.ROWS) * 0.5) 

            mvsdk.CameraSetImageResolution(self.hCamera, pImageResolution)

            # 手动曝光，曝光时间30ms
            mvsdk.CameraSetAeState(self.hCamera, 0)
            mvsdk.CameraSetExposureTime(self.hCamera, Video_capture.ExposureTime )
            mvsdk.CameraSetWbMode(self.hCamera, FALSE)
            mvsdk.CameraSetOnceWB(self.hCamera)

            # 让SDK内部取图线程开始工作
            mvsdk.CameraPlay(self.hCamera)

            # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
            FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

            # 分配RGB buffer，用来存放ISP输出的图像
            # 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
            self.pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
            Video_capture.CAMERA_OPEN = 1
        except:
            print('Not Find Camera')
    

    # 只开启摄像头
    def only_capture(self):
        while (cv2.waitKey(1) & 0xFF) != ord('q'):
            # 从相机取一帧图片
            try:
                t2 = time_sync()
                pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
                mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
                mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)
                
                # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
                # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
                frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3) )
                t3 = time_sync()
                print("Inference == " + str(1/(t3 - t2)))
                cv2.imshow("frame",frame)

                if Video_capture.IS_SAVE_VIDEO:
                    try:
                        self.out.write(frame)
                    except:
                        print("Write Frame Error")
                
            except mvsdk.CameraException as e:
                if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                    print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message) )  

        if Video_capture.IS_SAVE_VIDEO:
            try:
                self.out.release()
            except:
                print("Release Frame Error")

        mvsdk.CameraUnInit(self.hCamera)
        mvsdk.CameraAlignFree(self.pFrameBuffer)



# if __name__ == "__main__" :

#     video = Video_capture(1)
#     # video.only_capture()

#     # thread1 = threading.Thread(target=Video_capture.only_capture,args=())
#     thread2 = threading.Thread(target=video.only_capture)


#     # thread1.start()
#     thread2.start()

