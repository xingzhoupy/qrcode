import simple_barcode_detection
import cv2
import numpy as np
from PIL import Image

# import zbar
# 接下去是创建一个扫描器,他可以解析二维码的内容

# create a reader
# scanner = zbar.ImageScanner()
# configure the reader
# scanner.parse_config('enable')
# 设置屏幕显示字体
font = cv2.FONT_HERSHEY_SIMPLEX
# 启用摄像头
camera = cv2.VideoCapture(0)
# 接下去是一个大的while循环
while True:
    # 得到当前的帧
    # grab the current frame
    (grabbed, frame) = camera.read()
    # print(frame.shape)
    # 检测视频是否到底了，如果检测视频文件里面的二维码或条形码用这个，如果开启摄像头就无所谓了
    if not grabbed:
        break
    # 调用函数来查找二维码返回二维码的位置

    box = simple_barcode_detection.detect(frame)

    if box is not None:
        # 这下面的3步得到扫描区域，扫描区域要比检测出来的位置要大
        # min = np.min(box, axis=0)
        # max = np.max(box, axis=0)
        #
        # roi = frame[min[1] - 10:max[1] + 10, min[0] - 10:max[0] + 10]

        # 把区域里的二维码传换成RGB，并把它转换成pil里面的图像，因为zbar得调用pil里面的图像，而不能用opencv的图像
        # roi = cv2.cvtColor(box, cv2.COLOR_BGR2RGB)
        # pil = Image.fromarray(frame).convert('L')
        # width, height = pil.size
        # raw = pil.tostring()

        # 把图像装换成数据
        # zarimage = zbar.Image(width, height, 'Y800', raw)

        # 扫描器进行扫描Q
        # scanner.scan(zarimage)

        # 得到结果
        # for symbol in zarimage:
        # 对结果进行一些有用的处理
        #     print('decoded', symbol.type, 'symbol', '"%s"' %symbol.data)
        cv2.polylines(frame, np.int32([box]), True, (0, 0, 255), 10)
        # cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

        # 把解析的内容放到视频上
        # cv2.putText(frame, symbol.data, (20, 100), font, 1, (0, 255, 0), 4)
        cv2.putText(frame,"demo",(20,100),font,1,(0,255,0),4)

        # show the frame and record if the user presses a key
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    else:
        cv2.putText(frame, "0", (20, 100), font, 1, (0, 255, 0), 4)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
