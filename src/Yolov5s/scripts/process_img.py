#!/usr/bin/env python3
from email import header
import rospy
from rospkg import RosPack
from sensor_msgs.msg import Image
import queue
from  Yolov5s.msg import (Msg_x, Box)
import os
import sys
import numpy as np
import cv_bridge as bridge
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import (non_max_suppression, set_logging, yolov5_in_syspath)
from skimage.transform import resize
import torch



#方便调用path下的其他python文件中的方法
path = os.path.abspath(".")
sys.path.insert(0,path + "/home/shinan/catkin_ws/src/Yolov5s/scripts")

img0_buf  = queue.Queue()
img1_buf = queue.Queue()
def doMsg0(data):
    #接收图像 消息流
    #创建缓存区以队列保存消息流
    img0_buf.put(data)
    rospy.loginfo("get imgptr sucessfully!\n")

def doMsg1(data):
    #接收图像 消息流
    #创建缓存区以队列保存消息流
    img1_buf.put(data)


#图像预处理函数
def imagePreProcessing( img):
    # Extract image and shape
    img = np.copy(img)
    img = img.astype(float)
    h, w, channels = img.shape

    # Determine image to be used
    padded_image = np.zeros((max(h, w), max(h, w), channels)).astype(float)

    # Add padding
    if (w > h):
        padded_image[(w - h) // 2: h + (w - h) // 2, :, :] = img
    else:
        padded_image[:, (h - w) // 2: w + (h - w) // 2, :] = img

    # Resize and normalize    ---------------------------------------------------------------------？？？？？？？？
    input_img = resize(padded_image, (network_img_size, network_img_size, 3)) / 255.

    # Channels-first
    input_img = np.transpose(input_img, (2, 0, 1))

    # As pytorch tensor
    input_img = torch.from_numpy(input_img).float()
    input_img = input_img[None]

    return input_img


if __name__=="__main__":
    # Initialize node
    rospy.init_node("porcess_img_node")

    package = RosPack()
    package_path = package.get_path('Yolov5s')
    # Load weights parameter
    weights_name = rospy.get_param('~weights_name', 'yolov5/yolov5s.pt')
    weights_path = os.path.join(package_path, 'scripts', weights_name)
    rospy.loginfo("Found weights, loading %s", weights_path)

    # Load image parameter and confidence threshold
    confidence_th = rospy.get_param('~confidence', 0.25)
    nms_th = rospy.get_param('~nms_th', 0.45)

    # Load other parameters                     -----------------------------------------------------------------？？？？？？？
    network_img_size = rospy.get_param('~img_size', 640)

    # Initialize                                                    -----------------------------------------------------------？？？？？？？？？？？
    set_logging()
    device = 'cpu'
    # device = select_device('cpu')

    # Load model
    model = attempt_load(weights_name, map_location=device)  # load FP32 model
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    stride = int(model.stride.max())  # model stride

    # Load Class and initialize
    class_name = rospy.get_param('~class_name', None)
    # print(names)
    if class_name == "None":
        class_name = None
    elif type(class_name) == str:
        class_name = names.index(class_name)
    else:
        class_name = class_name.split(', ')
        class_name = [names.index(int(i)) for i in class_name]

    print(class_name)

    # Initialize width and height
    h = 0
    w = 0
   
    classes_colors = {}       # ----------------------------------------------？？？？？？？？？？？？？？？？？？？？？

    #订阅图像
    sub_img0 = rospy.Subscriber("/iris_0/stereo_camera/left/image_raw",Image,doMsg0,queue_size=1)
    sub_img1 = rospy.Subscriber("/iris_0/stereo_camera/right/image_raw",Image,doMsg1,queue_size=1)

    pub_msg0 = rospy.Publisher("pub_x_msg0",Msg_x,queue_size=1)
    pub_msg1 = rospy.Publisher("pub_x_msg1",Msg_x,queue_size=1)

    # 设置循环频率
    rate = rospy.Rate(1)
    while not rospy.is_shutdown(): 

        #操作订阅的图像
        #1.从缓存中取出消息
        img0 = img0_buf.get()
        img1 = img1_buf.get()

        rospy.loginfo("img info: img_height = %d\n img_width = %d\n encoding = %s\n", img0.height, img0.width, img0.encoding )
        #2.图片传入yolo，并接收框出的坐标列表
        try:
            cv_image0 = bridge.imgmsg_to_cv2(img0, "rgb8")
            cv_image1 = bridge.imgmsg_to_cv2(img1, "rgb8")
        except bridge.CvBridgeError as e:
            print(e)

        input_img0 = imagePreProcessing(cv_image0)
        input_img1 = imagePreProcessing(cv_image1)
        with torch.no_grad():
            # img /= 255.0  # 0 - 255 to 0.0 - 1.0
            # if img.ndimension() == 3:
            #     img = img.unsqueeze(0)
            detections0 = model(input_img0, augment=False)[0]
            detections0 = non_max_suppression(detections0, confidence_th, nms_th, classes=class_name, agnostic=False)
            detections1 = model(input_img1, augment=False)[0]
            detections1 = non_max_suppression(detections1, confidence_th, nms_th, classes=class_name, agnostic=False)

            #   -----------------------------------------------------------？？？？？？？？？？？？？？？？
        if detections0[0] is not None:
            for detection in detections0[0]:
                # # Get xmin, ymin, xmax, ymax, confidence and class
                # print(detection)
                xmin, ymin, xmax, ymax, conf, det_class = detection
                pad_x = max(h - w, 0) * (network_img_size/max(h, w))
                pad_y = max(w - h, 0) * (network_img_size/max(h, w))
                unpad_h = network_img_size-pad_y
                unpad_w = network_img_size-pad_x
                xmin_unpad = ((xmin-pad_x//2)/unpad_w)*w
                xmax_unpad = ((xmax-xmin)/unpad_w)*w + xmin_unpad
                ymin_unpad = ((ymin-pad_y//2)/unpad_h)*h
                ymax_unpad = ((ymax-ymin)/unpad_h)*h + ymin_unpad
            #组织被发布的数据，并编写逻辑发布数据（数据包括图像和坐标框列表）
        msg0  = Msg_x()
        msg0.header = img0.header
        msg0.height = img0.height
        msg0.width = img0.width
        msg0.is_bigendian = img0.is_bigendian
        msg0.step = img0.step
        msg0.encoding = img0.encoding
        msg0.data = img0.data
        detection0_msg = Box()
        detection0_msg.xmin = int(xmin_unpad.item())
        detection0_msg.xmax = int(xmax_unpad.item())
        detection0_msg.ymin = int(ymin_unpad.item())
        detection0_msg.ymax = int(ymax_unpad.item())
        detection0_msgprobability = conf.item()
        detection0_msg.Class = names[int(det_class.item())]
        print(detection0_msg.xmin, detection0_msg.xmax, detection0_msg.ymin, detection0_msg.ymax, detection0_msgprobability, detection0_msg.Class)
        msg0.boxes.append(detection0_msg)


        if detections1[0] is not None:
            for detection in detections1[0]:
                # # Get xmin, ymin, xmax, ymax, confidence and class
                # print(detection)
                xmin, ymin, xmax, ymax, conf, det_class = detection
                pad_x = max(h - w, 0) * (network_img_size/max(h, w))
                pad_y = max(w - h, 0) * (network_img_size/max(h, w))
                unpad_h = network_img_size-pad_y
                unpad_w = network_img_size-pad_x
                xmin_unpad = ((xmin-pad_x//2)/unpad_w)*w
                xmax_unpad = ((xmax-xmin)/unpad_w)*w + xmin_unpad
                ymin_unpad = ((ymin-pad_y//2)/unpad_h)*h
                ymax_unpad = ((ymax-ymin)/unpad_h)*h + ymin_unpad

        msg1 = Msg_x()
        msg1.header = img1.header
        msg1.height = img1.height
        msg1.width = img1.width
        msg1.is_bigendian = img1.is_bigendian
        msg1.step = img1.step
        msg1.encoding = img1.encoding
        detection1_msg = Box()
        detection1_msg.xmin = int(xmin_unpad.item())
        detection1_msg.xmax = int(xmax_unpad.item())
        detection1_msg.ymin = int(ymin_unpad.item())
        detection1_msg.ymax = int(ymax_unpad.item())
        detection1_msgprobability = conf.item()
        detection1_msg.Class = names[int(det_class.item())]
        msg1.boxes.append(detection1_msg)


        pub_msg0.publish(msg0)
        pub_msg0.publish(msg1)

        rate.sleep()

        rospy.loginfo("...")
        rospy.spin()
