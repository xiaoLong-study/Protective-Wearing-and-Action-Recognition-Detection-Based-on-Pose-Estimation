# -*- coding: UTF-8 -*-
import cv2 as cv
import argparse
import numpy as np
import tensorflow as tf
import time
from utils import choose_run_mode, load_pretrain_model, set_video_writer
from Pose.pose_visualizer import TfPoseVisualizer
from Action.recognizer import load_action_premodel, framewise_recognize

import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# import pandas as pd
# from enum import Enum
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.layers.normalization import BatchNormalization
# from keras.optimizers import Adam
# from keras.models import load_model
# from collections import Counter
# import matplotlib.pyplot as plt
# from keras.callbacks import Callback
# import itertools
# from sklearn.metrics import confusion_matrix


##########safety-helmet######
#from time import time
from gluoncv import model_zoo, data, utils
#from matplotlib import pyplot as plt
import mxnet as mx
import gluoncv as gcv
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO networks with random input shape.')
    parser.add_argument('--network', type=str, default='yolo3_darknet53_voc',
                        # use yolo3_darknet53_voc, yolo3_mobilenet1.0_voc, yolo3_mobilenet0.25_voc
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--short', type=int, default=416,
                        help='Input data shape for evaluation, use 320, 416, 512, 608, '
                             'larger size for dense object and big size input')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='confidence threshold for object detection')

    parser.add_argument('--gpu', action='store_true',
                        help='use gpu or cpu.')

    args = parser.parse_args()
    return args




############# 01-05为action recognition 初始化部分
data_raw = [];     ###储存训练数据

print('01:')
parser = argparse.ArgumentParser(description='Action Recognition by OpenPose')
parser.add_argument('--video', default = "E:/LSETM/程序/Online-Realtime-Action-Recognition-based-on-OpenPose/test_out/openpose.mp4",
                   help='Path to video file.')          ####输入视频
# parser.add_argument('--video', help='Path to video file.')    ####输入实时数据（摄像头）
args = parser.parse_args()


print('03:')
# 参数初始化
realtime_fps = '0.0000'
start_time = time.time()
fps_interval = 1
fps_count = 0
run_timer = 0
frame_count = 0

print('04:')
# 读写视频文件（仅测试过webcam输入）
cap = choose_run_mode(args)
video_writer = set_video_writer(cap, write_fps=int(24.0))

print('05:')
# # 保存关节数据的txt文件，用于训练过程(for training)
#f = open('origin_data.txt', 'a+')
iii=0
jjj=0
#########################################################################################


#################  06 部分是safety hetmet 程序初始化部分 #################
print('06:')
args1 = parse_args()
if args1.gpu:
    ctx = mx.gpu(0)
else:
    ctx = mx.gpu(0)

net = model_zoo.get_model(args1.network, pretrained=False)

classes1 = ['hat', 'helmet', 'person', 'vest']
for param in net.collect_params().values():
    if param._data is not None:
        continue
    param.initialize()
net.reset_class(classes1)
net.collect_params().reset_ctx(ctx)

if args1.network == 'yolo3_darknet53_voc':
    net.load_parameters('E:/LSETM/程序/Online-Realtime-Action-Recognition-based-on-OpenPose/Safety-Helmet-Wearing-Dataset/darknet.params', ctx=ctx)
    # net.load_parameters('yolo3_darknet53_voc_best.params', ctx=ctx)
    print('use darknet to extract feature')
elif args1.network == 'yolo3_mobilenet1.0_voc':
    net.load_parameters('D:/mxnet/double/Safety-Helmet-Wearing-Dataset/mobilenet.params', ctx=ctx)
    print('use mobile1.0 to extract feature')
else:
    net.load_parameters('mobilenet0.25.params', ctx=ctx)
    print('use mobile0.25 to extract feature')
 ############################################################################################################
aa = []
cc = []
class_hat=[]
score_hat=[]
############ 07部分为程序主循环  #############
print('07:')
while cv.waitKey(1) < 0:    #检测按键，有键值按下时退出，不按键，默认为-1
    has_frame, show = cap.read()
   #  has_frame = 1
   #  show = '10.jpg'
    jjj=jjj+1
    if jjj == 1:  #采集数据时，fps在30帧左右的图片，5帧取一帧，识别时不需要这样 控制数据一秒采6-7下.设置为视频帧速时，即可逐帧读取
        jjj = 0
        if has_frame:
            fps_count += 1
            frame_count += 1
################ safety hemet主循环 ##############
            frame = mx.nd.array(show, ctx=mx.cpu(0))
            rgb_nd, scaled_frame = gcv.data.transforms.presets.yolo.transform_test(frame, short=200, max_size=700)
            rgb_nd = rgb_nd.copyto(mx.gpu(0))
            # Run frame through network
            class_IDs, scores, bounding_boxes = net(rgb_nd)
            scale = 1.0 * frame.shape[0] / scaled_frame.shape[0]
            img, aa, class_hat, score_hat= gcv.utils.viz.cv_plot_bbox(show, bounding_boxes[0], scores[0], class_IDs[0],
                                             class_names=net.classes, scale=scale)

            for i in range(len(aa)):

                cv.rectangle(img, (aa[i][0], aa[i][1]), (aa[i][2], aa[i][3]), (255, 23, 140), 2)
                cv.putText(img, '{:s} {:s}'.format(class_hat[i], score_hat[i]), (aa[i][0], aa[i][1] - 15),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (220, 12 + i * 10, 150), 2)

            height, width = show.shape[:2]

            # 显示实时FPS值
            if (time.time() - start_time) > fps_interval:
                # 计算这个interval过程中的帧数，若interval为1秒，则为FPS
                realtime_fps = fps_count / (time.time() - start_time)
                fps_count = 0  # 帧数清零
                start_time = time.time()
            fps_label = 'FPS:{0:.2f}'.format(realtime_fps)
            cv.putText(show, fps_label, (160, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (22, 200, 25), 3)  # 在图片的指定位置写字
            print(realtime_fps)

            # 显示目前的运行时长及总帧数
            if frame_count == 1:
                run_timer = time.time()
            run_time = time.time() - run_timer
            time_frame_label = '[Time:{0:.2f} | Frame:{1}]'.format(run_time, frame_count)
            cv.putText(show, time_frame_label, (5, height-15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
############   处理后图像显示  ###############
            #show = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            cv.imshow('Safety-Helmet', show)
            #gcv.utils.viz.cv_plot_image(show)
            video_writer.write(show)

            # #采集数据，用于训练过程(for training)
            # joints_norm_per_frame = np.array(pose[-1]).astype(np.str)
            # data_raw.append(joints_norm_per_frame[0:28])
            # #print(joints_norm_per_frame[0:28])
            # iii=iii+1
            # print(iii)
###############   08为训练数据输出
print('08:')
# data = pd.DataFrame(data_raw)
# writer = pd.ExcelWriter('people1.xlsx')		# 写入Excel文件
# data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
# writer.save()
# writer.close()
###############     09为视频输出
print('09:')
video_writer.release()
cap.release()
# f.close()
