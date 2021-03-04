# -*- coding: UTF-8 -*-
import cv2 as cv
import argparse
import numpy as np
import tensorflow as tf
import time
from utils import choose_run_mode, load_pretrain_model, set_video_writer
from Pose.pose_visualizer import TfPoseVisualizer
from Action.recognizer import load_action_premodel, framewise_recognize

#import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "/gpu:0"
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True


import pandas as pd


from enum import Enum
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import load_model
from collections import Counter
import matplotlib.pyplot as plt
from keras.callbacks import Callback
import itertools
from sklearn.metrics import confusion_matrix





class Actions(Enum):
    # framewise_recognition.h5
    # squat = 0
    # stand = 1
    # walk = 2
    # wave = 3
    #sit = 0
    stand = 0
    wave = 1
    call = 2
    pick = 3
    #fall = 6
    # framewise_recognition_under_scene.h5
    # stand = 0
    # walk = 1
    # operate = 2
    # fall_down = 3
    # run = 4


# Callback class to visialize training progress
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# load data
#raw_data = pd.read_csv('data_with_scene.csv', header=0)
raw_data = pd.read_csv('Action/training/data.csv', header=0)
dataset = raw_data.values
X = dataset[:, 0:28].astype(float)   #所有行，0-36列都读进来
Y = dataset[:, 28]                 #所有行，第36列读进来，36列为动作标签
result = Counter(Y)
print(result)      #计算某个值出现的次数
print(X.shape[0])
print(Y.shape[0])
# X = dataset[0:3289, 0:36].astype(float)  # 忽略run数据
# Y = dataset[0:3289, 36]

# 将类别编码为数字
# encoder = LabelEncoder()
# encoder_Y = encoder.fit_transform(Y)
# print(encoder_Y[0], encoder_Y[900], encoder_Y[1800], encoder_Y[2700])
# encoder_Y = [0]*744 + [1]*722 + [2]*815 + [3]*1008 + [4]*811
encoder_Y = [0]*result['stand'] + [1]*result['wave']+[2]*result['call']+[3]*result['pick']
print(len(encoder_Y))
print(encoder_Y[0])
# one hot 编码
dummy_Y = np_utils.to_categorical(encoder_Y)
print(dummy_Y.shape[0])

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_Y, test_size=0.1, random_state=9)
#test_size=0.1表示所有数据中有10%的数据作为验证集

# build keras model
action_classifier = Sequential()
action_classifier.add(Dense(units=128, activation='relu'))
action_classifier.add(BatchNormalization())
action_classifier.add(Dense(units=64, activation='relu'))
action_classifier.add(BatchNormalization())
action_classifier.add(Dense(units=16, activation='relu'))
action_classifier.add(BatchNormalization())
action_classifier.add(Dense(units=4, activation='softmax'))  # units = nums of classes

# training
his = LossHistory()
action_classifier.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
action_classifier.fit(X_train, Y_train, batch_size=32, epochs=0, verbose=1, validation_data=(X_test, Y_test), callbacks=[his])
#model.fit  训练数据
action_classifier.summary()
his.loss_plot('epoch')



############# 01-05为action recognition 初始化部分
data_raw = [];     ###储存训练数据

print('01:')
parser = argparse.ArgumentParser(description='Action Recognition by OpenPose')
parser.add_argument('--video', default = "E:/LSETM/程序/Online-Realtime-Action-Recognition-based-on-OpenPose/test_out/yolov3111.mp4",
                   help='Path to video file.')          ####输入视频
# parser.add_argument('--video', help='Path to video file.')    ####输入实时数据（摄像头）
args = parser.parse_args()

print('02:')
# 导入相关模型（openpose网络模型）  两种：VGG_origin 和mobilenet_thin
estimator = load_pretrain_model('mobilenet_thin')
# 行为识别网络数据读入
action_classifier.load_weights('Action/framewise_recognition.h5',by_name=True)
# action_classifier = load_action_premodel('Action/framewise_recognition.h5')

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
video_writer = set_video_writer(cap, write_fps=int(7))

print('05:')
# # 保存关节数据的txt文件，用于训练过程(for training)
#f = open('origin_data.txt', 'a+')
iii=0
jjj=0
#########################################################################################



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
############ action recognition主循环 ##################
            # pose estimation
            humans,out= estimator.inference(show)


            # frameHeight, frameWidth = show.shape[:2]
            # ff=plt.figure(figsize=[80, 20])
            # f1=ff.add_subplot(141)
            # f2=ff.add_subplot(142)
            # f3=ff.add_subplot(143)
            # f4=ff.add_subplot(144)
            # f1.imshow(cv.cvtColor(show, cv.COLOR_BGR2RGB))
            # f1.imshow(cv.resize(out[0, :, :, 10], (frameWidth, frameHeight)), alpha=0.6)
            # f1.axis("off")
            #
            # f2.imshow(cv.cvtColor(show, cv.COLOR_BGR2RGB))
            # f2.imshow(cv.resize(out[0, :, :, 18], (frameWidth, frameHeight)), alpha=0.6)
            # f2.axis("off")
            # f3.imshow(cv.cvtColor(show, cv.COLOR_BGR2RGB))
            # f3.imshow(cv.resize(out[0, :, :, 31], (frameWidth, frameHeight)), alpha=0.6)
            # f3.axis("off")
            #
            # f4.imshow(cv.cvtColor(show, cv.COLOR_BGR2RGB))
            # f4.imshow(cv.resize(out[0, :, :, 39], (frameWidth, frameHeight)), alpha=0.6)
            # f4.axis("off")
            # plt.show()
            # ff.savefig('可视化-guo.png')

            # get pose info
            pose = TfPoseVisualizer.draw_pose_rgb(show, humans)  # return frame, joints, bboxes, xcenter
            #pose[0]=255
            #print(type(pose[0]))
            # recognize the action framewise
            show = framewise_recognize(pose, action_classifier)
            height, width = show.shape[:2]
            if (time.time() - start_time) > fps_interval:
                # 计算这个interval过程中的帧数，若interval为1秒，则为FPS
                realtime_fps = fps_count / (time.time() - start_time)
                fps_count = 0  # 帧数清零
                start_time = time.time()
            fps_label = 'FPS:{0:.2f}'.format(realtime_fps)
            cv.putText(show, fps_label, (160, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)#在图片的指定位置写字
            # 显示检测到的人数
            num_label = "Human: {0}".format(len(humans))
            cv.putText(show, num_label, (5, height-45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # 显示目前的运行时长及总帧数
            if frame_count == 1:
                run_timer = time.time()
            run_time = time.time() - run_timer
            time_frame_label = '[Time:{0:.2f} | Frame:{1}]'.format(run_time, frame_count)
            cv.putText(show, time_frame_label, (5, height-15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv.imshow('Action Recognition based on OpenPose', show)
            #gcv.utils.viz.cv_plot_image(show)
            video_writer.write(show)

            #采集数据，用于训练过程(for training)
            joints_norm_per_frame = np.array(pose[-1]).astype(np.str)
            data_raw.append(joints_norm_per_frame[0:28])
            #print(joints_norm_per_frame[0:28])
            iii=iii+1
            print(iii)

###############   08为训练数据输出
print('08:')
data = pd.DataFrame(data_raw)
writer = pd.ExcelWriter('call.xlsx')		# 写入Excel文件
data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
writer.save()
writer.close()
###############     09为视频输出
print('09:')
video_writer.release()
cap.release()
# f.close()
