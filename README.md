# Protective-Wearing-and-Action-Recognition-Detection-Based-on-Pose-Estimation
#1.项目介绍  
    本项目实现了护具佩戴检测（安全帽，安全背心），行为识别（几种人体动作的识别），以及护具佩戴检测和行为识别的融合使用。  
    护具佩戴检测是利用YoloV3模型，实现对护具的检测。而行为识别算法中，本项目加入了姿态估计算法（openpose）,并使用深度神经网络对姿态节点进行分类，从而实现行为识别。本项目将姿态信息加入到护具佩戴检测算法中，去除护具的误判信息，大大提高了护具佩戴检测的准确率。  
#2.环境配置  
  本项目实验环境配置如下：（本项目使用gpu加速，因此各种库均使用gpu版本）  
  Keras-gpu  2.2.0  
  glouncv    0.6.0  
  opencv     4.0.0  
  python     3.6.0  
  tensorflow-gpu 1.14.0  
  mxnet-cu100 1.5.0  
  numpy       1.16.5  
  scikit-learn 0.23.1  
 
#3.运行方法  
