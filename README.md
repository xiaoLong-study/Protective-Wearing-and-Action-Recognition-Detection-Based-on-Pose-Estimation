# Protective-Wearing-and-Action-Recognition-Detection-Based-on-Pose-Estimation
#1.项目介绍  
    本项目实现了护具佩戴检测（安全帽，安全背心），行为识别（几种人体动作的识别），以及护具佩戴检测和行为识别的融合使用。  
    护具佩戴检测是利用YoloV3模型，实现对护具的检测。而行为识别算法中，本项目加入了姿态估计算法（openpose）,并使用深度神经网络对姿态节点进行分类，从而实现行为识别。本项目将姿态信息加入到护具佩戴检测算法中，去除护具的误判信息，大大提高了护具佩戴检测的准确率。  
#2.环境配置  
  本项目实验环境配置如下：（本项目使用gpu加速，因此各种库均使用gpu版本）  
  Keras-gpu           2.2.0  
  glouncv             0.6.0  
  opencv              4.0.0  
  python              3.6.0  
  tensorflow-gpu      1.14.0  
  mxnet-cu100         1.5.0  
  numpy               1.16.5  
  scikit-learn        0.23.1  
 
#3.文件介绍  
   main.py为主函数，实现了护具佩戴检测和行为识别的并行识别  
   main_action.py为行为识别算法，实现了姿态估计和行为识别  
   main_helmet.py为护具佩戴检测识别，实现了护具佩戴检测  
   Pose文件夹中为Openpose姿态估计的实行方法 ，具体原理实现可以参考知网  
   training文件夹为行为识别网络的训练方法，其中data.csv为数据集，由本项目实验室成员采集，train.py为训练函数  
#4.运行方法  
   分别运行main.py、main_action.py、main_hemet.py均可以执行程序，注意在执行文件前保证程序中的路径为正确路径。  
#5.注意事项  
    行为识别的网络权重已经上传到项目中，为文件training/framewise_recognition.h5文件。  
    护具佩戴检测的网络权重较大无法上传，如需使用请留言。  
#6.实验结果  
    本项目在道路的实时监测，如下图所示  
![图12-e](https://user-images.githubusercontent.com/55353772/110061178-17a81f00-7da2-11eb-884e-b907d452dadf.png)
