# Protective-Wearing-and-Action-Recognition-Detection-Based-on-Pose-Estimation
#1.Project Introduction  
    This project realizes the use of protective gear wearing detection (safety helmet, safety vest), behavior recognition (recognition of several human actions), and the fusion of protective gear wearing detection and behavior recognition.    
    The protective gear wearing detection uses the YoloV3 model to realize the protective gear detection. In the behavior recognition algorithm, this project adds a pose estimation algorithm (openpose), and uses a deep neural network to classify the pose nodes to achieve behavior recognition. In this project, posture information is added to the protective gear wearing detection algorithm to remove the misjudgment information of the protective gear and greatly improve the accuracy of the protective gear wearing detection.    
#2.Environment configuration  
  The experimental environment configuration of this project is as follows: (This project uses gpu acceleration, so all libraries use the gpu version)    
  Keras-gpu           2.2.0  
  glouncv             0.6.0  
  opencv              4.0.0  
  python              3.6.0  
  tensorflow-gpu      1.14.0  
  mxnet-cu100         1.5.0  
  numpy               1.16.5  
  scikit-learn        0.23.1  
 
#3.File introduction  
   main.py is the main function, which realizes the parallel recognition of protective gear wearing detection and behavior recognition   
   main_action.pyis a behavior recognition algorithm, which implements pose estimation and behavior recognition   
   main_helmet.pydetects and recognizes the wearing of protective gear, and realizes the detection of wearing protective gear  
   The Pose folder contains the implementation method of Openpose pose estimation. For specific principles, please refer to HowNet.  
   The training folder is the training method of the behavior recognition network, where data.csv is the data set, collected by the laboratory members of this project, and train.py is the training function     
#4.How to run  
   Run main.py, main_action.py, and main_hemet.py respectively to execute the program. Please make sure that the path in the program is the correct path before executing the file.  
#5.Precautions  
    The network weight of behavior recognition has been uploaded to the project as the file training/framewise_recognition.h5.  
    The network weight of the protective gear wearing detection is too large and cannot be uploaded. If you need to use it, please leave a message.    
#6.Experimental result   
    The real-time monitoring of this project on the road, as shown in the figure below    
![图12-e](https://user-images.githubusercontent.com/55353772/110061178-17a81f00-7da2-11eb-884e-b907d452dadf.png)
