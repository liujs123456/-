# -
基于YOLOv1框架的关于无人驾驶的神经网络目标识别模型

我们的项目聚焦于算法开发，核心成果在于构建并训练一个高效的神经网络模型。该模型的设计初衷是具备广泛的物体识别能力，不仅限于车辆，而是涵盖了诸如飞机、自行车、鸟类、船只、瓶子、公交车、汽车、猫、椅子、牛、餐桌、狗、马、摩托车、人、盆栽植物、绵羊、沙发、火车及电视显示器等多样化的VOC（Visual Object Classes）类别。通过精细的训练过程，该神经网络能够学习并识别图片中的这些物体，展现出强大的泛化能力。

为了进一步提升模型在实际应用中的表现，我们集成了OpenCV这一强大的图像处理库，并结合YOLO（You Only Look Once）模型，专门优化了对火车轨道周边行人安全检测的场景。这一整合不仅提升了检测的准确性和实时性，还使得整个系统能够直接应用于实际环境中，如火车站或铁路沿线，有效预警潜在的行人安全风险。

项目成果不仅是一个高度专业化的神经网络模型，更是一个包含图像预处理、模型训练、目标检测及结果可视化等完整流程的学习案例。无论是对于计算机视觉领域的研究者，还是希望将AI技术应用于实际安全监控项目的开发者而言，本项目都提供了宝贵的实践经验和参考价值。

# 配置环境

1. 创建虚拟环境：conda create -n yolo python=3.6

2. 激活目标环境：conda activate yolo

3. 安装所有依赖包：pip install -r requirements.txt 

# 文件说明

1. train.py：使用数据集训练模型，默认使用CPU进行测试，会耗费大量时间，如果使用GPU训练需要安装pytorch的gpu版本

2. test.py：在数据集上对模型进行测试，默认使用CPU进行测试

3. test_costom.py：检测costom文件夹中的自定义图片，并保存到det_results/custom文件夹中,。
    需要检测图片时，只需要把图片放入到costom文件夹中，运行test_costom.py即可

4. models/yolo.py：定义了YOLO网络结构，需要重点学习，知道大概的模型结果以及预测结果是什么

5. weight文件下存放训练好的模型参数，可以直接使用训练好的./weights/voc/yolo/yolo_69.6.pth模型进行预测

6. run_rail.py: 对火车轨道进行行人安全检测，输入为待检测视频(Rail.mp4)，输出为检测后的处理视频(Rail_detected.mp4)


# 训练数据集准备

1. 模型使用PASCAL VOC数据集进行训练，使用命令下载（或者复制网址到浏览器下载）
    curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

2. 数据集解压后生成VOCdevkit文件夹，并放在当前项目目录下
   
———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#-
   Neural Network Object Recognition Model for Autonomous Driving Based on YOLOv1 Framework
   
   Our project focuses on algorithm development, with a core achievement being the construction and training of an efficient neural network model. Designed with the intention of possessing extensive object recognition capabilities, this model goes beyond vehicles to encompass a diverse range of VOC (Visual Object Classes) categories such as aeroplanes, bicycles, birds, boats, bottles, buses, cars, cats, chairs, cows, dining tables, dogs, horses, motorbikes, persons, potted plants, sheep, sofas, trains, and tv monitors. Through meticulous training, this neural network learns to identify these objects in images, exhibiting robust generalization abilities.

   To further enhance the model's performance in practical applications, we integrated OpenCV, a powerful image processing library, with the YOLO (You Only Look Once) model, specifically optimizing it for pedestrian safety detection scenarios around railway tracks. This integration not only improves detection accuracy and real-time performance but also enables the entire system to be directly applied in real-world environments such as train stations or railway lines, effectively alerting potential pedestrian safety risks.

   The project outcome is not merely a highly specialized neural network model but also a comprehensive learning case encompassing image preprocessing, model training, object detection, and result visualization. It offers invaluable practical experience and reference value for both researchers in the field of computer vision and developers looking to apply AI technology to real-world security monitoring projects.

# Environment Setup
Create a virtual environment: conda create -n yolo python=3.6
Activate the target environment: conda activate yolo
Install all dependencies: pip install -r requirements.txt

# File Description
train.py: Trains the model using a dataset. By default, it tests using the CPU, which can be time-consuming. For GPU training, install the GPU version of PyTorch.
test.py: Evaluates the model on a dataset, using the CPU by default.
test_custom.py: Detects objects in custom images placed in the "custom" folder and saves the results to the "det_results/custom" folder. Simply place images in the "custom" folder and run test_custom.py for detection.
models/yolo.py: Defines the YOLO network architecture. It's essential to understand the model's structure and prediction outcomes.
weights: Stores the trained model parameters. You can directly use the pre-trained ./weights/voc/yolo/yolo_69.6.pth model for predictions.
run_rail.py: Performs pedestrian safety detection on railway tracks. Takes a video file (Rail.mp4) as input and outputs a processed video (Rail_detected.mp4).

# Training Dataset Preparation
The model is trained using the PASCAL VOC dataset, which can be downloaded using the following commands (or copy the URLs into your browser):
bash
curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar  
curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar  
curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
After downloading, extract the datasets to generate the VOCdevkit folder, which should be placed in the current project directory.
