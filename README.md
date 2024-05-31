# DL_midterm_task2
神经网络和深度学习期中作业任务2：
在VOC数据集上训练并测试目标检测模型Faster R-CNN和YOLO V3

基本要求：
（1） 学习使用现成的目标检测框架——如mmdetection或detectron2——在VOC数据集上训练并测试目标检测模型Faster R-CNN和YOLO V3；
（2） 挑选4张测试集中的图像，通过可视化对比训练好的Faster R-CNN第一阶段产生的proposal box和最终的预测结果。
（3） 搜集三张不在VOC数据集内包含有VOC中类别物体的图像，分别可视化并比较两个在VOC数据集上训练好的模型在这三张图片上的检测结果（展示bounding box、类别标签和得分）；


本实验利用目标检测框架mmdetection在VOC数据集上训练并测试目标检测模型Faster R-CNN和YOLO V3。

1. `config/`

    该目录下有适应mmdetection v3.3版本的训练config文件。安装好mmdetection后，运行`python tools/train.py configs/faster-rcnn_r50_fpn_1x_voc0712.py`即可完成训练和测试.

2. `test_image/`

    该目录下有VOC07test中4个样本和搜寻的不在VOC数据集中的4张图片，可以用来测试两种算法的检测效果。

3. `vis.ipynb`

    可以利用相关代码块可视化对比训练好的Faster R-CNN第一阶段产生的proposal box和最终的预测结果、对比Faster R-CNN和YOLOv3的预测结果。

4. 相关模型权重保存在：<a href="https://drive.google.com/drive/folders/16UlK6v13bD3zRXF0B4corOPPmYpAXXqj?usp=sharing">google drive</a>