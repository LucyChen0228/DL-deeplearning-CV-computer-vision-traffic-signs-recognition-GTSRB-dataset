# DL-deeplearning-CV-computer-vision-traffic-signs-recognition-GTSRB-dataset
Keras &amp; CNN


【数据集】：德国GTSRB数据集，由于官网为pickle 格式，不太好进行操作，感谢一些前辈提供了后缀为.h的数据集
【网络结构】：卷积神经网络VGG16 + 自己在最后多加了两层
【图像预处理部分】：采用了IMAGE NET 竞赛中常用的图像去均值操作
【模型的保存与权重读取】：根据KERAS 框架中的model 函数进行保存，调用，且读取权重。
