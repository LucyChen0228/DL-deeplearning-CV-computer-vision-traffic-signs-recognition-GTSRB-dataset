# DL-deeplearning-CV-computer-vision-traffic-signs-recognition-GTSRB-dataset
Keras &amp; CNN


【数据集】：德国GTSRB数据集，由于官网为ppm 格式，不太好进行操作，感谢一些前辈提供了pickle 格式的数据集
【网络结构】：卷积神经网络VGG16 + 自己在最后多加了两层
【图像预处理部分】：采用了IMAGE NET 竞赛中常用的图像去均值操作
【模型的保存与权重读取】：根据KERAS 框架中的model 函数进行保存，调用，且读取权重。



【使用：model.summary() 所打印出的模型的概况】

"C:\Program Files\Anaconda3\python.exe" "C:/Users/c8780/Desktop/traffic signs/查看保存好的MODEL.py"
C:\Program Files\Anaconda3\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
2018-11-19 14:24:33.577698: I T:\src\github\tensorflow\tensorflow\core\platform\cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 224, 224, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 25088)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 32)                802848    
_________________________________________________________________
dropout_1 (Dropout)          (None, 32)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 43)                1419      
=================================================================
Total params: 15,518,955
Trainable params: 15,518,955
Non-trainable params: 0
_________________________________________________________________

Process finished with exit code 0



【使用文件：返回权重张量列表.py】
block1_conv1/kernel:0 (3, 3, 3, 64)
block1_conv1/bias:0 (64,)
block1_conv2/kernel:0 (3, 3, 64, 64)
block1_conv2/bias:0 (64,)
block2_conv1/kernel:0 (3, 3, 64, 128)
block2_conv1/bias:0 (128,)
block2_conv2/kernel:0 (3, 3, 128, 128)
block2_conv2/bias:0 (128,)
block3_conv1/kernel:0 (3, 3, 128, 256)
block3_conv1/bias:0 (256,)
block3_conv2/kernel:0 (3, 3, 256, 256)
block3_conv2/bias:0 (256,)
block3_conv3/kernel:0 (3, 3, 256, 256)
block3_conv3/bias:0 (256,)
block4_conv1/kernel:0 (3, 3, 256, 512)
block4_conv1/bias:0 (512,)
block4_conv2/kernel:0 (3, 3, 512, 512)
block4_conv2/bias:0 (512,)
block4_conv3/kernel:0 (3, 3, 512, 512)
block4_conv3/bias:0 (512,)
block5_conv1/kernel:0 (3, 3, 512, 512)
block5_conv1/bias:0 (512,)
block5_conv2/kernel:0 (3, 3, 512, 512)
block5_conv2/bias:0 (512,)
block5_conv3/kernel:0 (3, 3, 512, 512)
block5_conv3/bias:0 (512,)
dense_1/kernel:0 (25088, 32)
dense_1/bias:0 (32,)
dense_2/kernel:0 (32, 43)
dense_2/bias:0 (43,)

【一些小小的思考：】
1.pickle 格式的文本，能否更直观的进行查看
2.imagenet 中所保存的权重，如何去评估其在不同案例中的性能？











