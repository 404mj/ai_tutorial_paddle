import paddle as paddle
import paddle.fluid as fluid
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

'''
模拟LeNet-5  视频中是写法不懂啊！！
截图在wechat 文件传输助手中了！！！
'''

# 网络结构
img = fluid.layers.data(name="pixel", type=paddle.data_type.dense_vector(784))

conv_pool_1 = fluid.nets.simple_img_conv_pool(
    input=img,
    filter_size=5,
    num_filters=20,
    num_channel=1,
    pool_size=2,
    pool_stride=2,
    # act=fluid.layers.relu()
    act='relu'
)

conv_pool_2 = fluid.nets.simple_img_conv_pool(
    input=conv_pool_1,
    filter_size=5,
    num_channel=20,
    num_filters=50,
    pool_size=2,
    pool_stride=2,
    act='relu'
)

predict = fluid.layers.fc(
    input=conv_pool_2,
    size=10,
    act='softmax'
)


# 损失函数  反向传播 正则化
def get_trainer():
    label = fluid.layers.data(name='label', shape=[10], dtype='int')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    optimizer = fluid.optimizer.MomentumOptimizer(
        learning_rate=0.1 / 128.0,
        momentum=0.9,
        regularization=fluid.regularizer.L2DecayRegularizer(0.0005 * 128)
    )
