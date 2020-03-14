import numpy as np
from PIL import Image
import paddle as paddle
import paddle.fluid as fluid
import matplotlib.pyplot as plt
import os

'''
图片预处理
在预测之前，要对图像进行预处理。
首先进行灰度化，然后压缩图像大小为28*28，接着将图像转换成一维向量，最后再对一维向量进行归一化处理。
'''


def load_image(file):
    img = Image.open(file).convert('L')  # 将RGB转化为灰度图像，L代表灰度图像，像素值在0~255之间
    img = img.resize((28, 28, Image.ANTIALIAS))  # resize image with high-quality 图像大小为28*28
    img = np.array(img).reshape(1, 1, 28, 28).astype(np.float32)
    print(img)
    img = img / 255.0 * 2.0 - 1.0  # 归一化到【-1~1】之间
    return img


infer_path = './infer_3.png'
img = Image.open(infer_path)
plt.imshow(img)  # 根据数组绘制图像
plt.show()  # 显示图像

place = fluid.CPUPlace()
infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()
model_save_dir = "./hand.inference.model"

# 加载数据并开始预测
with fluid.scope_guard(inference_scope):
    # 获取训练好的模型
    # 从指定目录中加载 推理model(inference model)
    [inference_program,  # 推理Program
     feed_target_names,  # 是一个str列表，它包含需要在推理 Program 中提供数据的变量的名称。
     fetch_targets] = fluid.io.load_inference_model(model_save_dir,
                                                    # fetch_targets：是一个 Variable 列表，从中我们可以得到推断结果。model_save_dir：模型保存的路径
                                                    infer_exe)  # infer_exe: 运行 inference model的 executor
    infer_path = '/home/aistudio/data/data5286/infer_3.png'
    img = load_image(infer_path)

    results = infer_exe.run(program=inference_program,  # 运行推测程序
                            feed={feed_target_names[0]: img},  # 喂入要预测的img
                            fetch_list=fetch_targets)  # 得到推测结果,
    # 获取概率最大的label
    lab = np.argsort(results)  # argsort函数返回的是result数组值从小到大的索引值
    # print(lab)
    print("该图片的预测结果的label为: %d" % lab[0][0][-1])  # -1代表读取数组中倒数第一列
