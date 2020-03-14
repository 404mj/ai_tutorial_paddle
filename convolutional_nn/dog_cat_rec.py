# 猫狗识别 CNN
# https://aistudio.baidu.com/aistudio/projectdetail/47368

import paddle as paddle
import paddle.fluid as fluid
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# step1.数据准备
BATCH_SIZE = 128
train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.cifar.train10(),
        buf_size=BATCH_SIZE * 100),
    batch_size=BATCH_SIZE
)
test_reader = paddle.batch(
    paddle.dataset.cifar.test10(),
    batch_size=BATCH_SIZE
)


# Step2.网络配置
def cnn(img):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act='relu'
    )

    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_stride=2,
        pool_size=2,
        act='relu'
    )
    prediction = fluid.layers.fc(
        input=conv_pool_2,
        size=10,
        act='softmax'
    )
    return prediction


# 定义输入输出
data_shape = [3, 32, 32]
images = fluid.layers.data(name='images', shape=data_shape, dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
predict = cnn(images)

# 定义损失优化等
cost = fluid.layers.cross_entropy(input=predict, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=predict, label=label)
optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(avg_cost)

# Step3.模型训练
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
feeder = fluid.DataFeeder(feed_list=[images, label], place=place)
# Step4.模型评估

iter = 0
iters = []
train_costs = []
train_accs = []


def draw_train_process(iters, train_costs, train_accs):
    title = "training costs/training accs"
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=14)
    plt.ylabel("cost/acc", fontsize=14)
    plt.plot(iters, train_costs, color='red', label='training costs')
    plt.plot(iters, train_accs, color='green', label='training accs')
    plt.legend()
    plt.grid()
    plt.show()


EPOCH_NUM = 1
model_save_dir = "/home/aistudio/data/catdog.inference.model"

for pass_id in range(EPOCH_NUM):
    # 开始训练
    train_cost = 0
    for batch_id, data in enumerate(train_reader()):  # 遍历train_reader的迭代器，并为数据加上索引batch_id
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),  # 运行主程序
                                        feed=feeder.feed(data),  # 喂入一个batch的数据
                                        fetch_list=[avg_cost, acc])  # fetch均方误差和准确率
        if batch_id % 100 == 0:  # 每100次batch打印一次训练、进行一次测试
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))
        iter = iter + BATCH_SIZE
        iters.append(iter)
        train_costs.append(train_cost[0])
        train_accs.append(train_acc[0])
    # 开始测试
    test_costs = []  # 测试的损失值
    test_accs = []  # 测试的准确率
    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=fluid.default_main_program(),  # 运行测试程序
                                      feed=feeder.feed(data),  # 喂入一个batch的数据
                                      fetch_list=[avg_cost, acc])  # fetch均方误差、准确率
        test_costs.append(test_cost[0])  # 记录每个batch的误差
        test_accs.append(test_acc[0])  # 记录每个batch的准确率
    test_cost = (sum(test_costs) / len(test_costs))  # 计算误差平均值（误差和/误差的个数）
    test_acc = (sum(test_accs) / len(test_accs))  # 计算准确率平均值（ 准确率的和/准确率的个数）
    print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, test_cost, test_acc))

    # 保存模型
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    fluid.io.save_inference_model(model_save_dir,
                                  ['images'],
                                  [predict],
                                  exe)
print('训练模型保存完成！')
draw_train_process(iters, train_costs, train_accs)
# Step5.模型预测
infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()


def load_image(file):
    im = Image.open(file)
    im = im.resize((32, 32), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32)
    im = im.transpose((2, 0, 1))
    im = im / 255.0
    im = np.expand_dims(im, axis=0)
    print('im_shape的维度：', im.shape)
    return im


# 开始预测
with fluid.scope_guard(inference_scope):
    [inference_program,  # 预测用的program
     feed_target_names,  # 是一个str列表，它包含需要在推理 Program 中提供数据的变量的名称。
     fetch_targets] = fluid.io.load_inference_model(model_save_dir,  # fetch_targets：是一个 Variable 列表，从中我们可以得到推断结果。
                                                    infer_exe)  # infer_exe: 运行 inference model的 executor

    infer_path = '/home/aistudio/data/dog.png'
    img = Image.open(infer_path)
    plt.imshow(img)
    plt.show()

    img = load_image(infer_path)

    results = infer_exe.run(inference_program,  # 运行预测程序
                            feed={feed_target_names[0]: img},  # 喂入要预测的img
                            fetch_list=fetch_targets)  # 得到推测结果
    print('results', results)
    label_list = [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
        "ship", "truck"
    ]
    print("infer results: %s" % label_list[np.argmax(results)[0]])
