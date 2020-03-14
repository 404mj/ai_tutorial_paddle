import paddle as paddle
import paddle.fluid as fluid
import matplotlib.pyplot as plt
import os

# Step1：准备数据
BUF_SIZE = 512
BATCH_SIZE = 128
EPOCH_NUM = 20
train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.mnist.train(),
                          BUF_SIZE),
    batch_size=BATCH_SIZE
)

test_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.mnist.test(),
                          BUF_SIZE),
    batch_size=BATCH_SIZE
)

# train_data=paddle.dataset.mnist.train()
# ndata = next(train_data())
# print(ndata)


# Step2.网络配置
'''
定义一个简单的多层感知器，一共有三层，两个大小为100的隐层和一个大小为10的输出层，
(为啥是100？？就这么定义的，你也可以定义为200或者50啊)
因为MNIST数据集是手写0到9的灰度图像，类别有10个，所以最后的输出大小是10。最后输出层的激活函数是Softmax，
所以最后的输出层相当于一个分类器。加上一个输入层的话，多层感知器的结构是：输入层-->>隐层-->>隐层-->>输出层。
'''


# 定义多层感知器
def multilayer_perceptron(input):
    hidden1 = fluid.layers.fc(input=input, size=100, act='relu')
    hidden2 = fluid.layers.fc(input=hidden1, size=100, act='relu')
    prediction = fluid.layers.fc(input=hidden2, size=10, act='softmax')
    return prediction


# 定义输入输出层
image = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')  # 单通道，28*28像素值
label = fluid.layers.data(name='label', shape=[1], dtype='int64')  # 图片标签
# 获取分类器
predict = multilayer_perceptron(image)

# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=predict, label=label)  # 使用交叉熵损失函数,描述真实样本标签和预测概率之间的差值
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=predict, label=label)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)  # 使用Adam算法进行优化
opts = optimizer.minimize(avg_cost)

# Step3.模型训练 and Step4.模型评估
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

# 展示模型训练曲线
all_train_iter = 0
all_train_iters = []
all_train_costs = []
all_train_accs = []


def show_train_process(title, iters, costs, accs, label_cost, lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("cost/acc", fontsize=20)
    plt.plot(iters, costs, color='red', label=label_cost)
    plt.plot(iters, accs, color='green', label=lable_acc)
    plt.legend()
    plt.savefig("hw_procese.jpg")
    # plt.grid()
    # plt.show()


model_save_dir = "./hand.inference.model"
for pass_id in range(EPOCH_NUM):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):  # 遍历train_reader
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),  # 运行主程序
                                        feed=feeder.feed(data),  # 给模型喂入数据
                                        fetch_list=[avg_cost, acc])  # fetch 误差、准确率

        all_train_iter = all_train_iter + BATCH_SIZE
        all_train_iters.append(all_train_iter)
        all_train_costs.append(train_cost[0])
        all_train_accs.append(train_acc[0])

        # 每100个batch打印一次信息  误差、准确率
        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

    # 进行测试
    test_accs = []
    test_costs = []
    # 每训练一轮 进行一次测试
    for batch_id, data in enumerate(test_reader()):  # 遍历test_reader
        test_cost, test_acc = exe.run(program=fluid.default_main_program(),  # 执行训练程序
                                      feed=feeder.feed(data),  # 喂入数据
                                      fetch_list=[avg_cost, acc])  # fetch 误差、准确率
        test_accs.append(test_acc[0])  # 每个batch的准确率
        test_costs.append(test_cost[0])  # 每个batch的误差

    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))  # 每轮的平均误差
    test_acc = (sum(test_accs) / len(test_accs))  # 每轮的平均准确率
    print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))

# 保存模型
# 如果保存路径不存在就创建
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
print('save models to %s' % (model_save_dir))
fluid.io.save_inference_model(model_save_dir,  # 保存推理model的路径
                              ['image'],  # 推理（inference）需要 feed 的数据
                              [predict],  # 保存推理（inference）结果的 Variables
                              exe)  # executor 保存 inference model
show_train_process("training", all_train_iters, all_train_costs, all_train_accs, "trainning cost", "trainning acc")
