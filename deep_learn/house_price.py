# https://aistudio.baidu.com/aistudio/projectDetail/238429
# 房价预测
import paddle.fluid as fluid
import paddle
import numpy as np
import os
import matplotlib.pyplot as plt

# 准备数据 缓存大小和批量处理大小
BUF_SIZE = 50
BATCH_SIZE = 20
EPOCH_NUM = 50

train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.uci_housing.train(),
                          BUF_SIZE),
    batch_size=BATCH_SIZE
)

test_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.uci_housing.test(),
                          BUF_SIZE),
    batch_size=BATCH_SIZE
)

'''
1、训练数据14列，前13列为信息，最后一列为价格，所以在这个模型里面，前13为输入x
价格为y，对应到准备好的数据上

2、线性回归为y=ax+b,这里就是z=a1x1+a2x2...+b,网络结构就清晰了，
输入，输出，全连接
'''
x = fluid.layers.data(name='x', shape=[13], dtype='float32')
y = fluid.layers.data(name='y', shape=[1], dtype='float32')

y_predict = fluid.layers.fc(input=x, size=1, act=None)

# 损失函数,均方差
cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_cost = fluid.layers.mean(cost)

# 参数优化方案
optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01)
opts = optimizer.minimize(avg_cost)

test_program = fluid.default_main_program().clone(for_test=True)

# 开始执行训练和评估
use_cuda = False  # use_cuda为False,表示运算场所为CPU;use_cuda为True,表示运算场所为GPU
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)  # 创建一个Executor实例exe
exe.run(fluid.default_startup_program())  # Executor的run()方法执行startup_program(),进行参数初始化

feeder = fluid.DataFeeder(place=place, feed_list=[x, y])

# 绘制训练过程
iter = 0
iters = []
train_costs = []
def draw_train_process(iters,train_costs):
    title="training cost"
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=14)
    plt.ylabel("cost", fontsize=14)
    plt.plot(iters, train_costs,color='red',label='training cost')
    plt.savefig("a01.jpg")
    # plt.grid()
    # plt.show()


# 训练并保存模型
model_save_dir = "./uci_housing_inference.model"
for pass_id in range(EPOCH_NUM):
    train_cost = 0
    for batch_id, data in enumerate(train_reader()):
        train_cost = exe.run(fluid.default_main_program(),
                             feed=feeder.feed(data),
                             fetch_list=[avg_cost]
                             )
        if batch_id % 40 == 0:
            print("Pass:%d, Cost:%0.5f" % (pass_id, train_cost[0][0]))
            iter = iter + BATCH_SIZE
            iters.append(iter)
            train_costs.append(train_cost[0][0])

    # 开始测试并输出最后一个batch的损失值
    test_cost = 0
    for batch_id, data in enumerate(test_reader()):  # 遍历test_reader迭代器
        test_cost = exe.run(program=test_program,  # 运行测试cheng
                            feed=feeder.feed(data),  # 喂入一个batch的测试数据
                            fetch_list=[avg_cost])  # fetch均方误差
    print('Test:%d, Cost:%0.5f' % (pass_id, test_cost[0][0]))  # 打印最后一个batch的损失值

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
print('save models to %s' % (model_save_dir))
# 保存训练参数到指定路径中，构建一个专门用预测的program
fluid.io.save_inference_model(model_save_dir,  # 保存推理model的路径
                              ['x'],  # 推理（inference）需要 feed 的数据
                              [y_predict],  # 保存推理（inference）结果的 Variables
                              exe)  # exe 保存 inference model
draw_train_process(iters, train_costs)

# 模型预测。
