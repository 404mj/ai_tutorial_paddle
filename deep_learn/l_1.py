import paddle.fluid as fluid

# Program
a = fluid.layers.fill_constant([2,2],'int64',1)
b = fluid.layers.fill_constant([2,2],'int64',1)
res = fluid.layers.elementwise_add(a,b)
# print(res)

# 定义Executor
place = fluid.CPUPlace()
exec = fluid.Executor(place)
exec.run(fluid.default_startup_program())
myres = exec.run(
    program=fluid.default_main_program(),
    fetch_list=[res]
)
print(myres)

