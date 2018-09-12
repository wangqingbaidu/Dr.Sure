# tf.identity && tf.control_dependencies
最近看到有些TensorFlow的代码中使用到了`tf.control_dependencies`、`tf.identity`操作，在Stack Overflow上看到一个很好的解释，原地址：[https://stackoverflow.com/questions/34877523/in-tensorflow-what-is-tf-identity-used-for](https://stackoverflow.com/questions/34877523/in-tensorflow-what-is-tf-identity-used-for)

下面是关于这两个函数用途的介绍：

## `tf.identity` 

`tf.identity` is useful when you want to explicitly transport tensor between devices (like, from GPU to a CPU). The op adds send/recv nodes to the graph, which make a copy when the devices of the input and the output are different.

换句话说，`tf.identity`是为了显示地转化Tensor的存储设备，例如可以在CPU以及GPU中转化。经过`tf.identity(x)`相当于产生了一个`x`完整的copy。下面是这个函数实现的代码：

```python
if context.executing_eagerly():
    input = ops.convert_to_tensor(input)
    in_device = input.device
    # TODO(ashankar): Does 'identity' need to invoke execution callbacks?
    context_device = context.context().device_name
    if not context_device:
        context_device = "/job:localhost/replica:0/task:0/device:CPU:0"
    if context_device != in_device:
        return input._copy()  # pylint: disable=protected-access
    return input
else:
    return gen_array_ops.identity(input, name=name)
```

## `tf.control_dependencies`

这个函数会在with代码快中，仅包含一个参数`control_inputs`，也就是代码快中的op在执行之前首先需要执行这个参数中的op。如果with代码块中没有形成新的op，那么这个管理器就相当于失效了。下面是TF官方关于这个参数的介绍。

* `control_inputs`: A list of Operation or Tensor objects which must be executed or computed before running the operations defined in the context. Can also be None to clear the control dependencies. If eager execution is enabled, any callable object in the `control_inputs` list will be called.


## 一个例子
这个例子参考: [https://blog.csdn.net/hu_guan_jie/article/details/78495297](https://blog.csdn.net/hu_guan_jie/article/details/78495297)

下面程序的功能是，做5次循环，每次循环给x加1，赋值给y，然后打印出来，所以我们预期达到的效果是输出2，3，4，5，6。

```python
x = tf.Variable(1.0)
y = tf.Variable(0.0)

#返回一个op，表示给变量x加1的操作
x_plus_1 = tf.assign_add(x, 1)

#control_dependencies的意义是，在执行with包含的内容（在这里就是 y = x）前，
#先执行control_dependencies参数中的内容（在这里就是 x_plus_1），这里的解释不准确，先接着看。。。
with tf.control_dependencies([x_plus_1]):
    y = x
init = tf.initialize_all_variables()

with tf.Session() as session:
    init.run()
    for i in xrange(5):
        print(y.eval())#相当于sess.run(y)，按照我们的预期，由于control_dependencies的作用，所以应该执行print前都会先执行x_plus_1，但是这种情况会出问题

```
这个打印的是1，1，1，1，1 。可以看到，没有达到我们预期的效果，y只被赋值了一次。

如果改成这样：

```python
x = tf.Variable(1.0)
y = tf.Variable(0.0)
x_plus_1 = tf.assign_add(x, 1)

with tf.control_dependencies([x_plus_1]):
    y = tf.identity(x)#修改部分
init = tf.initialize_all_variables()

with tf.Session() as session:
    init.run()
    for i in xrange(5):
        print(y.eval())
```
这时候打印的是2，3，4，5，6

解释：对于`control_dependencies`这个管理器，只有当里面的操作是一个op时，才会生效，也就是先执行传入的参数op，再执行里面的op。而y=x仅仅是tensor的一个简单赋值，不是定义的op，所以在图中不会形成一个节点，这样该管理器就失效了。`tf.identity`是返回一个一模一样新的tensor的op，这会增加一个新节点到gragh中，这时`control_dependencies`就会生效，所以第二种情况的输出符合预期。