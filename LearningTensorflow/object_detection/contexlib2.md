# [models.research.object_detection](https://github.com/tensorflow/models/tree/master/research/object_detection)源码解析——支线contextlib2

近期在研究Tensorflow中的Object Detection的源代码，在build TFRecord的时候，发现了一个非常有意思的库。这里总结一下，下面是这个代码片段，想要实现的功能就是生成对应的TFRecord句柄，把数据写入到这个文件中。

```python
def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    tf_record_output_filenames = [
        '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
        for idx in range(num_shards)
    ]

    tfrecords = [
        exit_stack.enter_context(tf.python_io.TFRecordWriter(file_name))
        for file_name in tf_record_output_filenames
    ]

    return tfrecords 
    
with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filename, num_shards)
```

这里面有一个原来没见过的库，contexlib2。这里就把它简单总结一下。

## with代码块
我们在操作文件时最常用的就是使用with上下文管理器，这样会让代码的可读性更强而且错误更少，这样写的好处在于，在执行完毕缩进代码块后会自动关闭文件，例如：

```python
with open('/tmp/a.txt', 'w') as f:
    f.write("hello wangqingbaidu!")
```

同样的例子还有threading.Lock，如果不使用with，需要这样写：

```python
import threading
lock = threading.Lock()

lock.acquire()
try:
    my_list.append(item)
finally:
    lock.release()
```
如果使用with，那就会非常简单：

```python
with lock:
    my_list.append(item)
```

## contexlib
大家都知道，创建上下文管理实际就是创建一个类，添加`__enter__`和`__exit__`方法。下面我们来实现open的上下文管理功能：

```python
class OpenContext(object):

    def __init__(self, filename, mode):
        self.fp = open(filename, mode)

    def __enter__(self):
        return self.fp

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fp.close()
        
with OpenContext('/tmp/a.txt', 'w') as f:
    f.write("hello wangqingbaidu!")
```

上面我们自定义上下文管理器确实很方便，但是Python标准库还提供了更加易用的上下文管理器工具模块contextlib，它是通过生成器实现的，我们不需要再创建类以及`__enter__`和`__exit__`这两个特殊的方法：

```python
from contextlib import contextmanager

@contextmanager
def make_open_context(filename, mode):
    fp = open(filename, mode)
    try:
        yield fp
    finally:
        fp.close()

with make_open_context('/tmp/a.txt', 'w') as f:
    f.write("hello wangqingbaidu!")
```

在上文中，yield关键词把上下文分割成两部分：yield之前就是`__init__`中的代码块；yield之后其实就是`__exit__`中的代码块，yield生成的值会绑定到with语句as子句中的变量。下面看一个具体的case。

```python
# _*_ coding:utf-8 _*_
from contextlib import contextmanager

class MyResource:
    def query(self):
        print("query data")

@contextmanager
def make_myresource():
    print("connect to resource")
    yield MyResource()
    print("connect to resource")

with make_myresource() as r:
    r.query()
```

上面的例子就充分体现了contextmanager的强大作用，将一个不是上下问管理器的类 MyResource变成了一个上下文管理器，这样做的好处在于，我们就可以在执行真正的核心代码之前可以执行一部分代码，然后在执行完毕后，又可以执行一部分代码，这种场景在实际需求中还是很常见的。上面yield MyResource() 生成了一个实例对象，然后我们可以在with语句中调用类中的方法。看看最终的打印结果：

```shell
$ connect to resource
$ query data
$ connect to resource
```

上面例子参考：[http://www.cnblogs.com/pyspark/articles/8819803.html](http://www.cnblogs.com/pyspark/articles/8819803.html)

## contextlib2.ExitStack()
`contextlib2`与`contextlib`是差不多的，`contextlib2.ExitStack()`相当于是一个with代码的堆栈，把所有的`contexmanager`放入到这个堆栈当中去管理。

使用方法：

1. `with`初始化堆栈，`with contextlib2.ExitStack() as stack`。
2. 把所有需要管理的资源追加入栈，`stack.enter_context(cm)`。
3. 使用对应的句柄资源，`output_tfrecords[shard_idx].write(sth)`。

这种做法可以尽量保证所有的资源都进行了`__close__`操作，使代码更加安全。
