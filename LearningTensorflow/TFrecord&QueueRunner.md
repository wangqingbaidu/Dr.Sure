# [TFrecord & QueueRunner](https://www.tensorflow.org/api_guides/python/reading_data)

### 1. 计算图数据输入方式
Tensorflow把数据输送到计算图的方式主要有三种：

1. Constant: 把Dataset中的数据以const的形式存放在tensorflow的计算图中，主要使用的是`tf.constant`函数。这种形式主要适用于小数据集，由于数据固化到了计算图中，所以它的数据读取速度是最快的。

2. Feeding: 这种方式是在每次`session.run`的时候，把numpy形式的数据输入到feed_dict参数中。这种方式主要包括两种存在的状态。  [参见一个例子]()
>* 数据一次性全部load到内存中。自己维护一个`DataProvider`类，每次都会获取一部分训练数据。需要注意的是training的时候最好把数据集shuffle，但是test的时候最好不好shuffle。  
>* 当训练数据无法一次性全部load到内存中区的时候，分批次load数据。自己维护的`DataProvider`类要做好队列的管理。这种形式的一个小的trick是可以每次载入的数据使用多次进行训练，这样可以减少重复地读取数据。

3. Queue: 从文件中读取数据，使用`QueueRunner`的形式从文件中读取。tensorflow以一种黑箱的方式读取数据，必要的时候会启动多进程（需要设置，但是这些代码是用c++封装的，所以多线程支持的效果比较好）。

### 2. Feeding的优缺点
#### 优点：
1. 当数据集较小的时候，数据可以全部载入到内存，这时数据的处理速度就会比较快。
2. 训练和测试几乎可以共用一套代码，仅需要把反馈网络去掉，不适用参数更新即可。工程师改动相对较少
3. 在inference的时候，一般数据不会是文件的形式，所以这时候就只能使用feeding的方式。

#### 缺点：
1. 自己维护`DataProvider`类相对麻烦，而且自己写的类原生是不支持Multi-Process的。
2. 单进程读取数据较慢，很多时间花费在数据读取上，所以训练时间相对较长。

### 3. 为什么要使用 Tfrecord和Queuerunner?
>python不支持多线程（伪多线程，虽然启动multi-thread，但是所有启用线程的处理能力加起来等于一个核的处理能力），多进程如果是任务可分，不用和主线程交互的情况下是可用的，但是对于tensorflow的训练来说，肯定需要使用多（线程、进程）与主（线程、进程）交互。

Tensorflow使用黑箱的方式为数据的解析提供支持，相比于自己写multi-process然后共享变量来说在代码实现上更加友好。


## 一、TFrecord
特点：存储时以Key-Vaule键值对的形式进行存储，不改变原始数据的大小（不对数据进行编码或者解码，图像以图像原始的形式的二进制形式存储），读取时使用相同的feature_map即可使用`tf.parse_single_example`函数进行解析。

#### 1. 生成TFrecord

```python
def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    
def _float_feature(value):
	"""Wrapper for inserting float features into Example proto."""
	if not isinstance(value, list):
		value = [value]
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
def _convert_to_example(filename, image_buffer, labels, height, width, channels=3):
    """Build an Example proto for an example.
    
    Parameters
    ---------------
    @filename: string, path to an image file, e.g., '19901221.jpg'
    @image_buffer: string, JPEG encoding of RGB image
    @labels: list of semantic level name and one-hot encoder, 
        format list of tuple: [(level_name, [0,0,1,0,0,1,0,0]), ..]
    @height: integer, image height in pixels.
    @width: integer, image width in pixels.
    @channels: integer, image channels.
    Return:
    ---------------
    @example: Example proto
    """
    feature = {'image/height': _int64_feature(height),
               'image/width': _int64_feature(width),
               'image/channels': _int64_feature(channels),
               'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
               'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}
    for level_name, one_hot in labels:
        feature.update({'image/label/%s'%level_name: _int64_feature(one_hot)})
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example
```

`feature`变量对应的是一个example，其中`labels`或者`height`, `width`等相关的信息使用`int64`的类型进行存储，其他的例如`filename`以及`image`都是使用byte进行存储的。把`feature`的dict建立好之后使用`tf.train.Example(features=tf.train.Features(feature=feature))`可以生成一个对应的tf的example。

```python
writer = tf.python_io.TFRecordWriter(output_file)
example = balabala
writer.write(example.SerializeToString())
```
`writer.write`方法类似于`file`的方式，不停地向后追加example即可。[完整代码参考]()

#### 2. 解析TFrecord
解析TFrecord相对容易，直接使用跟生成TFrecord一样的`feature_map`即可，同样以dict的形式返回。使用`tf.parse_single_example(example_serialized, feature_map)`即可解析出一个相应的TFrecord样本。`example_serialized`会在`QueueRunner`详细介绍。

```python
def parse_example_proto(example_serialized, semantic_level_settings):
    """Parses an Example proto containing a training example of an image.
    
    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields:
      image/class/root: [0,1,1,0,0,0]
      image/filename: '19901221.jpg'
      image/encoded: <JPEG encoded string>
    
    Parameters
    ---------------
    @example_serialized: Scalar Tensor tf.string containing a serialized Example protocol buffer.
    @semantic_level_settings: Specify number of classes in each semantic level.
        format list of tuple: [(level_name, 10), ..]
    
    Return:
    ---------------
    @image_buffer: Tensor tf.string containing the contents of a JPEG file.
    @labels: List of Tensor tf.int32 containing the one-hot label.
    @filename: Tensor tf.string filename of the sample.
    """
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    }
    for level, num in semantic_level_settings:
        feature_map.update({'image/label/%s' %level:
                            tf.FixedLenFeature([num], dtype=tf.int64, default_value=[-1] * num)})
    features = tf.parse_single_example(example_serialized, feature_map)
    labels = [tf.cast(features['image/label/%s' %level], dtype=tf.int32) 
              for level, _ in semantic_level_settings]
    return features['image/encoded'], labels, features['image/filename']
```
`feature_map`中数据类型参数`dtype`要与前文一致，同样需要设置默认的数值（当TFrecord中不存在相应的字段时使用）。

## 二、QueueRunner
在使用文件对数据进行读取时，主要维护的两个队列。一个是文件的队列，这个队列里面放的是TFrecord文件，另一个队列维护的是单个TFrecord中example的队列。
流程如下：

1. `data_files = tf.gfile.Glob(tf_record_pattern)`生成TFrecord文件列表，`tf_record_pattern`为正则表达式。生成列表也可以使用其他的一些自定义方式例如`os.system("find dir -name 'pattern'")`方式。
2. `filename_queue = tf.train.string_input_producer(data_files, shuffle=True, capacity=16)`首先生成一个`filename_queue`,这个队列中存放了所有的文件（包括文件的组织形式），通过`shuffle`参数指定是不是打乱文件，一般在training的时候会设置成`True`，预测的时候设置成`False`，`capacity`为队列里至少的filename容量，这个参数随意。
3. 使用`tf.TFRecordReader()`生成example的队列。   

	> 当reader的个数大于1的时候，使用`QueueRunner`进行调度，其中queue包括`RandomShuffleQueue`和`FIFOQueue`前者主要用在训练的时候，后者主要用在预测的时候。其中`dtype`参数为`[string]`
4. 启动多(单)线程读取数据。
5. 获取一个batch的数据`tf.train.batch_join`。

```python
def batch_inputs(dataset, batch_size, train, semantic_level_settings, num_preprocess_threads=16):
    """Generate batches of data for training or validating or something.
    
    Parameters
    ---------------
    @dataset: instance of Dataset class specifying the dataset.
    @batch_size: integer, number of examples in batch
    @train: The gotten batch data if for training or not.
        if True, number of reader if large and doesn't shuffle batch data
        if False, only 1 reader and data is not shuffled.
    @semantic_level_settings: Specify number of classes in each semantic level.
        format list of tuple: [(level_name, 10), ..]
    @num_preprocess_threads: integer, total number of preprocessing threads.
    
    Return:
    ---------------
    @images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                       image_size, 3].
    @labels: Dict, key is `level_name` in `semantic_level_settings`, value is label.
    @filenames: 1-D string Tensor of [batch_size].
    """    
    # Force all input processing onto CPU in order to reserve the GPU for the forward and backward.
    with tf.device('/cpu:0'):
        with tf.name_scope('batch_processing'):
            data_files = dataset.data_files()
            if data_files is None:
                raise ValueError('No data files found for this dataset')
            
            examples_per_shard = 1024
            # Create filename_queue
            if train:
                filename_queue = tf.train.string_input_producer(data_files, shuffle=True, capacity=16)
                input_queue_memory_factor = 16
                num_readers = 4
            else:
                filename_queue = tf.train.string_input_producer(data_files, shuffle=False, capacity=1)
                input_queue_memory_factor = 1
                num_readers = 1
            if num_preprocess_threads % 4:
                raise ValueError('Please make num_preprocess_threads a multiple '
                               'of 4 (%d % 4 != 0).', num_preprocess_threads)
            
            min_queue_examples = examples_per_shard * input_queue_memory_factor
            if train:
                examples_queue = tf.RandomShuffleQueue(
                    capacity=min_queue_examples + 3 * batch_size,
                    min_after_dequeue=min_queue_examples,
                    dtypes=[tf.string])
                # Create multiple readers to populate the queue of examples.
                enqueue_ops = []
                for _ in range(num_readers):
                    reader = dataset.reader()
                    _, value = reader.read(filename_queue)
                    enqueue_ops.append(examples_queue.enqueue([value]))
                
                tf.train.queue_runner.add_queue_runner(
                    tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
                example_serialized = examples_queue.dequeue()
            else:
                examples_queue = tf.FIFOQueue(
                    capacity=examples_per_shard + 3 * batch_size,
                    dtypes=[tf.string])
                # Create multiple readers to populate the queue of examples.
                reader = dataset.reader()
                _, example_serialized = reader.read(filename_queue)
            
            images_and_labels = []
            for thread_id in range(num_preprocess_threads):
                # Parse a serialized Example proto to extract the image and metadata.
                image_buffer, labels, filename = parse_example_proto(example_serialized,
                                                                     semantic_level_settings)
                image = decode_jpeg(image_buffer)
                if train:
                    image = distort_image(image, dataset.height, dataset.width, thread_id)
                else:
                    image = eval_image(image, dataset.height, dataset.width)
                
                # Finally, rescale to [-1,1] instead of [0, 1)
                image = tf.subtract(image, 0.5)
                image = tf.multiply(image, 2.0)
                images_and_labels.append([image, filename] + labels)
            
            batch_data = tf.train.batch_join(
                images_and_labels,
                batch_size=batch_size,
                capacity=2 * num_preprocess_threads * batch_size)
            
            # Get image data, filenames, level_labels separately.
            images = batch_data[0]
            images = tf.cast(images, tf.float32)
            images = tf.reshape(images, shape=[batch_size, dataset.height, dataset.width, 3])
        
            filenames = tf.reshape(batch_data[1], [batch_size])
            level_labels = {}
            for idx, settings in enumerate(semantic_level_settings):
                level_labels[settings[0]] = tf.reshape(batch_data[2 + idx], [batch_size, -1])
            
            return (images, level_labels, filenames)
```

## 三、使用TFrecord和QueueRunner
本质上来说这两个方式的结合是把原本model需要以图像和标签作为源头的进行训练的形式，通过这两个方式改成以文件名的形式进行输入，这两种方式的结合可以看成是对通过Tensorflow提供的op，把文件转化成了图像以及标签等相关数据，这个也类似于计算图。

所以通过这两种方式生成的数据可以直接放入到后面的计算图中，直接使用。

```python
images, labels, _ = DataProvider.batch_inputs(dataset, args.batch_size)
model = MultiLabelTree(images, labels, is_training=True, reuse=reuse)
```

在使用queue的时候需要使用到一个辅助类`Coordinator`

```python
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=session, coord=coord)

TRAING ITERS...

coord.request_stop()
coord.join(threads)
```
全部的一个流程放在[MultiLabelTree]()。
