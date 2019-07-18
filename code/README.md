# Components
1. [attention](./attention.py)，Attention模型，使用了spatial以及channel的Attention，在一个[英文的文本算法文献](https://arxiv.org/abs/1612.08083)中叫做Gated Convolutional。

2. [losses](./losses.py)，损失函数汇总，摘抄自[tensorflow.models.research.object_detection.core.losses.py](https://github.com/tensorflow/models/blob/master/research/object_detection/core/losses.py) 。
	1. `Loss`: 损失函数的基类。包括两个方法：
		1. `__call__`: 相当于直接计算损失值。
		2. `_compute_loss`: 计算损失值的具体实现方式，接口需要子类实现。
	2. `WeightedL2LocalizationLoss`:L2距离损失函数，主要是bbox的回归损失。
	3. `WeightedSmoothL1LocalizationLoss`: 又称HuberLoss，详细的介绍可以[参考](https://blog.csdn.net/lanchunhui/article/details/50427055)。
	4. `WeightedIOULocalizationLoss`: 计算IOU Loss，目的是让bbox之间的overlap尽量地大。 `loss = 1 - iou`。实现方法参考了`box_list_ops.matched_iou`。
	5. `WeightedSigmoidClassificationLoss`: sigmoid loss，中间有一个参数`class_indices`，这个参数类似于ont_hot，只对里面的class计算损失。
	
		```python
		weights *= tf.reshape(ops.indices_to_dense_vector(class_indices,
			tf.shape(prediction_tensor)[2]), [1, 1, -1])
		```
	6. `SigmoidFocalClassificationLoss`: KM He 的[FocalLoss](https://arxiv.org/abs/1708.02002)具体实现。
	7. `WeightedSoftmaxClassificationLoss`: 与sigmoid类似，唯一不同的是有一个Logit的scale参数。
	
		```python
		prediction_tensor = tf.divide(prediction_tensor, self._logit_scale,
                                      name='scale_logits')
		```
	8. `BootstrappedSigmoidClassificationLoss`: 使用一定的预测结果作为groundtruth，主要的出发点是：模型随着训练的进行会越来越准，而标注数据中间可能存在着一定的噪声，所以可以使用一定的预测结果作为GT，使用的比例通过`alpha`参数控制。bootstrap的方式有两种：
		1. `soft`, 即使用概率作为gt。
		2. `hard`，使用sigmoid的概率>0.5的作为预测结果，然后强转成概率为1。

	9. `HardExampleMiner`, Online困难样本挖掘的具体实现，参考论文：[
Training Region-based Object Detectors with Online Hard Example Mining](https://arxiv.org/abs/1604.03540)

3. [numpy_fc](numpy_fc.py)，基于Numpy的前向传播和反向传播，但却是框架设计最简单的雏形。
