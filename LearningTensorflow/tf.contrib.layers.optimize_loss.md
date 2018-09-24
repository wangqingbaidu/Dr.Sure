# tf.contrib.layers.optimize_loss

optimize_loss用来优化网络参数。

这个相当于联系应用下面几个步骤：

1. `compute_gradients`: 根据对应的参数，计算每个参数的梯度。
2. `do sth with gradients`: 对每个参数采取一定的操作，例如clip这个梯度，防止梯度爆炸
3. `apply_gradients`: 对每个参数应用梯度更新。

下面是这个函数的构造函数：

```python
tf.contrib.layers.optimize_loss(
    loss,
    global_step,
    learning_rate,
    optimizer,
    gradient_noise_scale=None,
    gradient_multipliers=None,
    clip_gradients=None,
    learning_rate_decay_fn=None,
    update_ops=None,
    variables=None,
    name=None,
    summaries=None,
    colocate_gradients_with_ops=False,
    increment_global_step=True
)
```


### How to Use.

这个模块可以包括4中主要的用法：

1. `optimizer`: 可以使用字符串，例如`Adam`，但是必须要使用OPTIMIZER_CLS_NAMES。

	```python
	OPTIMIZER_CLS_NAMES = {
	"Adagrad": train.AdagradOptimizer,
	"Adam": train.AdamOptimizer,
	"Ftrl": train.FtrlOptimizer,
	"Momentum": lambda learning_rate: train.MomentumOptimizer(learning_rate, 
		momentum=0.9),
	"RMSProp": train.RMSPropOptimizer,
	"SGD": train.GradientDescentOptimizer,
	}
	```
	
2. 可以是一个使用`learing_rate`作为参数，返回Optimizer一个实例的function。例如：
	
	```python
	optimize_loss(..., optimizer=lambda lr: tf.train.MomentumOptimizer(lr, 
		momentum=0.5))
	```
	
	```python
	optimize_loss(..., learning_rate=None, 
		optimizer=lambda: tf.train.MomentumOptimizer(0.5, momentum=0.5))
	```
	
3. 当然也可以是一个`Optimizer`的子类，例如：`optimize_loss(..., optimizer=tf.train.AdagradOptimizer)`。

4. 或者是一个`Optimizer`的子类的实例。例如：`optimize_loss(..., optimizer=tf.train.AdagradOptimizer(0.5))`。

### Args
下面是这个模块的英文参数解释

* `loss`: Scalar Tensor.

* `global_step`: Scalar int Tensor, step counter to update on each step unless `increment_global_step` is False. If not supplied, it will be fetched from the default graph (see `tf.train.get_global_step` for details). If it has not been created, no step will be incremented with each weight update. `learning_rate_decay_fn` requires global_step. 
	
	一般使用`tf.train.get_or_create_global_step()`函数获取到当前graph的`global_step`。

* `learning_rate`: float or Tensor, magnitude of update per each training step. Can be None.

* `optimizer`: string, class or optimizer instance, used as trainer. string should be name of optimizer, like 'SGD', 'Adam', 'Adagrad'. Full list in `OPTIMIZER_CLS_NAMES` constant. class should be sub-class of tf.Optimizer that implements `compute_gradients` and `apply_gradients` functions. optimizer instance should be instantiation of tf.Optimizer sub-class and have `compute_gradients` and `apply_gradients` functions. 

	按照上面的使用方式。

* `gradient_noise_scale`: float or None, adds 0-mean normal noise scaled by this value.

* `gradient_multipliers`: dict of variables or variable names to floats. If present, gradients for specified variables will be multiplied by given constant.
`clip_gradients`: float, callable or None. If float, is provided, a global clipping is applied to prevent the norm of the gradient to exceed this value. Alternatively, a callable can be provided e.g.: `adaptive_clipping`. This callable takes a list of (gradients, variables) tuples and returns the same thing with the gradients modified.

* `learning_rate_decay_fn`: function, takes `learning_rate` and `global_step` Tensors, returns Tensor. Can be used to implement any learning rate decay functions. For example: `tf.train.exponential_decay`. Ignored if `learning_rate` is not supplied.

* `update_ops`: list of update Operations to execute at each step. If None, uses elements of `UPDATE_OPS` collection. The order of execution between update_ops and loss is non-deterministic.

* `variables`: list of variables to optimize or None to use all trainable variables.
name: The name for this operation is used to scope operations and summaries.
summaries: List of internal quantities to visualize on tensorboard. If not set, the loss, the learning rate, and the global norm of the gradients will be reported. The complete list of possible values is in `OPTIMIZER_SUMMARIES`.

* `colocate_gradients_with_ops`: If True, try colocating gradients with the corresponding op.

* `increment_global_step`: Whether to increment `global_step`. If your model calls optimize_loss multiple times per training step (e.g. to optimize different parts of the model), use this arg to avoid incrementing `global_step` more times than necessary.