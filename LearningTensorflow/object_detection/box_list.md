# [models.research.object_detection](https://github.com/tensorflow/models/tree/master/research/object_detection)源码解析—`core.box_list`

`box_list`是一个ObjectDetection项目中，一个综合管理bounding box的工具库，我们把下面的讲解分成两个主要的方面。

1. [`box_list.BoxList`](https://github.com/tensorflow/models/blob/master/research/object_detection/core/box_list.py)，管理box的类。每个BBox必须包括4个数据，`[y_min, x_min, y_max, x_max]`分别对应左上角的坐标和右下角的坐标，这个坐标可以是相对的，也可以是绝对的。
2. [`box_list_ops`](https://github.com/tensorflow/models/blob/master/research/object_detection/core/box_list_ops.py)对于box操作的所有的op，包括合并、拆分、剪枝、计算IOU等操作。

## BoxList

这个类里面最重要的属性是`data`, 它可以有很多的字段，当然要有一个最重要的字段--`boxes`，用来表示Box的数据，这个box是一个rank=2并且最后一维4个数，即`[N, 4]`的shape。

下面就挑选一些关键的方法进行介绍。

在具体介绍这个相关的方法之前，先介绍一个在这些代码中经常出现的一段代码。

```python
y_min, x_min, y_max, x_max = tf.split(
	value=boxlist.get(), num_or_size_splits=4, axis=1)
```
关于`tf.split`的用法大家可以参考官方文档


### 1. 初始化
```python
def __init__(self, boxes):
    """Constructs box collection.

    Args:
      boxes: a tensor of shape [N, 4] representing box corners

    Raises:
      ValueError: if invalid dimensions for bbox data or if bbox data is not in float32 format.
    """
    if len(boxes.get_shape()) != 2 or boxes.get_shape()[-1] != 4:
        raise ValueError('Invalid dimensions for box data.')
    if boxes.dtype != tf.float32:
        raise ValueError('Invalid tensor type: should be tf.float32')
    self.data = {'boxes': boxes}
```
这个地方相对好理解一些，首先要检查输入`boxes`的形状以及数据类型，然后就是形成data属性。

### 2. get\_all\_fields
这个方法返回`data`中所有的字段名称，一般情况下，Boxlist除了维度一个`boxes`字段还会维护一个`scores`字段，这个地段用来表示当前box属于物体的置信度。

因为`data`属性属于dict类型，所以他的操作也是很简单，直接返回`self.data.keys()`就行。

### 3. get，set
这个类里面包括两种get和set。

1. `get()`、`set()`，用来获取当前对象的boxes数据和设置当前对象的boxes数据。
2. `get_field()`、`set_field()`，设置和获取某一个具体的field数据, 前面的只是后者的一种特殊情况。

### 4. get\_center\_coordinates\_and\_sizes
```python
def get_center_coordinates_and_sizes(self, scope=None):
    """Computes the center coordinates, height and width of the boxes.

    Args:
      scope: name scope of the function.

    Returns:
      a list of 4 1-D tensors [ycenter, xcenter, height, width].
    """
    with tf.name_scope(scope, 'get_center_coordinates_and_sizes'):
        box_corners = self.get()
        ymin, xmin, ymax, xmax = tf.unstack(tf.transpose(box_corners))
        width = xmax - xmin
        height = ymax - ymin
        ycenter = ymin + height / 2.
        xcenter = xmin + width / 2.
        return [ycenter, xcenter, height, width]
```
这个方法是获得bbox的中心和宽高，因为输入的bbox的数据格式是`[y_min, x_min, y_max, x_max]`。

### 5. transpose\_coordinates
```python
def transpose_coordinates(self, scope=None):
    """Transpose the coordinate representation in a boxlist.

    Args:
      scope: name scope of the function.
    """
    with tf.name_scope(scope, 'transpose_coordinates'):
        y_min, x_min, y_max, x_max = tf.split(
            value=self.get(), num_or_size_splits=4, axis=1)
        self.set(tf.concat([x_min, y_min, x_max, y_max], 1))
```
交换坐标，从(x, y)交换成(y, x)。这里的`tf.split`的作用和上面的那个`tf.unstack(tf.transpose())`是相同的效果。

### 6. as\_tensor\_dict
```python
def as_tensor_dict(self, fields=None):
    """Retrieves specified fields as a dictionary of tensors.

    Args:
      fields: (optional) list of fields to return in the dictionary.
        If None (default), all fields are returned.

    Returns:
      tensor_dict: A dictionary of tensors specified by fields.

    Raises:
      ValueError: if specified field is not contained in boxlist.
    """
    tensor_dict = {}
    if fields is None:
        fields = self.get_all_fields()
    for field in fields:
        if not self.has_field(field):
            raise ValueError('boxlist must contain all specified fields')
        tensor_dict[field] = self.get_field(field)
    return tensor_dict
```
这个是返回指定fields的数据，并且返回的是一个dict。如果`field=None`，就跟返回`data`是一样的。

## box\_list\_ops.py
在这个模块中，包括各种常用的对于box处理的op，下面我们逐一进行分析。

### 1. area, 求bbox的面积。
```python
def area(boxlist, scope=None):
    """Computes area of boxes.

    Args:
      boxlist: BoxList holding N boxes
      scope: name scope.

    Returns:
      a tensor with shape [N] representing box areas.
    """
    with tf.name_scope(scope, 'Area'):
        y_min, x_min, y_max, x_max = tf.split(
            value=boxlist.get(), num_or_size_splits=4, axis=1)
        return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

```
这个操作比较好理解，就是长×宽，也就是坐标的最大值-最小值分别得到长和宽，这个对应下面的那个`height_width()`函数。

### 2. scale，对bbox进行尺度变换
```python
def scale(boxlist, y_scale, x_scale, scope=None):
    """scale box coordinates in x and y dimensions.

    Args:
      boxlist: BoxList holding N boxes
      y_scale: (float) scalar tensor
      x_scale: (float) scalar tensor
      scope: name scope.

    Returns:
      boxlist: BoxList holding N boxes
    """
    with tf.name_scope(scope, 'Scale'):
        y_scale = tf.cast(y_scale, tf.float32)
        x_scale = tf.cast(x_scale, tf.float32)
        y_min, x_min, y_max, x_max = tf.split(
            value=boxlist.get(), num_or_size_splits=4, axis=1)
        y_min = y_scale * y_min
        y_max = y_scale * y_max
        x_min = x_scale * x_min
        x_max = x_scale * x_max
        scaled_boxlist = box_list.BoxList(
            tf.concat([y_min, x_min, y_max, x_max], 1))
        return _copy_extra_fields(scaled_boxlist, boxlist)
```
这个是分别对x坐标和y坐标进行尺度变换，然后形成一个新的BoxList类，当然最后如果需要copy其他的字段。

### 3. clip\_to\_window，将bbox裁剪到给定的window
```python
def clip_to_window(boxlist, window, filter_nonoverlapping=True, scope=None):
    """Clip bounding boxes to a window.

    This op clips any input bounding boxes (represented by bounding box
    corners) to a window, optionally filtering out boxes that do not
    overlap at all with the window.

    Args:
      boxlist: BoxList holding M_in boxes
      window: a tensor of shape [4] representing the [y_min, x_min, y_max, x_max]
        window to which the op should clip boxes.
      filter_nonoverlapping: whether to filter out boxes that do not overlap at
        all with the window.
      scope: name scope.

    Returns:
      a BoxList holding M_out boxes where M_out <= M_in
    """
    with tf.name_scope(scope, 'ClipToWindow'):
        y_min, x_min, y_max, x_max = tf.split(
            value=boxlist.get(), num_or_size_splits=4, axis=1)
        win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
        y_min_clipped = tf.maximum(tf.minimum(y_min, win_y_max), win_y_min)
        y_max_clipped = tf.maximum(tf.minimum(y_max, win_y_max), win_y_min)
        x_min_clipped = tf.maximum(tf.minimum(x_min, win_x_max), win_x_min)
        x_max_clipped = tf.maximum(tf.minimum(x_max, win_x_max), win_x_min)
        clipped = box_list.BoxList(
            tf.concat([y_min_clipped, x_min_clipped, y_max_clipped, x_max_clipped],
                      1))
        clipped = _copy_extra_fields(clipped, boxlist)
        if filter_nonoverlapping:
            areas = area(clipped)
            nonzero_area_indices = tf.cast(
                tf.reshape(tf.where(tf.greater(areas, 0.0)), [-1]), tf.int32)
            clipped = gather(clipped, nonzero_area_indices)
        return clipped
```
cliped的过程比较简单，就不再介绍，里面有一个非常有意思的参数`filter_nonoverlapping`,这个参数用于控制要不要把里面的没有overlap的bbox去掉。实现起来也是符合逻辑，即去掉里面的面积为零的框。所以

> 1. 第一步计算当前bbox的面积。
> 2. 获取为0的bbox对应的index。
> 3. gather出来新的bbox。

### 4. prune\_outside\_window, prune\_completely\_outside\_window，去掉（全）落在windows外面的Bbox

```python
def prune_outside_window(boxlist, window, scope=None):
    """Prunes bounding boxes that fall outside a given window.

    This function prunes bounding boxes that even partially fall outside the given
    window. See also clip_to_window which only prunes bounding boxes that fall
    completely outside the window, and clips any bounding boxes that partially
    overflow.

    Args:
      boxlist: a BoxList holding M_in boxes.
      window: a float tensor of shape [4] representing [ymin, xmin, ymax, xmax]
        of the window
      scope: name scope.

    Returns:
      pruned_corners: a tensor with shape [M_out, 4] where M_out <= M_in
      valid_indices: a tensor with shape [M_out] indexing the valid bounding boxes
       in the input tensor.
    """
    with tf.name_scope(scope, 'PruneOutsideWindow'):
        y_min, x_min, y_max, x_max = tf.split(
            value=boxlist.get(), num_or_size_splits=4, axis=1)
        win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
        ...
        valid_indices = tf.reshape(
            tf.where(tf.logical_not(tf.reduce_any(coordinate_violations, 1))), [-1])
        return gather(boxlist, valid_indices), valid_indices
```

这两个函数类似，唯一不同的就是在最后条件判断的地方。

1. prune\_outside\_window

	```python
	coordinate_violations = tf.concat([
	    tf.less(y_min, win_y_min), tf.less(x_min, win_x_min),
	    tf.greater(y_max, win_y_max), tf.greater(x_max, win_x_max)
	], 1)
	```
	在windows外面，即2个多标点，都不在windows的两个坐标点之内，所以任何坐标的(x, y)在windows之外都是不行的，即any。

2. prune\_completely\_outside\_window
	
	```python
	coordinate_violations = tf.concat([
	    tf.greater_equal(y_min, win_y_max), tf.greater_equal(x_min, win_x_max),
	    tf.less_equal(y_max, win_y_min), tf.less_equal(x_max, win_x_min)
	], 1)
	```
	完全在window外面，要求任何一个坐标值只要在window坐标值之外就行。
	
### 5. intersection，matched\_intersection求两个boxlist的相交面积
```python
def intersection(boxlist1, boxlist2, scope=None):
    """Compute pairwise intersection areas between boxes.

    Args:
      boxlist1: BoxList holding N boxes
      boxlist2: BoxList holding M boxes
      scope: name scope.

    Returns:
      a tensor with shape [N, M] representing pairwise intersections
    """
    with tf.name_scope(scope, 'Intersection'):
        y_min1, x_min1, y_max1, x_max1 = tf.split(
            value=boxlist1.get(), num_or_size_splits=4, axis=1)
        y_min2, x_min2, y_max2, x_max2 = tf.split(
            value=boxlist2.get(), num_or_size_splits=4, axis=1)
        all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
        all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
        intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
        all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
        all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
        intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
        return intersect_heights * intersect_widths
```
这个相交面积有点类似于笛卡尔积的意思，既是求任意一对box的相交面积。这里需要说明的一点是`tf.minimum`、`tf.maximum`，在API文档中给出来的，要求`y`必须要和`x`的形状相同，这一点如果不了解broadcasting可能会有点费解。

大家可以看到输入到API中的`tf.minimum(y_max1, tf.transpose(y_max2))`对`y_max2`进行了转置，也就是`x`,`y`的形状不在相同，但是可以看到，如果支持broadcast，可以认为`x`依次和`y`中的每个数进行minimum，最后得到的形状就是[N, M]。

相比来说matched\_intersection就比较简单了，就是相对应的位置，求相交面积。

### 6. iou，matched\_iou
```python
def iou(boxlist1, boxlist2, scope=None):
    """Computes pairwise intersection-over-union between box collections.

    Args:
      boxlist1: BoxList holding N boxes
      boxlist2: BoxList holding M boxes
      scope: name scope.

    Returns:
      a tensor with shape [N, M] representing pairwise iou scores.
    """
    with tf.name_scope(scope, 'IOU'):
        intersections = intersection(boxlist1, boxlist2)
        areas1 = area(boxlist1)
        areas2 = area(boxlist2)
        unions = (
                tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
        return tf.where(
            tf.equal(intersections, 0.0),
            tf.zeros_like(intersections), tf.truediv(intersections, unions))
```
求两个boxlist对应的iou，可以看到最核心的点在于计算两个bbox面积之和。

```python
unions = (tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
```
也就是先把area1和area2分别扩充一个axis，这样分别形成一个[N, 1]和[1, M]的矩阵，然后再通过加法的broadcast，得到一个[N, M]的矩阵，最后减去那个相交面积就是总共面积。

`matched_iou`就比较简单了，调用对应的`matched_intersection`方法就行了。

### 7. ioa计算相交面积占bbox2的比例。
```python
def ioa(boxlist1, boxlist2, scope=None):
    """Computes pairwise intersection-over-area between box collections.

    intersection-over-area (IOA) between two boxes box1 and box2 is defined as
    their intersection area over box2's area. Note that ioa is not symmetric,
    that is, ioa(box1, box2) != ioa(box2, box1).

    Args:
      boxlist1: BoxList holding N boxes
      boxlist2: BoxList holding M boxes
      scope: name scope.

    Returns:
      a tensor with shape [N, M] representing pairwise ioa scores.
    """
    with tf.name_scope(scope, 'IOA'):
        intersections = intersection(boxlist1, boxlist2)
        areas = tf.expand_dims(area(boxlist2), 0)
        return tf.truediv(intersections, areas)
```
### 8. prune\_non\_overlapping\_boxes裁掉没有相交的bbox
```python
def prune_non_overlapping_boxes(
        boxlist1, boxlist2, min_overlap=0.0, scope=None):
    """Prunes the boxes in boxlist1 that overlap less than thresh with boxlist2.

    For each box in boxlist1, we want its IOA to be more than minoverlap with
    at least one of the boxes in boxlist2. If it does not, we remove it.

    Args:
      boxlist1: BoxList holding N boxes.
      boxlist2: BoxList holding M boxes.
      min_overlap: Minimum required overlap between boxes, to count them as
                  overlapping.
      scope: name scope.

    Returns:
      new_boxlist1: A pruned boxlist with size [N', 4].
      keep_inds: A tensor with shape [N'] indexing kept bounding boxes in the
        first input BoxList `boxlist1`.
    """
    with tf.name_scope(scope, 'PruneNonOverlappingBoxes'):
        ioa_ = ioa(boxlist2, boxlist1)  # [M, N] tensor
        ioa_ = tf.reduce_max(ioa_, reduction_indices=[0])  # [N] tensor
        keep_bool = tf.greater_equal(ioa_, tf.constant(min_overlap))
        keep_inds = tf.squeeze(tf.where(keep_bool), squeeze_dims=[1])
        new_boxlist1 = gather(boxlist1, keep_inds)
        return new_boxlist1, keep_inds
```
这个地方有一个`min_overlap`参数，用于控制相加的比例，这个里面仔细看一下ioa的计算，他计算的是相交面积占boxlist1的比例。

### 9. prune\_small\_boxes，去掉边比较小的Bbox
```python
def prune_small_boxes(boxlist, min_side, scope=None):
    """Prunes small boxes in the boxlist which have a side smaller than min_side.

    Args:
      boxlist: BoxList holding N boxes.
      min_side: Minimum width AND height of box to survive pruning.
      scope: name scope.

    Returns:
      A pruned boxlist.
    """
    with tf.name_scope(scope, 'PruneSmallBoxes'):
        height, width = height_width(boxlist)
        is_valid = tf.logical_and(tf.greater_equal(width, min_side),
                                  tf.greater_equal(height, min_side))
        return gather(boxlist, tf.reshape(tf.where(is_valid), [-1]))
```
这个就是看所有的边都要比最小的边要长就行。

### 10. change\_coordinate\_frame，把bbox的坐标归一化window相对坐标
```python
def change_coordinate_frame(boxlist, window, scope=None):
    """Change coordinate frame of the boxlist to be relative to window's frame.

    Given a window of the form [ymin, xmin, ymax, xmax],
    changes bounding box coordinates from boxlist to be relative to this window
    (e.g., the min corner maps to (0,0) and the max corner maps to (1,1)).

    An example use case is data augmentation: where we are given groundtruth
    boxes (boxlist) and would like to randomly crop the image to some
    window (window). In this case we need to change the coordinate frame of
    each groundtruth box to be relative to this new window.

    Args:
      boxlist: A BoxList object holding N boxes.
      window: A rank 1 tensor [4].
      scope: name scope.

    Returns:
      Returns a BoxList object with N boxes.
    """
    with tf.name_scope(scope, 'ChangeCoordinateFrame'):
        win_height = window[2] - window[0]
        win_width = window[3] - window[1]
        boxlist_new = scale(box_list.BoxList(
            boxlist.get() - [window[0], window[1], window[0], window[1]]),
            1.0 / win_height, 1.0 / win_width)
        boxlist_new = _copy_extra_fields(boxlist_new, boxlist)
        return boxlist_new
```

### 11. boolean\_mask，根据bool值获取到True对应的Bbox
```python
def boolean_mask(boxlist, indicator, fields=None, scope=None,
                 use_static_shapes=False, indicator_sum=None):
    """Select boxes from BoxList according to indicator and return new BoxList.

    `boolean_mask` returns the subset of boxes that are marked as "True" by the
    indicator tensor. By default, `boolean_mask` returns boxes corresponding to
    the input index list, as well as all additional fields stored in the boxlist
    (indexing into the first dimension).  However one can optionally only draw
    from a subset of fields.

    Args:
      boxlist: BoxList holding N boxes
      indicator: a rank-1 boolean tensor
      fields: (optional) list of fields to also gather from.  If None (default),
        all fields are gathered from.  Pass an empty fields list to only gather
        the box coordinates.
      scope: name scope.
      use_static_shapes: Whether to use an implementation with static shape
        gurantees.
      indicator_sum: An integer containing the sum of `indicator` vector. Only
        required if `use_static_shape` is True.

    Returns:
      subboxlist: a BoxList corresponding to the subset of the input BoxList
        specified by indicator
    Raises:
      ValueError: if `indicator` is not a rank-1 boolean tensor.
    """
    with tf.name_scope(scope, 'BooleanMask'):
        if indicator.shape.ndims != 1:
            raise ValueError('indicator should have rank 1')
        if indicator.dtype != tf.bool:
            raise ValueError('indicator should be a boolean tensor')
        if use_static_shapes:
            if not (indicator_sum and isinstance(indicator_sum, int)):
                raise ValueError('`indicator_sum` must be a of type int')
            selected_positions = tf.to_float(indicator)
            indexed_positions = tf.cast(
                tf.multiply(
                    tf.cumsum(selected_positions), selected_positions),
                dtype=tf.int32)
            one_hot_selector = tf.one_hot(
                indexed_positions - 1, indicator_sum, dtype=tf.float32)
            sampled_indices = tf.cast(
                tf.tensordot(
                    tf.to_float(tf.range(tf.shape(indicator)[0])),
                    one_hot_selector,
                    axes=[0, 0]),
                dtype=tf.int32)
            return gather(boxlist, sampled_indices, use_static_shapes=True)
        else:
            subboxlist = box_list.BoxList(tf.boolean_mask(boxlist.get(), indicator))
            if fields is None:
                fields = boxlist.get_extra_fields()
            for field in fields:
                if not boxlist.has_field(field):
                    raise ValueError('boxlist must contain all specified fields')
                subfieldlist = tf.boolean_mask(boxlist.get_field(field), indicator)
                subboxlist.add_field(field, subfieldlist)
            return subboxlist
```
这个实现很有意思，先看看`use_static_shapes=False`的情况，这个情况比较简单，就是用用tensorflow自带的`tf.boolean_mask`函数就行，如果有需要就获取对应的field。

主要看一下这个`use_static_shapes=True`的情况。这个意思就是，使用这个静态的筛选的box的个数，下面来看看它的具体实现：

```python
selected_positions = tf.to_float(indicator)
indexed_positions = tf.cast(
    tf.multiply(tf.cumsum(selected_positions), selected_positions), dtype=tf.int32)
one_hot_selector = tf.one_hot(indexed_positions - 1, indicator_sum, dtype=tf.float32)
sampled_indices = tf.cast(tf.tensordot(
        tf.to_float(tf.range(tf.shape(indicator)[0])),
        one_hot_selector,
        axes=[0, 0]),
    dtype=tf.int32)
return gather(boxlist, sampled_indices, use_static_shapes=True)
```
> 1. 把`indicator`从bool型变量转化成float类型。
> 2. 获取索引之后的位置，这里有一个tensorflow的API, [tf.cumsum](https://www.tensorflow.org/api_docs/python/tf/math/cumsum?hl=en)，相当于是一个累进求职，即`tf.cumsum([a, b, c])  # [a, a + b, a + b + c]`，eg. `[0,0,1,2,2,3]`类似的情况。这个是获得新的筛选之后对应的index，但是在乘以`selected_positions`之后，变成[0, 0, 1, 2, 0, 3]。
> 3. 下面就是把这个index之后的tensor，转化成one_hot。这里面有个`indexed_positions - 1`, 也就是把那些为0的index变道`off_value`。
> 4. 下面就是通过这个对应的indexer，进行点积，点积的效果就是index只与hot的点变成1，其余都是0。这样就获得了最终的`static_indices`。

### 12. gather, 汇总所有indices的tensor
```python
def gather(boxlist, indices, fields=None, scope=None, use_static_shapes=False):
    """Gather boxes from BoxList according to indices and return new BoxList.

    By default, `gather` returns boxes corresponding to the input index list, as
    well as all additional fields stored in the boxlist (indexing into the
    first dimension).  However one can optionally only gather from a
    subset of fields.

    Args:
      boxlist: BoxList holding N boxes
      indices: a rank-1 tensor of type int32 / int64
      fields: (optional) list of fields to also gather from.  If None (default),
        all fields are gathered from.  Pass an empty fields list to only gather
        the box coordinates.
      scope: name scope.
      use_static_shapes: Whether to use an implementation with static shape
        gurantees.

    Returns:
      subboxlist: a BoxList corresponding to the subset of the input BoxList
      specified by indices
    Raises:
      ValueError: if specified field is not contained in boxlist or if the
        indices are not of type int32
    """
    with tf.name_scope(scope, 'Gather'):
        if len(indices.shape.as_list()) != 1:
            raise ValueError('indices should have rank 1')
        if indices.dtype != tf.int32 and indices.dtype != tf.int64:
            raise ValueError('indices should be an int32 / int64 tensor')
        gather_op = tf.gather
        if use_static_shapes:
            gather_op = ops.matmul_gather_on_zeroth_axis
        subboxlist = box_list.BoxList(gather_op(boxlist.get(), indices))
        if fields is None:
            fields = boxlist.get_extra_fields()
        fields += ['boxes']
        for field in fields:
            if not boxlist.has_field(field):
                raise ValueError('boxlist must contain all specified fields')
            subfieldlist = gather_op(boxlist.get_field(field), indices)
            subboxlist.add_field(field, subfieldlist)
        return subboxlist
```
### 13. concatenate，连接多个boxlist的data数据，形成新的BoxList
```python
def concatenate(boxlists, fields=None, scope=None):
    """Concatenate list of BoxLists.

    This op concatenates a list of input BoxLists into a larger BoxList.  It also
    handles concatenation of BoxList fields as long as the field tensor shapes
    are equal except for the first dimension.

    Args:
      boxlists: list of BoxList objects
      fields: optional list of fields to also concatenate.  By default, all
        fields from the first BoxList in the list are included in the
        concatenation.
      scope: name scope.

    Returns:
      a BoxList with number of boxes equal to
        sum([boxlist.num_boxes() for boxlist in BoxList])
    Raises:
      ValueError: if boxlists is invalid (i.e., is not a list, is empty, or
        contains non BoxList objects), or if requested fields are not contained in
        all boxlists
    """
    with tf.name_scope(scope, 'Concatenate'):
        if not isinstance(boxlists, list):
            raise ValueError('boxlists should be a list')
        if not boxlists:
            raise ValueError('boxlists should have nonzero length')
        for boxlist in boxlists:
            if not isinstance(boxlist, box_list.BoxList):
                raise ValueError('all elements of boxlists should be BoxList objects')
        concatenated = box_list.BoxList(
            tf.concat([boxlist.get() for boxlist in boxlists], 0))
        if fields is None:
            fields = boxlists[0].get_extra_fields()
        for field in fields:
            first_field_shape = boxlists[0].get_field(field).get_shape().as_list()
            first_field_shape[0] = -1
            if None in first_field_shape:
                raise ValueError('field %s must have fully defined shape except for the'
                                 ' 0th dimension.' % field)
            for boxlist in boxlists:
                if not boxlist.has_field(field):
                    raise ValueError('boxlist must contain all requested fields')
                field_shape = boxlist.get_field(field).get_shape().as_list()
                field_shape[0] = -1
                if field_shape != first_field_shape:
                    raise ValueError('field %s must have same shape for all boxlists '
                                     'except for the 0th dimension.' % field)
            concatenated_field = tf.concat(
                [boxlist.get_field(field) for boxlist in boxlists], 0)
            concatenated.add_field(field, concatenated_field)
        return concatenated
```
这个里面，要求所有的`fields`在所有的boslist变量里面存在。

### 14. sort\_by\_field，对制定的field进行排序
```python
def sort_by_field(boxlist, field, order=SortOrder.descend, scope=None):
    """Sort boxes and associated fields according to a scalar field.

    A common use case is reordering the boxes according to descending scores.

    Args:
      boxlist: BoxList holding N boxes.
      field: A BoxList field for sorting and reordering the BoxList.
      order: (Optional) descend or ascend. Default is descend.
      scope: name scope.

    Returns:
      sorted_boxlist: A sorted BoxList with the field in the specified order.

    Raises:
      ValueError: if specified field does not exist
      ValueError: if the order is not either descend or ascend
    """
    with tf.name_scope(scope, 'SortByField'):
        if order != SortOrder.descend and order != SortOrder.ascend:
            raise ValueError('Invalid sort order')

        field_to_sort = boxlist.get_field(field)
        if len(field_to_sort.shape.as_list()) != 1:
            raise ValueError('Field should have rank 1')

        num_boxes = boxlist.num_boxes()
        num_entries = tf.size(field_to_sort)
        length_assert = tf.Assert(
            tf.equal(num_boxes, num_entries),
            ['Incorrect field size: actual vs expected.', num_entries, num_boxes])

        with tf.control_dependencies([length_assert]):
            _, sorted_indices = tf.nn.top_k(field_to_sort, num_boxes, sorted=True)

        if order == SortOrder.ascend:
            sorted_indices = tf.reverse_v2(sorted_indices, [0])

        return gather(boxlist, sorted_indices)
```
我原来不知道在tensorflow里面怎么对tensor的值进行sort，现在发现其实`tf.nn.top_k`可以实现这个点，这个函数只能接受值sort，所以`sort_by_field`这个函数实现的过程中有个assert：

```python
num_boxes = boxlist.num_boxes()
num_entries = tf.size(field_to_sort)
length_assert = tf.Assert(
    tf.equal(num_boxes, num_entries),
    ['Incorrect field size: actual vs expected.', num_entries, num_boxes])

with tf.control_dependencies([length_assert]):
    _, sorted_indices = tf.nn.top_k(field_to_sort, num_boxes, sorted=True)
```
也就是boxlist里面的`num_boxes`必须要和field里面的数值个数相同。

### 15. visualize\_boxes\_in\_image，在图像中可视化Bbox
```python
def visualize_boxes_in_image(image, boxlist, normalized=False, scope=None):
    """Overlay bounding box list on image.

    Currently this visualization plots a 1 pixel thick red bounding box on top
    of the image.  Note that tf.image.draw_bounding_boxes essentially is
    1 indexed.

    Args:
      image: an image tensor with shape [height, width, 3]
      boxlist: a BoxList
      normalized: (boolean) specify whether corners are to be interpreted
        as absolute coordinates in image space or normalized with respect to the
        image size.
      scope: name scope.

    Returns:
      image_and_boxes: an image tensor with shape [height, width, 3]
    """
    with tf.name_scope(scope, 'VisualizeBoxesInImage'):
        if not normalized:
            height, width, _ = tf.unstack(tf.shape(image))
            boxlist = scale(boxlist,
                            1.0 / tf.cast(height, tf.float32),
                            1.0 / tf.cast(width, tf.float32))
        corners = tf.expand_dims(boxlist.get(), 0)
        image = tf.expand_dims(image, 0)
        return tf.squeeze(tf.image.draw_bounding_boxes(image, corners), [0])
```
这个函数很有用，就是在`tf.summary`的时候，可视化中间任何图像对应的bbox。这里有个`normalized`用来表示当前的bbox是不是被归一化了，因为`tf.image.draw_bounding_boxes`只接受相对坐标值。

### 16 filter\_field\_value\_equals, filter\_scores\_greater\_than获取值相等的field或者大于的某个阈值的Bbox
```python
def filter_field_value_equals(boxlist, field, value, scope=None):
    """Filter to keep only boxes with field entries equal to the given value.

    Args:
      boxlist: BoxList holding N boxes.
      field: field name for filtering.
      value: scalar value.
      scope: name scope.

    Returns:
      a BoxList holding M boxes where M <= N

    Raises:
      ValueError: if boxlist not a BoxList object or if it does not have
        the specified field.
    """
    with tf.name_scope(scope, 'FilterFieldValueEquals'):
        if not isinstance(boxlist, box_list.BoxList):
            raise ValueError('boxlist must be a BoxList')
        if not boxlist.has_field(field):
            raise ValueError('boxlist must contain the specified field')
        filter_field = boxlist.get_field(field)
        gather_index = tf.reshape(tf.where(tf.equal(filter_field, value)), [-1])
        return gather(boxlist, gather_index)
```
这两个函数相对简单，就是获取对应field的值等于或者大于某个值的Bbox。

### 17. pad\_or\_clip\_box\_list, 填充或者补足对应的box的长度
```python
def pad_or_clip_box_list(boxlist, num_boxes, scope=None):
    """Pads or clips all fields of a BoxList.

    Args:
      boxlist: A BoxList with arbitrary of number of boxes.
      num_boxes: First num_boxes in boxlist are kept.
        The fields are zero-padded if num_boxes is bigger than the
        actual number of boxes.
      scope: name scope.

    Returns:
      BoxList with all fields padded or clipped.
    """
    with tf.name_scope(scope, 'PadOrClipBoxList'):
        subboxlist = box_list.BoxList(shape_utils.pad_or_clip_tensor(
            boxlist.get(), num_boxes))
        for field in boxlist.get_extra_fields():
            subfield = shape_utils.pad_or_clip_tensor(
                boxlist.get_field(field), num_boxes)
            subboxlist.add_field(field, subfield)
        return subboxlist
```
