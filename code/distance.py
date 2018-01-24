# -*- coding: UTF-8 -*- 
# Authorized by Vlon Jang
# Created on 2018-01-24
# Blog: www.wangqingbaidu.cn
# Email: wangqingbaidu@gmail.com
# From kwai, www.kuaishou.com
# ©2015-2018 All Rights Reserved.
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def check_dim(x, dim):
    assert x == dim, 'Dimension is not equal. x=%d, dim=%d' %(x, dim)

def cosine_distance(x, y):
    """Compute cosine distance between two tensor."""
    x_shape = x.get_shape()
    y_shape = y.get_shape()
    check_dim(len(x_shape), 2)
    check_dim(len(y_shape), 2)
    x_norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))
    y_norm = tf.sqrt(tf.reduce_sum(tf.square(y), axis=1))
    xy = tf.reduce_sum(tf.multiply(x, y), axis=1)
    cos_distance = xy / (x_norm * y_norm)
    return cos_distance

def cosine_distance_for_each_y(x, y):
    """Compute cosine distance between two tensor. y's tensor rank - x's tensor rank = 1."""
    x_shape = x.get_shape()
    y_shape = y.get_shape()
    check_dim(len(x_shape), 2)
    check_dim(len(y_shape), 3)
    batch_size = y_shape[0].value
    duplicate_num = y_shape[1].value
    assert len(x_shape) + 1 == len(y_shape), "#y(%d) - #x(%d) != 1." %(len(x_shape), len(y_shape))
    x_expanded = tf.tile(tf.expand_dims(x, axis=1), [1, duplicate_num, 1])
    dis = cosine_distance(tf.reshape(x_expanded, [batch_size * duplicate_num, -1]),
                          tf.reshape(y, [batch_size * duplicate_num, -1]))
    cos_distance = tf.reshape(dis, [batch_size, duplicate_num])
    return cos_distance