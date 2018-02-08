# -*- coding: UTF-8 -*- 
# Authorized by Vlon Jang
# Created on 2017-09-26
# Blog: www.wangqingbaidu.cn
# Email: wangqingbaidu@gmail.com
# From kwai, www.kuaishou.com
# ©2015-2017 All Rights Reserved.
#

"""
    Attention Model:
    WARNING: Use BatchNorm layer otherwise no accuracy gain.
    Lower layer with SpatialAttention, high layer with ChannelWiseAttention.
    In Visual155, Accuracy at 1, from 75.39% to 75.72%(↑0.33%).
"""
import tensorflow as tf
def spatial_attention(feature_map, K=1024, weight_decay=0.00004, scope="", reuse=None):
    """This method is used to add spatial attention to model.
    
    Parameters
    ---------------
    @feature_map: Which visual feature map as branch to use.
    @K: Map `H*W` units to K units. Now unused.
    @reuse: reuse variables if use multi gpus.
    
    Return
    ---------------
    @attended_fm: Feature map with Spatial Attention.
    """
    with tf.variable_scope(scope, 'SpatialAttention', reuse=reuse):
        # Tensorflow's tensor is in BHWC format. H for row split while W for column split.
        _, H, W, C = tuple([int(x) for x in feature_map.get_shape()])
        w_s = tf.get_variable("SpatialAttention_w_s", [C, 1],
                              dtype=tf.float32,
                              initializer=tf.initializers.orthogonal,
                              regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        b_s = tf.get_variable("SpatialAttention_b_s", [1],
                              dtype=tf.float32,
                              initializer=tf.initializers.zeros)
        spatial_attention_fm = tf.matmul(tf.reshape(feature_map, [-1, C]), w_s) + b_s
        spatial_attention_fm = tf.nn.sigmoid(tf.reshape(spatial_attention_fm, [-1, W * H]))
#         spatial_attention_fm = tf.clip_by_value(tf.nn.relu(tf.reshape(spatial_attention_fm, 
#                                                                       [-1, W * H])), 
#                                                 clip_value_min = 0, 
#                                                 clip_value_max = 1)
        attention = tf.reshape(tf.concat([spatial_attention_fm] * C, axis=1), [-1, H, W, C])
        attended_fm = attention * feature_map
        return attended_fm
    
def channel_wise_attention(feature_map, K=1024, weight_decay=0.00004, scope='', reuse=None):
    """This method is used to add spatial attention to model.
    
    Parameters
    ---------------
    @feature_map: Which visual feature map as branch to use.
    @K: Map `H*W` units to K units. Now unused.
    @reuse: reuse variables if use multi gpus.
    
    Return
    ---------------
    @attended_fm: Feature map with Channel-Wise Attention.
    """
    with tf.variable_scope(scope, 'ChannelWiseAttention', reuse=reuse):
        # Tensorflow's tensor is in BHWC format. H for row split while W for column split.
        _, H, W, C = tuple([int(x) for x in feature_map.get_shape()])
        w_s = tf.get_variable("ChannelWiseAttention_w_s", [C, C],
                              dtype=tf.float32,
                              initializer=tf.initializers.orthogonal,
                              regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        b_s = tf.get_variable("ChannelWiseAttention_b_s", [C],
                              dtype=tf.float32,
                              initializer=tf.initializers.zeros)
        transpose_feature_map = tf.transpose(tf.reduce_mean(feature_map, [1, 2], keep_dims=True), 
                                             perm=[0, 3, 1, 2])
        channel_wise_attention_fm = tf.matmul(tf.reshape(transpose_feature_map, 
                                                         [-1, C]), w_s) + b_s
        channel_wise_attention_fm = tf.nn.sigmoid(channel_wise_attention_fm)
#         channel_wise_attention_fm = tf.clip_by_value(tf.nn.relu(channel_wise_attention_fm), 
#                                                      clip_value_min = 0, 
#                                                      clip_value_max = 1)
        attention = tf.reshape(tf.concat([channel_wise_attention_fm] * (H * W), 
                                         axis=1), [-1, H, W, C])
        attended_fm = attention * feature_map
        return attended_fm
