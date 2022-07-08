# Copyright (c) 2021-2022 Alibaba Group Holding Limited.


import tensorflow as tf
import numpy as np

from networks.modules.ops import print_tensor


class ContextModel(object):

	def __init__(self, is_train=True, name="context_mask"):

		self.is_train = is_train
		self.name = name


	def __call__(self, features):
		# quantize inputs
		[_, _, _, in_channel] = features.get_shape().as_list() # 这里需要固定的常量，使用get_shape
		with tf.variable_scope(self.name):
			flow = features

			kernelSize = [5, 5, in_channel, 2*in_channel]
			weights = tf.get_variable(name='context_weights', shape=kernelSize,
                                initializer=tf.contrib.layers.variance_scaling_initializer())
			biases = tf.get_variable(name='context_biases', shape=[kernelSize[3]],
                                initializer=tf.constant_initializer(0.01))
			# print_tensor(weights)
			kernel_mask = np.ones(shape=kernelSize, dtype=np.float32)
			kernel_mask[kernelSize[0]-1, kernelSize[1]//2:,...] = 0
			weights_masked = tf.multiply(kernel_mask, weights, name='context_weights_masked')
			# print_tensor(weights_masked)

			# with tf.name_scope('weight_display') as scope:
			# 	weights_image = tf.transpose(weights, [2, 0, 1, 3])
			# 	tf.summary.image('weights_image', weights_image[..., 0:1], 1)
			# 	tf.summary.histogram('weights_histogram', tf.clip_by_value(weights,-100 ,100))
			# 	weights_masked_image = tf.transpose(weights_masked, [2,0,1,3])
			# 	tf.summary.image('weights_image', weights_masked_image[..., 0:1], 1)
			# 	tf.summary.histogram('weights_masked_histogram', tf.clip_by_value(weights_masked,-100 ,100))
			k_size = kernelSize[1]

			paddings = [[0, 0], [(k_size - 1), 0], [(k_size - 1) // 2, (k_size - 1) // 2], [0, 0]]
			flow = tf.pad(flow, paddings, mode='CONSTANT')
			flow = tf.nn.conv2d(flow, weights_masked, [1, 1, 1, 1], padding='VALID', name='context_mask_conv')
			flow = tf.nn.bias_add(flow, biases, name='context__addBiases')

		return flow
