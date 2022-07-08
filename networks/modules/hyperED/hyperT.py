# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import tensorflow as tf

from networks.modules.ops import print_tensor


class HyperTransform(object):

	def __init__(self, is_train=True, name="Pmod_"):

		self.is_train = is_train
		self.name = name


	def __call__(self, features, kernel_size=5):
		# quantize inputs
		with tf.variable_scope(self.name):
			flow = features
			[_, _, _, in_ch] = features.get_shape().as_list() # 这里需要固定的常量，使用get_shape
			flow = tf.layers.conv2d(inputs=flow, filters= in_ch*4//3, kernel_size=kernel_size, padding="same",
								activation=tf.nn.leaky_relu,name = "conv_0")

			flow = tf.layers.conv2d(inputs=flow, filters= in_ch*5//3, kernel_size=kernel_size, padding="same",
								activation=tf.nn.leaky_relu,name = "conv_1")

			flow = tf.layers.conv2d(inputs=flow, filters= in_ch*2, kernel_size=kernel_size, padding="same", name = "conv_2")
			# print_tensor(flow)

		return flow