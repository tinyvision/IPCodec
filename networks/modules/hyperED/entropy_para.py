# Copyright (c) 2021-2022 Alibaba Group Holding Limited.


import numpy as np
import tensorflow as tf

from networks.modules.ops import print_tensor


class EntropyPara(object):

	def __init__(self, is_train=True, name="entropy_para", out_filter=None):

		self.is_train = is_train
		self.name = name
		self.out_filter = out_filter


	def __call__(self, features):
		# quantize inputs
		with tf.variable_scope(self.name):
			flow = features
			flow_shape = features.get_shape().as_list()
			flow = tf.layers.conv2d(inputs=flow, filters=flow_shape[-1]*5//6, kernel_size=1, padding="same",\
								activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name="conv_0")
				# print_tensor(flow)

			flow = tf.layers.conv2d(inputs=flow, filters=flow_shape[-1]*2//3, kernel_size=1, padding="same",\
								activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name="conv_1")
				# print_tensor(flow)
			
			if self.out_filter:
				flow = tf.layers.conv2d(inputs=flow, filters=self.out_filter, kernel_size=1, padding="same",\
								kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name="conv_2")
			else:
				flow = tf.layers.conv2d(inputs=flow, filters=flow_shape[-1]//2, kernel_size=1, padding="same",\
								kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name="conv_2")
				# print_tensor(flow)

		return flow