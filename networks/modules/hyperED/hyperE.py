# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import tensorflow as tf

from networks.modules.ops import print_tensor


class HyperEncoder(object):

	def __init__(self, is_train=True, name="HE", ch_E=[192, 192, 192]):

		self.is_train = is_train
		self.name = name
		self.ch_E = ch_E


	def __call__(self, features, kernel_size=5):
		# quantize inputs
		with tf.variable_scope(self.name):
			flow = features
			with tf.variable_scope('flow0'):
				flow = tf.layers.conv2d(inputs=flow, filters= self.ch_E[0], kernel_size=3, padding="same",
									activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name = "conv_0")
				# print_tensor(flow)

			with tf.variable_scope('down1'):
				flow = tf.layers.conv2d(inputs=flow, filters= self.ch_E[1], kernel_size=kernel_size, strides=(2, 2), padding="same",
									activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name = "conv_1")
				# print_tensor(flow)

			with tf.variable_scope('down2'):
				flow = tf.layers.conv2d(inputs=flow, filters= self.ch_E[2], kernel_size=kernel_size, strides=(2, 2), padding="same",
									kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name = "conv_2")
				# print_tensor(flow)

		return flow