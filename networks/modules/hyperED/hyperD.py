# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import numpy as np
import tensorflow as tf

from networks.modules.ops import print_tensor


class HyperDecoder(object):

	def __init__(self, is_train=True, name="HD", up_type="depth2space", ch_D=[192, 192, 512]):
		"""[summary]

		Keyword Arguments:
			is_train {bool} -- [description] (default: {True})
			name {str} -- [description] (default: {"HD"})
			up_type {str} -- [description] (default: {"depth2space", "deconv"})
		"""

		self.is_train = is_train
		self.name = name
		self.up_type = up_type
		self.ch_D = ch_D


	def __call__(self, recon_z, kernel_size=5):

		with tf.variable_scope(self.name):
			flow = recon_z
			with tf.variable_scope('up0'):
				if self.up_type == "depth2space":
					flow = tf.layers.conv2d(inputs=flow, filters= self.ch_D[0]*4, kernel_size=kernel_size, padding="same",
										activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name = "conv_0")
					flow = tf.depth_to_space(flow, 2)
				if self.up_type == "deconv":
					flow = tf.layers.conv2d_transpose(inputs=flow, filters=self.ch_D[0], kernel_size=kernel_size, strides=(2, 2), padding="same",
										activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name = "conv_0")					
				# print_tensor(flow)
				
			out_N = np.round(1.5 * self.ch_D[1]).astype(np.int)
			with tf.variable_scope('up1'):
				if self.up_type == "depth2space":
					flow = tf.layers.conv2d(inputs=flow, filters= out_N*4, kernel_size=kernel_size, padding="same",
										activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name = "conv_1")
					flow = tf.depth_to_space(flow, 2)
				if self.up_type == "deconv":
					flow = tf.layers.conv2d_transpose(inputs=flow, filters=out_N, kernel_size=kernel_size, strides=(2, 2), padding="same",
										activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name = "conv_1")					
				# print_tensor(flow)

			with tf.variable_scope('flow2'):
				flow = tf.layers.conv2d(inputs=flow, filters= self.ch_D[2], kernel_size=3, padding="same",
									kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name = "conv_2")
				# print_tensor(flow)
				
		return flow