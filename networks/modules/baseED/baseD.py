# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import numpy as np
import tensorflow as tf
gdn = tf.contrib.layers.gdn

from networks.modules.ops import lambda_mask, print_tensor


class Decoder(object):

	def __init__(self, is_train=True, name="decoder", ch_D=[192, 192, 192, 3], is_multi=False):

		self.is_train = is_train
		self.name = name
		self.ch_D = ch_D
		self.is_multi = is_multi


	def __call__(self, features, kernel_size=5, lambda_onehot=None):

		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			flow = features

			with tf.variable_scope('up0'):
				flow = tf.layers.conv2d_transpose(inputs=flow, filters=self.ch_D[0], kernel_size=kernel_size, strides=(2, 2), padding="same",
									kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name = "conv_0")
				if self.is_multi:
					flow = lambda_mask(flow, lambda_onehot, layername="mask0")				
				# print_tensor(flow)

			with tf.variable_scope('up1'):
				flow = gdn(flow, inverse =True)
				flow = tf.layers.conv2d_transpose(inputs=flow, filters=self.ch_D[1], kernel_size=kernel_size, strides=(2, 2), padding="same",
									kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name = "conv_1")
				if self.is_multi:
					flow = lambda_mask(flow, lambda_onehot, layername="mask1")				
				# print_tensor(flow)

			with tf.variable_scope('up2'):
				flow = gdn(flow, inverse =True)
				flow = tf.layers.conv2d_transpose(inputs=flow, filters=self.ch_D[2], kernel_size=kernel_size, strides=(2, 2), padding="same",
									kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name = "conv_2")
				if self.is_multi:
					flow = lambda_mask(flow, lambda_onehot, layername="mask2")				
				# print_tensor(flow)

			with tf.variable_scope('up3'):
				flow = gdn(flow, inverse =True)
				flow = tf.layers.conv2d_transpose(inputs=flow, filters=self.ch_D[3], kernel_size=kernel_size, strides=(2, 2), padding="same",
									kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name = "conv_3")
				if self.is_multi:
					flow = lambda_mask(flow, lambda_onehot, layername="mask3")				
				# print_tensor(flow)

		return flow