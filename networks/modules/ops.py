# Copyright (c) 2021-2022 Alibaba Group Holding Limited.


import tensorflow as tf


def print_tensor(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def _arr(stride_or_ksize):
    # data format NHWC
    return [1, stride_or_ksize, stride_or_ksize, 1]


@tf.RegisterGradient("Lower_bound")
def lower_bound_grad(op, grad):
    inputs = op.inputs[0]
    bound = op.inputs[1]
    pass_through_if = tf.logical_or(inputs >= bound, grad < 0)
    return tf.cast(pass_through_if, grad.dtype) * grad, None
	

def gdn_adjust(flow, inverse=False, coeff=1e4):

	with tf.variable_scope("gdn"):
		feature = flow #
		# if is_quan:
		#     feature = insertFakeQuant(feature, is_training=is_training)
		#     variable_summaries(feature, "feature_quan", "gdn")
		shape_flow = flow.get_shape().as_list()

		gamma_shape = [shape_flow[-1], shape_flow[-1]]
		gamma = tf.get_variable(name='reparam_gamma', shape=gamma_shape, initializer=tf.variance_scaling_initializer)# 修正的gamma
		beta = tf.get_variable(name='reparam_beta', shape=[shape_flow[-1]], initializer=tf.variance_scaling_initializer)# 修正的beta

		# """
		beta_min = 0.0010000072652474046
		gamma_min = 0.000003814697265625
		const = 1.4551915228366852e-11
		# """

		with tf.get_default_graph().gradient_override_map({"Maximum": "Lower_bound"}):
			gamma_lower_bound = tf.maximum(gamma, gamma_min)
			beta_lower_bound = tf.maximum(beta, beta_min)

		gamma_square = tf.subtract(tf.square(gamma_lower_bound), const)
		gamma_square = tf.reshape(gamma_square, [1, 1, shape_flow[-1], shape_flow[-1]])
		beta_square = tf.subtract(tf.square(beta_lower_bound), const)/coeff
		# if is_quan:
		#     beta_square = insertFakeQuant(beta_square, is_training=is_training)
		#     variable_summaries(flow, "beta_square_quan", "gdn")

		flow = tf.square(feature)
		flow = tf.nn.conv2d(flow, gamma_square, _arr(1), padding="VALID")
		flow = tf.nn.bias_add(flow, beta_square)
		# 会出现nan
		# if is_quan:
		#     flow = insertFakeQuant(flow, is_training=is_training)
		#     variable_summaries(flow, "conv_quan", "gdn")

		flow = tf.sqrt(flow)
		if inverse:
			flow = tf.multiply(flow, feature)
		else:
			flow = tf.divide(feature, flow)
		return flow


def lambda_mask(features, lambda_onehot, layername="lambda_mask"):

		with tf.variable_scope(layername):
			flow = features
			flow_shape = features.get_shape().as_list()

			mask_w = tf.layers.dense(lambda_onehot, flow_shape[-1], kernel_initializer=tf.ones_initializer, activation=tf.nn.softplus, use_bias=False, name="mask_w")
			# mask_b = tf.layers.dense(self.lambda_onehot, flow_shape[-1], kernel_initializer=tf.zeros_initializer, use_bias=False, name="mask_b")
			flow = tf.multiply(flow, mask_w)
			# flow = tf.multiply(flow, mask_w) + mask_b
			# Config.list_w.append(mask_w)
			# Config.list_b.append(mask_b)
			
		return flow