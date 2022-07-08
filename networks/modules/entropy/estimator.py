# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import tensorflow as tf


class Entropy(object):

    def __init__(self, name="base"):

        self.name = name


    def entropy_z(self, z):

        with tf.variable_scope(self.name+"_z"):
            [_, _, _, num_channel] = z.get_shape().as_list() # 这里需要固定的常量，使用get_shape
            sigma = tf.get_variable('sigma_z', [num_channel], tf.float32, tf.random_uniform_initializer(-1,1))
            mu = tf.get_variable('mu_z', [num_channel], tf.float32, initializer=tf.zeros_initializer())
            # z = tf.expand_dims(z, -1)
            laplace = tf.contrib.distributions.Laplace(mu, tf.exp(sigma))
            probs = laplace.cdf(z + 0.5) - laplace.cdf(z - 0.5)
            totbal_bits = tf.reduce_sum(-1.0 * tf.log(probs+1e-8) / tf.log(2.0))
        
            # 下面的几个变量用于实际编解码中将计算尽可能放在GPU中进行
            # self.z_plus = tf.add(z, 255, "z_plus")
            # self.mu_z_plus = tf.add(mu, 255, "mu_z_plus")
            # self.sigma_z_exp = tf.exp(tf.clip_by_value(sigma, -10, 11), "sigma_z_exp")
        
        return totbal_bits


    def entropy_y(self, y_hat, mu_sigma):

        [mu, sigma] = tf.split(mu_sigma, 2, axis=-1, name='split')
        # 下面的几个变量用于实际编解码中将计算尽可能放在GPU中进行        
        # self.y_plus = tf.add(y_hat, 255, "y_hat_plus")
        # self.mu_plus = tf.add(mu, 255, "mu_plus")
        # self.sigma_exp = tf.exp(tf.clip_by_value(sigma, -10, 11), "sigma_exp")

        sigma = tf.exp(sigma)  # add on 2018/10/22
        sigma = tf.clip_by_value(sigma, 1e-10, 1e50)

        gaussian = tf.contrib.distributions.Laplace(mu, sigma)
        probs = gaussian.cdf(y_hat + 0.5) - gaussian.cdf(y_hat - 0.5)

        total_bits = tf.reduce_sum(tf.clip_by_value(-1.0 * tf.log(probs + 1e-10) / tf.log(2.0), 0, 50))

        return total_bits