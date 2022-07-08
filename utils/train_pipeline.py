# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import tensorflow as tf
import glob
import os
import random
import numpy as np


class InputPipeline(object):

	def __init__(self, train_dir, h, w, num_threads=16, batch_size=8, model_type="P"):
		"""[summary]

		Arguments:
			object {[type]} -- [description]

		Keyword Arguments:
			num_threads {int} -- [description] (default: {16})
			batch_size {int} -- [description] (default: {8})
			model_type {str} -- [description] (default: {"6P", "P", "I"})
			
		min_after_dequeue defines how big a buffer we will randomly sample
		from -- bigger means better shuffling but slower start up and more
		memory used.
		capacity must be larger than min_after_dequeue and the amount larger
		determines the maximum we will prefetch.  Recommendation:
		min_after_dequeue + (num_threads + a small safety margin) * batch_size
		
		"""
		self.model_type = model_type
		self.h = h
		self.w = w
		
		train_image = self.read_from_filename_queue(train_dir)
		train_image.set_shape((self.h, self.w, 3))
			
		min_after_dequeue = 32
		capacity = min_after_dequeue + (num_threads + 32) * batch_size
		self.train_image_batch = tf.train.shuffle_batch(
			[train_image], batch_size=batch_size, capacity=capacity,
			min_after_dequeue=min_after_dequeue, num_threads=num_threads)


	def read_from_filename_queue(self, train_dir):
		"""[summary]

		Arguments:
			train_dir {[type]} -- [description]

		Returns:
			train_image [type] -- [iteration]
		"""

		train_filenames= [y for y in glob.glob(os.path.join(train_dir, '*.png'))]
		random.shuffle(train_filenames)
		self.file_len = len(train_filenames)
		print("the number of train images: %d"%(self.file_len))
		train_filenames = tf.convert_to_tensor(train_filenames)
		train_filename_queue = tf.train.string_input_producer(train_filenames, num_epochs=None, shuffle=False)
		train_reader = tf.WholeFileReader()
		_, train_files = train_reader.read(train_filename_queue)
		train_image = tf.image.decode_png(train_files, channels=3)
		
		if self.model_type == "I":
			print("the pipiline type %s"%self.model_type)
			train_image = tf.random_crop(train_image, [self.h, self.w, 3])
			train_image = tf.image.random_flip_left_right(train_image)
		
		train_image = tf.image.random_flip_up_down(train_image)

		if self.model_type == "6P":
			# Input image consists of 1I1Rec6P. Randomly select 1P from 6P to simulate the big movement
			train_image_tf = tf.convert_to_tensor(train_image)
			p_idx = tf.random_uniform([], minval=1, maxval=8, dtype=tf.int32) 
			train_image = tf.concat([train_image_tf[:, 0*self.h:1*self.h, :],
									train_image_tf[:, 1*self.h:2*self.h, :],
									train_image_tf[:, p_idx*self.h:(p_idx+1)*self.h, :]], axis=1)

		return train_image


if __name__ == '__main__':
	pass
