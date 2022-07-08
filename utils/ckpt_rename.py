# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'

from config import Config

import numpy as np
import tensorflow as tf

model_checkpoint_path = Config.checkpoint_dir # rename后的路径
checkpoint_dir=os.path.join(Config.checkpoint_dir, "ori/")
print("the original checkPoint_dir is %s" % checkpoint_dir)
checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
print("the state is %s" % checkpoint)

with tf.Session() as sess:
    for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
        var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
        new_name= var_name.replace("train_net_inference_one_pass", "train_net")

        if var_name.find("global_step")>=0:
            global_step = tf.contrib.framework.load_variable(checkpoint_dir, var_name)-1
            new_name = "train_net/" + var_name
            # print(new_name)
        #print(var)

        if new_name.find("context_mask_net")==-1:
            for i in range(4):
                if new_name.find("stage%d_num_biases"%i)>=0:
                    new_name = new_name.replace("stage%d_num_biases"%i, "conv_%d/bias"%i)
                if new_name.find("stage%d_num_weights"%i)>=0:
                    new_name = new_name.replace("stage%d_num_weights"%i, "conv_%d/kernel"%i)
                if new_name.find("stage%d_biases"%i)>=0:
                    new_name = new_name.replace("stage%d_biases"%i, "conv_%d/bias"%i)
                if new_name.find("stage%d_weights"%i)>=0:
                    new_name = new_name.replace("stage%d_weights"%i, "conv_%d/kernel"%i)
                # if new_name.find("train_net/synthesis_net/stage%d/conv_%d/kernel"%(i, i))>=0:
                #     var = var.transpose(0,1,3,2)
                #     print(var.shape)

        # if new_name.find("train_net/synthesis_prior_net/stage0/conv_0/kernel")>=0 or \
        #     new_name.find("train_net/synthesis_prior_net/stage1/conv_1/kernel")>=0 or \
        #         new_name.find("train_net/synthesis_prior_net/stage2/conv_2/kernel")>=0:
        #     var = var.transpose(0,1,3,2)
        #     print(var.shape)

                
        print(new_name)
        var = tf.Variable(var, name=new_name)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    print("the acutal_step is %s" % (global_step))
    saver.save(sess, os.path.join(model_checkpoint_path, "model"), global_step=global_step, write_meta_graph=False)
