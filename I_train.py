# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os
import sys
import random
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from time import time, sleep
from datetime import datetime

from I_config import IConfig
from networks.I_model import IModel
from utils.utils import load_single_image, generate_weight_list, str2bool


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--loss_metric', type=str, default="PSNR", help='loss_metric: PSNR or SSIM')
    parser.add_argument('--model_name', type=str, default="CM", help='loss_metric: PSNR or SSIM')
    parser.add_argument('--work_dir', type=str, default=None, help='the dir to save logs and models, load the models')
    parser.add_argument('--is_post', type=str2bool, default=True, help='add the Unet post network')
    parser.add_argument('--with_context_model', type=str2bool, default=True, help='add the context model network')
    parser.add_argument('--is_multi', type=str2bool, default=True, help='enable variable rate control')
    parser.add_argument('--seed', type=int, default=1000, help='random seed')
    args = parser.parse_args()
    return args


def main(unused_argv):
    args = parse_args()
    IConfig.cckpt(args)
    print(args)
    print(IConfig)
    
    #model and train config
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(["%d"%id for id in IConfig.gpus_list])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if not os.path.isdir(IConfig.info_dir):
        os.makedirs(IConfig.info_dir)

    show_step = IConfig.show_step
    total_batch_size = IConfig.total_batch_size

    #built the IModel
    IMod = IModel(is_train=True)
    graph, sess = IMod.build()
    steps_per_epoch = IMod.train_len//total_batch_size
    valid_len = IMod.valid_len
    batch_num = int(IConfig.total_batch_size/len(IConfig.gpus_list))

    #open graph context manager
    with graph.as_default():
        
        actual_step = 0
        tic = time()

        # 使用pipeline的时候需要调用该线程启动
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if IConfig.is_multi:
            weight_alpha_list = generate_weight_list(IConfig.alpha[0], IConfig.alpha[1])
            print("the weighted alpha list is \n", weight_alpha_list)

        for step in range(3000000):

            if actual_step>2500002:
                break

            if actual_step % 5000 == 0 and actual_step > 0:
                IMod.ckpt_write(actual_step)
                
            input_image = sess.run(IMod.train_next)
            # print("the input shape is ", np.shape(input_image))
            # input_image = load_single_image("./kodak/kodim01.png", is_zero_one=False) #True 为0-1； False为-1-1
            if IConfig.is_multi:
                index_rand = np.random.randint(0, len(IConfig.lambda_list)-1)
                alpha_rand = random.choice(weight_alpha_list)
                l_onehot = alpha_rand*IConfig.lambda_onehot[index_rand] + (1-alpha_rand)*IConfig.lambda_onehot[index_rand+1]

            if IConfig.gpus_list:
                feed_dict = {}
                for idx, _ in enumerate(IConfig.gpus_list):
                    if IConfig.is_multi:
                        feed_dict.update({IMod.tower_input[idx]:input_image[idx*batch_num:(idx+1)*batch_num], IMod.tower_onehot[idx]:l_onehot})
                    else:
                        feed_dict.update({IMod.tower_input[idx]:input_image[idx*batch_num:(idx+1)*batch_num]})
            else:
                # use CPU, not a good method
                if IConfig.is_multi:
                    feed_dict = ({IMod.input_image_in:input_image, IMod.lambda_onehot:l_onehot})
                else:
                    feed_dict = {IMod.input_image_in:input_image}
                
            try:
                actual_step, _ = sess.run([IMod.global_step, IMod.train_ops], feed_dict=feed_dict)
                # actual_step, recon_image, input_feature, weight_q= sess.run([IMod.global_step, IMod.recon_image, IMod.Net.H[1], IMod.Net.W_q[1]], feed_dict=feed_dict)
            except:
                print('the inf or NAN happened')
                continue

            # 训练基本，epoch输出
            train_choose = 'train_whole'
            sys.stdout.write("actual_step %d: epoch: %d, %.2f%% in a epoch, train_op: %s\r" % (
                actual_step, actual_step // steps_per_epoch, 100 * (actual_step % steps_per_epoch) / (steps_per_epoch),
                train_choose))
            sys.stdout.flush()

            if actual_step % show_step == 0:
                lambda_train, total_loss, psnr, ms_ssim, bpp, bpp_y, LR = sess.run([IMod.lambda_train, IMod.total_loss, IMod.psnr, IMod.ms_ssim, IMod.bpp, IMod.bpp_y, IMod.LR],feed_dict=feed_dict)
                toc = time()
                nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%2S')
                print("%s -- %d steps, %d epoch, fps %.2f, lambda:%.2f, total_loss:%.2e, psnr %.4f, ssim %.5f, bpp %.5f, bpp_y %.5f, LR %.5f" % (
                        nowtime, actual_step, actual_step // steps_per_epoch + 1, (toc - tic) / show_step, lambda_train, total_loss, psnr, ms_ssim, bpp, bpp_y, LR))
                with open(os.path.join(IConfig.info_dir, "/info.txt"), "a+") as f:
                    f.write("%s -- %d steps, %d epoch, fps %.2f, lambda:%.2f, total_loss:%.2e, psnr %.4f, ssim %.5f, bpp %.5f, bpp_y %.5f, LR %.5f" % (
                        nowtime, actual_step, actual_step // steps_per_epoch + 1, (toc - tic) / show_step, lambda_train, total_loss, psnr, ms_ssim, bpp, bpp_y, LR))
                    f.write("\n")

                tic = time()
                if step > 0:
                    IMod.summary_write(actual_step, feed_dict)

            if actual_step % 20000 == 0 and step > 1:

                psnr_all = 0.0
                msssim_all = 0.0
                total_loss_all = 0.0
                valid_bpp_all = 0.0
                valid_bpp_y_all = 0.0

                sess.run(IMod.init_valid) # 每次都重新开始初始化valid iteration迭代器，重头开始
                for iter in tqdm(range(int(valid_len / total_batch_size))):
                    try:
                        valid_image_batch = sess.run(IMod.valid_next)

                        if IConfig.is_multi:
                            feed_dict={IMod.input_image_in:valid_image_batch, IMod.lambda_onehot:IConfig.lambda_onehot[2]}
                        else:
                            feed_dict={IMod.input_image_in:valid_image_batch}

                        initial_learning_rate, total_loss, psnr, ms_ssim, bpp, bpp_y = sess.run([IMod.initial_learning_rate, 
                            IMod.total_loss, IMod.psnr, IMod.ms_ssim, IMod.bpp, IMod.bpp_y], feed_dict=feed_dict)
                        
                        psnr_all += psnr
                        msssim_all += ms_ssim
                        total_loss_all += total_loss
                        valid_bpp_all += bpp
                        valid_bpp_y_all += bpp_y

                    except tf.errors.OutOfRangeError:
                        print("Consume all valid samples ")
                        break
                     

                with open(os.path.join(IConfig.info_dir, "valid_log.txt"), "a+") as f:
                    f.write('test: steps %d, total_loss %.8f, psnr %.4f, ssim  %.6f, bpp: %.4f, bpp_y: %.4f, learng_rate: %.e, batch: %.d' %
                        (actual_step, total_loss_all / int(valid_len / total_batch_size),\
                        psnr_all / int(valid_len / total_batch_size),\
                        msssim_all / int(valid_len / total_batch_size),\
                        valid_bpp_all / int(valid_len / total_batch_size),\
                        valid_bpp_y_all / int(valid_len / total_batch_size),\
                        initial_learning_rate,\
                        total_batch_size))
                    f.write('\n')


if __name__ == '__main__':
    tf.app.run()
