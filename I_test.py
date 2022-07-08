# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os
import sys
import argparse
import numpy as np
from time import time
from PIL import Image
import tensorflow as tf
from datetime import datetime

from I_config import IConfig
from networks.I_model import IModel
from utils.dataloader import Dataloader
from utils.utils import load_single_image, str2bool


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

    #model and test config
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(["%d"%id for id in IConfig.gpus_test])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    batch_size = 1
    # IConfig.test_set_dir = "../data/1080P/"
    # IConfig.test_comment = "test_for_valid"
    #built the model
    IMod=IModel(is_train=False)
    graph, sess = IMod.build()

    if not os.path.isdir(IConfig.info_dir):
        os.makedirs(IConfig.info_dir)

    # alpha_list = [1, 0.9, 0.7, 0.5, 0.3, 0.1]
    M = 5 # Interpolation parameters
    alpha_list = [1-j/M for j in range(5)]
    #open graph context manager
    print("imgaeGenerator")
    
    bpp_rd = [] # 多码率用来保存不同lambda下的结果
    psnr_rd = []
    ssim_rd = []
    with graph.as_default():
        
        print("the test dir is %s" % IConfig.test_set_dir)
        test_loader = Dataloader(IConfig.test_set_dir, 8, batch_size, model_type="test") # pipeline的方式
        test_len = test_loader.file_len
        test_filenames = test_loader.test_filenames
        init_test = test_loader.initializer
        test_next = test_loader.test_image_batch

        for index_random in range(len(IConfig.lambda_list)-1): 
            # index_random = 8 
            for alpha_rand in alpha_list: 
                # alpha_rand = 0.7
                if IConfig.is_multi:
                    l_onehot = alpha_rand*IConfig.lambda_onehot[index_random] + (1-alpha_rand)*IConfig.lambda_onehot[index_random+1]
                    lambda_test = alpha_rand*IConfig.lambda_list[index_random] + (1-alpha_rand)*IConfig.lambda_list[index_random+1]
                    IConfig.test_comment = "lambda_%s"%(lambda_test)
                else:
                    lambda_test = IConfig.train_lambda     
                    IConfig.test_comment = "lambda_%s"%(lambda_test)

                sess.run(init_test) #  每次都重新开始初始化test iteration迭代器，重头开始

                average_psnr = 0.
                average_bpp = 0.
                average_estbpp = 0.
                average_msssim = 0.

                if not os.path.isdir(IConfig.rescon_dir):
                    os.makedirs(IConfig.rescon_dir)
                if not os.path.isdir(IConfig.bin_dir):
                    os.makedirs(IConfig.bin_dir)

                ####################################################################### 
                for i in range(test_len):

                    tic = time()
                    input_image_batch = sess.run(test_next)
                    image_shape = np.shape(input_image_batch)
                    image_name = os.path.basename(test_filenames[i])

                    if IConfig.is_multi:
                        feed_dict={IMod.input_image_in:input_image_batch, IMod.lambda_onehot:l_onehot}
                    else:
                        feed_dict={IMod.input_image_in:input_image_batch}

                    psnr, ms_ssim, bpp, bpp_y, recon_image = sess.run([IMod.psnr, IMod.ms_ssim, IMod.bpp, IMod.bpp_y, IMod.clip_recon_image], feed_dict=feed_dict)
                    tic1 = time()

                    #########################调用最新的编码文件###################################################
                    # bin_path = IConfig.bin_dir + image_name.replace(".png",".bin")
                    actual_total_bits = 0 # 这里还没有开始写实际编解码函数-entropy_encoding
                    actual_bpp = actual_total_bits / (batch_size * image_shape[1] * image_shape[2])
                    tic2 = time()

                    average_psnr += psnr
                    # average_bpp += actual_bpp
                    average_estbpp += bpp
                    average_msssim += ms_ssim

                    # 将训练后的图像保存到data-recon-mini512中
                    clipped_recon_image = (np.round(recon_image*255)).astype(np.uint8)
                    # print('The shape is Recon', clipped_recon_image.shape)
                    Image.fromarray(clipped_recon_image[0]).save(IConfig.rescon_dir+'recon_'+image_name)
                    nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%2S')
                    print("%s %s : the nn time is %.2f s, the entropy time is %.2f s "%(nowtime, image_name, tic1-tic, tic2-tic1), image_shape)
                    print("id: %d, psnr: %.2f, ms-ssim: %.4f, Actual-bpp: %.4f, Val-bpp: %.4f, bpp_feature: %.4f" % \
                        (i + 1, psnr, ms_ssim, actual_bpp, bpp, bpp_y))
                    # txt_write[image_name] = [psnr_val, ms_ssim_np, 0, bpp_val]
                    if i == 0:
                        file = open(IConfig.datatext_dir, 'a+')
                        file.write("\n%s, %.2f, %.4f, %.4f, %.4f\n" % (image_name, psnr, ms_ssim, actual_bpp, bpp))
                        file.close()
                bpp_rd.append(average_estbpp / test_len)
                psnr_rd.append(average_psnr / test_len)
                ssim_rd.append(average_msssim / test_len)
                print("lambda:%.2f, PSNR:%.3f, msssim:%.4f, estbpp:%.4f, total_bpp:%.4f\n\n" % (
                    lambda_test,
                    average_psnr / test_len,
                    average_msssim / test_len,
                    average_estbpp / test_len,
                    average_bpp / test_len))
                file = open(IConfig.datatext_dir, 'a+')
                file.write("the number of IMod is %s %s\n" % (IMod.module_file, IConfig.test_comment))
                file.write(
                    "PSNR:%.3f, msssim:%.4f, estbpp:%.4f, total_bpp:%.4f\n\n" % (
                        average_psnr / test_len,
                        average_msssim / test_len,
                        average_estbpp / test_len,
                        average_bpp / test_len))
                file.close()
                ## 在非多码率模型时，要跳出多码率的循环
                if not IConfig.is_multi:
                    break
            if not IConfig.is_multi:
                break

    if IConfig.is_multi:
        file1 = open(os.path.join(IConfig.info_dir, "RD.txt"), 'a+')
        file1.write("the number of IMod is %s\nbpp = [" % (IMod.module_file))
        for bpp_s in bpp_rd:
            file1.write("%.8f, "%(bpp_s))
        file1.write("];\nPSNR = [")
        for psnr_s in psnr_rd:
            file1.write("%.8f, "%(psnr_s))
        file1.write("];\nSSIM = [")
        for ssim_s in ssim_rd:
            file1.write("%.8f, "%(ssim_s))
        file1.write("];\n\n")
        file1.close()

if __name__ == '__main__':
    tf.app.run()
