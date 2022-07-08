# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os
import sys
import glob
import argparse
import numpy as np
from time import time
from PIL import Image
import tensorflow as tf
from datetime import datetime

from IP_config import PConfig
from networks.IP_model import PModel
from utils.dataloader import Dataloader
from utils.utils import load_single_image, str2bool


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--loss_metric', type=str, default="PSNR", help='loss_metric: PSNR or SSIM')
    parser.add_argument('--model_name', type=str, default="STPM", help='loss_metric: PSNR or SSIM')
    parser.add_argument('--work_dir', type=str, default=None, help='the dir to save logs and models, load the models')
    parser.add_argument('--is_post', type=str2bool, default=True, help='add the Unet post network')
    parser.add_argument('--with_context_model', type=str2bool, default=True, help='add the context model network')
    parser.add_argument('--is_multi', type=str2bool, default=True, help='enable variable rate control')
    parser.add_argument('--seed', type=int, default=1000, help='random seed')
    parser.add_argument('--idx_test', type=int, default=0, help='idx_test')
    parser.add_argument('--ckpt_dir_pre', type=str, default=None, help='I-frame pretrained models are needed for P-frame compression')
    args = parser.parse_args()
    return args


def main(unused_argv):
    args = parse_args()
    PConfig.cckpt(args)
    print(args)
    print(PConfig)

    #model and test config
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(["%d"%id for id in PConfig.gpus_test])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    batch_size = 1
    PConfig.fix_size =[1, 1080, 1920*3, 3]
    # PConfig.test_set_dir = "../data/1080P/"
    # PConfig.test_comment = "test_for_valid"
    #built the model
    PMod=PModel(is_train=False)
    graph, sess = PMod.build()

    if not os.path.isdir(PConfig.info_dir):
        os.makedirs(PConfig.info_dir)

    alpha_list = [1, 0.9, 0.7, 0.5, 0.3, 0.1]
    #open graph context manager
    print("imgaeGenerator")
    
    with graph.as_default():
        bpp_avg = 0 # 不同序列间测试均值
        bpp_res_avg = 0
        bpp_min_avg = 0
        psnr_avg = 0
        ssim_avg = 0
        PConfig.test_seq_names.sort()
        for seq_name in PConfig.test_seq_names:
            bpp_rd = [] # 多码率用来保存不同lambda下的结果
            psnr_rd = []
            ssim_rd = []
            bpp_res_rd = []
            bpp_min_rd = []
            
            print("the test seq_name is %s" % seq_name)
            file = open(PConfig.datatext_dir, 'a+')
            file.write("\n#######################################################\n")
            file.write("Below is %s sequence test results (1I%dP)\n"%(seq_name, PConfig.test_GOP-1))
            file.write("#######################################################\n")
            file.close()
            ori_seq_dir = os.path.join(PConfig.test_set_dir, seq_name)
            # I_seq_dir = os.path.join(PConfig.test_I_dir, seq_name)
            # I_pathes = glob.glob(os.path.join(I_seq_dir, "*.png"))
            # I_pathes.sort()
            # info_seq_path = os.path.join(PConfig.test_I_dir, "info", "%s.npy"%seq_name)
            # info_seq = np.load(info_seq_path)
            
            test_loader = Dataloader(ori_seq_dir, 8, batch_size, model_type="test") # pipeline的方式
            test_len = test_loader.file_len
            test_filenames = test_loader.test_filenames
            init_test = test_loader.initializer
            test_next = test_loader.test_image_batch

            for index_random in range(len(PConfig.lambda_list)-1): 
                for alpha_rand in alpha_list: 
                    # alpha_rand = 0.7
                    if PConfig.is_multi:
                        # l_onehot = alpha_rand*PConfig.lambda_onehot[index_random] + (1-alpha_rand)*PConfig.lambda_onehot[index_random+1]
                        # lambda_test = alpha_rand*PConfig.lambda_list[index_random] + (1-alpha_rand)*PConfig.lambda_list[index_random+1]
                        l_onehot = PConfig.lambda_onehot[PConfig.idx_test]
                        lambda_test = PConfig.lambda_list[PConfig.idx_test]
                        PConfig.test_comment = "lambda_%d"%(lambda_test)
                    else:
                        lambda_test = PConfig.train_lambda
                        PConfig.test_comment = "lambda_%d"%(lambda_test)

                    sess.run(init_test) #  每次都重新开始初始化test iteration迭代器，重头开始

                    average_psnr = 0.
                    average_bpp = 0.
                    average_estbpp = 0.
                    average_estbpp_res = 0.
                    average_estbpp_min = 0.
                    average_msssim = 0.

                    if not os.path.isdir(PConfig.rescon_dir):
                        os.makedirs(PConfig.rescon_dir)
                    if not os.path.isdir(PConfig.bin_dir):
                        os.makedirs(PConfig.bin_dir)
                    if not os.path.isdir(os.path.join(PConfig.rescon_dir, seq_name)):
                        os.makedirs(os.path.join(PConfig.rescon_dir, seq_name))

                    ####################################################################### 
                    for i in range(test_len):
                    # for i in range(3):

                        tic = time()
                        if i%PConfig.test_GOP == 0: # I帧直接读取已经完成的编解码信息
                            frame_flag = "I"
                            cur_img_batch = sess.run(test_next)
                            image_name = os.path.basename(test_filenames[i])

                            input_image_batch = np.concatenate((cur_img_batch, cur_img_batch, cur_img_batch), axis=2)
                            if PConfig.is_multi:
                                feed_dict={PMod.input_image_in:input_image_batch, PMod.lambda_onehot:l_onehot}
                            else:
                                feed_dict={PMod.input_image_in:input_image_batch}
                            psnr, ms_ssim, bpp, bpp_y = sess.run([PMod.psnr, PMod.ms_ssim, PMod.bpp, PMod.bpp_y], feed_dict=feed_dict)
                            bpp_res = bpp
                            bpp_min = np.minimum(bpp, bpp_res)
                        else:
                            frame_flag = "P"
                            cur_img_batch = sess.run(test_next)
                            # pre_img_recon_batch = clipped_recon_image
                            image_shape = np.shape(cur_img_batch)
                            image_name = os.path.basename(test_filenames[i])
                            image_recon_path = os.path.join(PConfig.rescon_dir, seq_name, image_name)

                            input_image_batch = np.concatenate((pre_img_batch, pre_img_batch, cur_img_batch), axis=2)
                            if PConfig.is_multi:
                                feed_dict={PMod.input_image_in:input_image_batch, PMod.lambda_onehot:l_onehot}
                            else:
                                feed_dict={PMod.input_image_in:input_image_batch}

                            psnr, ms_ssim, bpp, bpp_res, bpp_y_res, recon_image = sess.run([PMod.psnr, PMod.ms_ssim, PMod.bpp, PMod.bpp_res, PMod.bpp_y_res, PMod.clip_recon_image], feed_dict=feed_dict)
                            tic1 = time()
                            bpp_min = np.minimum(bpp, bpp_res)
                            print(bpp_res, bpp_y_res, bpp_res-bpp_y_res)
                            #########################调用最新的编码文件###################################################
                            # bin_path = PConfig.bin_dir + image_name.replace(".png",".bin")
                            actual_total_bits = 0 # 这里还没有开始写实际编解码函数-entropy_encoding
                            actual_bpp = actual_total_bits / (batch_size * image_shape[1] * image_shape[2])
                            tic2 = time()

                            # 将训练后的图像保存到data-recon-mini512中
                            # print('The shape is Recon', clipped_recon_image.shape)
                            clipped_recon_image = (np.round(recon_image*255)).astype(np.uint8)
                            Image.fromarray(clipped_recon_image[0]).save(image_recon_path)
                            # nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%2S')
                            # print("%s %s : the nn time is %.2f s, the entropy time is %.2f s "%(nowtime, image_name, tic1-tic, tic2-tic1), image_shape)

                        pre_img_batch = cur_img_batch
                        
                        average_psnr += psnr
                        # average_bpp += actual_bpp
                        average_estbpp += bpp
                        average_estbpp_res += bpp_res
                        average_estbpp_min += bpp_min
                        average_msssim += ms_ssim

                        print("%dth frame (%s) result is %.4f, %.4f, %.4f, %.4f, %.4f (bpp, bpp_res, bpp_min, psnr, msssim)"%(i, frame_flag, bpp, bpp_res, bpp_min, psnr, ms_ssim))
                        # txt_write[image_name] = [psnr_val, ms_ssim_np, 0, bpp_val]
                        if i < 10:
                            file = open(PConfig.datatext_dir, 'a+')
                            file.write("%dth frame (%s) result is %.4f, %.4f, %.4f, %.4f, %.4f (bpp, bpp_res, bpp_min, psnr, msssim)\n"%(i, frame_flag, bpp, bpp_res, bpp_min, psnr, ms_ssim))
                            file.close()

                    bpp_rd.append(average_estbpp / test_len)
                    bpp_res_rd.append(average_estbpp_res / test_len)
                    bpp_min_rd.append(average_estbpp_min / test_len)
                    psnr_rd.append(average_psnr / test_len)
                    ssim_rd.append(average_msssim / test_len)
                    print("lambda:%d, %s sequence test result, PSNR:%.3f, msssim:%.4f, Ibpp:%.4f, Resbpp:%.4f, Minbpp:%.4f\n\n" % (
                        lambda_test, seq_name, 
                        average_psnr / test_len,
                        average_msssim / test_len,
                        average_estbpp / test_len,
                        average_estbpp_res / test_len,
                        average_estbpp_min / test_len))
                    file = open(PConfig.datatext_dir, 'a+')
                    file.write("the number of PMod is %s %s\n" % (PMod.module_file, PConfig.test_comment))
                    file.write("%s sequence test result is PSNR:%.3f, msssim:%.4f, Ibpp:%.4f, Resbpp:%.4f, Minbpp:%.4f\n\n" % (seq_name,
                            average_psnr / test_len,
                            average_msssim / test_len,
                            average_estbpp / test_len,
                            average_estbpp_res / test_len,
                            average_estbpp_min / test_len))
                    file.close()
                    ## 在非多码率模型时，要跳出多码率的循环
                    # if not PConfig.is_multi:
                    break
                # if not PConfig.is_multi:
                break

            # if PConfig.is_multi:
            #     file1 = open(PConfig.info_dir+"RD_%s.txt"%seq_name, 'a+')
            #     file1.write("the number of PMod is %s\nthe seq name is %s:\nbpp = [" % (PMod.module_file, seq_name))
            #     for bpp_s in bpp_rd:
            #         file1.write("%.4f, "%(bpp_s))
            #     file1.write("];\nPSNR = [")
            #     for psnr_s in psnr_rd:
            #         file1.write("%.3f, "%(psnr_s))
            #     file1.write("];\nSSIM = [")
            #     for ssim_s in ssim_rd:
            #         file1.write("%.4f, "%(ssim_s))
            #     file1.write("];\n\n")
            #     file1.close()

            bpp_avg += bpp_rd[0]
            bpp_res_avg += bpp_res_rd[0]
            bpp_min_avg += bpp_min_rd[0]
            psnr_avg += psnr_rd[0]
            ssim_avg += ssim_rd[0]
        print("lambda:%d, the final test result is %.4f, %.4f, %.4f, %.4f, %.4f (bpp, bpp_res, bpp_min, psnr, msssim)\n\n" % (
            lambda_test,
            bpp_avg / len(PConfig.test_seq_names),
            bpp_res_avg / len(PConfig.test_seq_names),
            bpp_min_avg / len(PConfig.test_seq_names),
            psnr_avg / len(PConfig.test_seq_names),
            ssim_avg / len(PConfig.test_seq_names)))
        file = open(PConfig.datatext_dir, 'a+')
        file.write("the final test result is %.4f, %.4f, %.4f, %.4f, %.4f (bpp, bpp_res, bpp_min, psnr, msssim)\n\n" % (
                bpp_avg / len(PConfig.test_seq_names),
                bpp_res_avg / len(PConfig.test_seq_names),
                bpp_min_avg / len(PConfig.test_seq_names),
                psnr_avg / len(PConfig.test_seq_names),
                ssim_avg / len(PConfig.test_seq_names)))
        file.close()
            
if __name__ == '__main__':
    tf.app.run()
