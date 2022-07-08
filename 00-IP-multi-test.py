# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os
import re
import argparse
import threading
from math import ceil
from time import time
from IP_config import PConfig

parser = argparse.ArgumentParser(description="Resolution Adaptive Codec Network")

# 节点及GPU使用
parser.add_argument("--data", type=str, default="UVG", choices=["UVG", "HB", "HC", "HD", "HE", "MCL"],
                    help="test different data")    
parser.add_argument("--gop", type=int, default=10, choices=[2, 5, 6, 10, 12, 108],
                    help="test gop in video")    # UVG=12/108, HEVC=10/108
parser.add_argument("--frames", type=int, default=100, choices=[100, 108, 9999],
                    help="numbers of test frames in video") # UVG=9999, HEVC=100
parser.add_argument("--idx_test", type=int, default=0, choices=range(10),
                    help="idx")    
# [50, 160, 300, 480, 710, 1000, 1350, 1780, 2302, 2915]
parser.add_argument("--alpha_test", type=float, default=1,
                    help="alpha range 0-1")
parser.add_argument("--sw", type=int, default=2, choices=[0, 1, 2],
                    help="2: main & cal ; 1: main ; 0: cal")

opt = parser.parse_args()

if opt.data == "UVG":
    test_set_dir = "/home/your_path/Datasets/Videos/test/uvg_all_gen/uvg_1080p"
    Names = sorted(["ShakeNDry", "Beauty", "HoneyBee", "Jockey", "ReadySteadyGo", "Bosphorus", "YachtRide"])
elif opt.data == "HB":
    test_set_dir = "/home/your_path/Datasets/Videos/test/HB_all_gen/HB_1080p"
    Names = ["BasketballDrive", "Cactus", "ParkScene", "BQTerrace", "Kimono"]
elif opt.data == "HC":
    test_set_dir = "/home/your_path/Datasets/Videos/test/HC_all_gen/HC_480p"
    Names = ["BasketballDrill", "BQMall", "PartyScene", "RaceHorsesC"]
elif opt.data == "HD":
    test_set_dir = "/home/your_path/Datasets/Videos/test/HD_all_gen/HD_240p"
    Names = ["BasketballPass", "BlowingBubbles", "BQSquare", "RaceHorses"]
elif opt.data == "HE":
    test_set_dir = "/home/your_path/Datasets/Videos/test/HE_all_gen/HE_720p"
    Names = ["FourPeople", "Johnny", "KristenAndSara"]
elif opt.data == "MCL":
    test_set_dir = "/home/your_path/Datasets/Videos/test/MCL_all_gen/MCL_1080p"
    Names = ["videoSRC%02d"%a for a in range(1, 31)]
    # Names = ["videoSRC%02d"%a for a in range(28, 29)]
    
THREAD_NUM = len(Names) if len(Names)<=8 else 8 # 多线程线程数

class preprocess:
    def __init__(self, ID, Name):
        self.ID = ID
        self.Name = Name
        #判断目标位置所有文件夹可以启动断点继续功能

    def IP_test(self):
        cmd_IP = "python IP_test_multi.py --data %s --gpu %d --seq_name %s --test_set_dir %s --gop %d --frames %d \
                --idx_test %d --alpha_test %f"%(opt.data, self.ID, self.Name, \
                test_set_dir, opt.gop, opt.frames, opt.idx_test, opt.alpha_test)
        os.system(cmd_IP)

class myThread (threading.Thread):
    def __init__(self, threadID, name, movs_part):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.movs_part = movs_part

    def run(self):
        print ("######开始线程%s#######"%self.name)
        tic = time()
        video_proc = preprocess(self.threadID, self.movs_part)
        video_proc.IP_test()
        toc = time()
        print ("退出线程：%s, the time is %.2f min"%(self.name, (toc-tic)/60))


def main():
    # 创建新线程
    thread_pool = []
    for i in range(len(Names)):
        thread_pool.append(myThread(i%THREAD_NUM, "Thread-%d"%(i), Names[i]))
    # 开启线程，小于GPU数量和多余GPU数量两种情况
    if len(Names)==THREAD_NUM:
        for i in range(THREAD_NUM):
            thread_pool[i].start()
        for i in range(THREAD_NUM):
            thread_pool[i].join()
    else:
        loop_num = ceil(len(Names)/THREAD_NUM)
        for j in range(loop_num):
            for i in range(THREAD_NUM):
                if j*THREAD_NUM+i < len(Names):
                    thread_pool[j*THREAD_NUM+i].start()
            for i in range(THREAD_NUM):
                if j*THREAD_NUM+i < len(Names):
                    thread_pool[j*THREAD_NUM+i].join()

def cal_all():
    text_final = os.path.join(PConfig.info_dir, "test_%s.txt"%(opt.data))
    file = open(text_final, 'a+')
    file.write("\n#######################################################\n")
    file.write("Below is %d sequences test results (1I%dP-frames%d-idx%d-alpha%.2f)\n"%\
                (len(Names), opt.gop-1, opt.frames, opt.idx_test, opt.alpha_test))
    file.close()

    bpp_avg = 0.0
    bpp_res_avg = 0.0
    bpp_min_avg = 0.0
    psnr_avg = 0.0
    ssim_avg = 0.0
    
    for idx, Name in enumerate(Names):
        text_temp = os.path.join(PConfig.info_dir, opt.data, \
                            "test_GOP%d_%s.txt"%(opt.gop, Name))
        with open(text_temp,'r') as f:
            data = f.readlines()
            a2 = re.split(r'[a-zA-Z_, :\n]', data[-2])
            para_all = [float(elem) for elem in a2 if elem]
            print(para_all)
        file = open(text_final, 'a+')
        if idx == 0:
            file.write("%s\n"%data[-3])
            print("%s"%data[-3])
        file.write("%s"%data[-2])
        print("%s"%data[-2])
        if opt.data == "MCL":
            psnr_avg += para_all[1]
            ssim_avg += para_all[2]
            bpp_avg += para_all[3]
            bpp_res_avg += para_all[4]
            bpp_min_avg += para_all[5]
        else:
            psnr_avg += para_all[0]
            ssim_avg += para_all[1]
            bpp_avg += para_all[2]
            bpp_res_avg += para_all[3]
            bpp_min_avg += para_all[4]
        
    lambda_test = opt.alpha_test*PConfig.lambda_list[opt.idx_test] + (1-opt.alpha_test)*PConfig.lambda_list[opt.idx_test+1]
    print("lambda:%d, the final test result is %.4f, %.4f, %.4f, %.4f, %.4f (bpp, bpp_res, bpp_min, psnr, msssim)\n\n" % (
        lambda_test,
        bpp_avg / len(Names),
        bpp_res_avg / len(Names),
        bpp_min_avg / len(Names),
        psnr_avg / len(Names),
        ssim_avg / len(Names)))
    file = open(text_final, 'a+')
    file.write("\nthe final test result is %.4f, %.4f, %.4f, %.4f, %.4f (bpp, bpp_res, bpp_min, psnr, msssim)\n" % (
            bpp_avg / len(Names),
            bpp_res_avg / len(Names),
            bpp_min_avg / len(Names),
            psnr_avg / len(Names),
            ssim_avg / len(Names)))
    file.write("#######################################################\n\n")
    file.close()

    
if __name__ == "__main__":
    if opt.sw == 2:
        main()
        cal_all()
    elif opt.sw == 0:
        cal_all()
    elif opt.sw == 1:
        main()