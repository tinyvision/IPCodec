data=HB # "UVG", "HB", "HC", "HD", "HE", "MCL"
gop=108 # UVG=12/108, HEVC=10/108
frames=108 # UVG=9999, HEVC=100
# python 00-IP-multi-test.py --sw 2 --data ${data} --gop ${gop} --frames ${frames} --idx_test 0 --alpha_test 1 
# python 00-IP-multi-test.py --sw 2 --data ${data} --gop ${gop} --frames ${frames} --idx_test 0 --alpha_test 0.5
# python 00-IP-multi-test.py --sw 2 --data ${data} --gop ${gop} --frames ${frames} --idx_test 1 --alpha_test 1
# python 00-IP-multi-test.py --sw 2 --data ${data} --gop ${gop} --frames ${frames} --idx_test 2 --alpha_test 1
python 00-IP-multi-test.py --sw 2 --data ${data} --gop ${gop} --frames ${frames} --idx_test 3 --alpha_test 1
# python 00-IP-multi-test.py --sw 2 --data ${data} --gop ${gop} --frames ${frames} --idx_test 4 --alpha_test 1
# python 00-IP-multi-test.py --sw 2 --data ${data} --gop ${gop} --frames ${frames} --idx_test 5 --alpha_test 1
# python 00-IP-multi-test.py --sw 2 --data ${data} --gop ${gop} --frames ${frames} --idx_test 7 --alpha_test 1
# python 00-IP-multi-test.py --sw 2 --data ${data} --gop ${gop} --frames ${frames} --idx_test 8 --alpha_test 0
