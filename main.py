'''
Edited by Fang Chen [cfun.cqupt@outlook.com].
Please cite our paper as follows if you use this code:
@ARTICLE{9735292,
  author={Chen, Fang and Gao, Chenqiang and Liu, Fangcen and Zhao, Yue and Zhou, Yuxi and Meng, Deyu and Zuo, Wangmeng},
  journal={IEEE Transactions on Aerospace and Electronic Systems}, 
  title={Local Patch Network with Global Attention for Infrared Small Target Detection}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TAES.2022.3159308}}
'''
import os
from models.LPNetGA import LPNetGA
from models.LPNetGA_v2 import LPNetGA as LPNetGA_v2
from tools import train, test, eval, metrics_visual

if __name__ == '__main__':
	im_size = (120,120)
	patch_size = (30,30)
	stride = (10,10)

	# # ========================================== MFIRST ===========================================
	train_path = '../data/MFIRST/train/'
	test_path = '../data/MFIRST/test/'
	out_path = '../data/MFIRST/test/LPNetGA/'
	tmp_path = '../data/MFIRST/test/tmp_eval/'
	im_type = '.bmp'
	
	batch_size = 20
	epochs = 40
	data_aug = False
	load_path = None
	eval_config = {'eval_path':test_path,
				   'tmp_path':tmp_path,
				   'eval_start_epoch':10, # 10
				   'eval_per_epoch':1}
	attention_supervision = True  # the attention layer should be supervised in complicated dataset
	gaussian_std = 20 # 5 std=1/σ
	device_ids = [0,1]
	
	model = LPNetGA_v2(im_size, patch_size, stride)
	# train('MFIRST', train_path, im_type, model, epochs, batch_size,
	# 	  attention_supervision=attention_supervision,
	# 	  gaussian_std=gaussian_std,
	# 	  data_aug=data_aug,
	# 	  eval_config=eval_config,
	# 	  load_path=load_path,
	# 	  device_ids=device_ids)
	
	
	# # # testing
	test(test_path, out_path, im_type, model, load_path='./pretrained/LPNetGA-MFIRST.pth')
	eval('LPNetGA', test_path, './', 0.01, '.bmp')
	metrics_visual(['LPNetGA'], './')


	# ========================================== SIRST =============================================
	train_path = '../data/SIRST/train/'
	test_path = '../data/SIRST/test/'
	out_path = '../data/SIRST/test/LPNetGA/'
	tmp_path = '../data/SIRST/test/tmp_eval/'
	im_type = '.bmp'

	batch_size = 8
	epochs = 600
	data_aug = False
	load_path = None
	eval_config = {'eval_path':test_path,
				   'tmp_path':tmp_path,
				   'eval_start_epoch':300,
				   'eval_per_epoch':10}
	attention_supervision = False  # the attention layer can be self-supervised in few-shot dataset
	gaussian_std = 5
	device_ids = [0,1]

	model = LPNetGA_v2(im_size, patch_size, stride)
	# train('SIRST', train_path, im_type, model, epochs, batch_size,
	# 	  attention_supervision=attention_supervision,
	# 	  gaussian_std=gaussian_std,
	# 	  data_aug=data_aug,
	# 	  eval_config=eval_config,
	# 	  load_path=load_path,
	# 	  device_ids=device_ids)

	# # # testing
	test(test_path, out_path, im_type, model, load_path='./pretrained/LPNetGA-SIRST.pth')
	eval('LPNetGA', test_path, './', 0.01, '.bmp')
	metrics_visual(['LPNetGA'], './')
