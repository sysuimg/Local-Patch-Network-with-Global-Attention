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
	epoches = 40
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
	# train('MFIRST', train_path, im_type, model, epoches, batch_size,
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
	epoches = 600
	data_aug = False
	load_path = None
	eval_config = {'eval_path':test_path,
				   'tmp_path':tmp_path,
				   'eval_start_epoch':300,
				   'eval_per_epoch':10}
	attention_supervision = False  # the attention layer can be self-attented in few-shot dataset
	gaussian_std = 5
	device_ids = [0,1]

	model = LPNetGA_v2(im_size, patch_size, stride)
	# train('SIRST', train_path, im_type, model, epoches, batch_size,
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
