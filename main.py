import os
from models.LPNetGA import LPNetGA
from tools import train, test, eval, metrics_visual

if __name__ == '__main__':
	im_size = (120,120)
	patch_size = (30,30)
	stride = (10,10)


	train_path = '../data/SIRST/train/'
	test_path = '../data/SIRST/test/'
	out_path = '../data/SIRST/test/predict'
	im_type = '.bmp'


	batch_size = 2
	epoches = 2


	model = LPNetGA(im_size, patch_size, stride)
	train(train_path, im_type, model, epoches, batch_size)
	test(test_path, out_path, im_type, model, load_path='./pretrained/LPNetGA.pth')
	eval('LPNetGA', test_path, './', 0.01, '.bmp')
	metrics_visual(['LPNetGA'], './')