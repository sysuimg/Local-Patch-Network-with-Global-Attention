import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import *
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import os
import cv2 as cv2
import random
import shutil

def data_generator(path, im_type, im_size, subim_ksize, subim_stride, batch_size:int = 128, mode: str= "train", shuffle: bool= True):
	file_list = list_file(path+"/", im_type)
	x_path = path+"/"
	y_path = path+"/gt/"

	candidates = [i for i in range(len(file_list))]
	if shuffle:
		random.shuffle(candidates)
	length = len(candidates)
	yield int(length / batch_size)
	for i in range(0, length, batch_size):
		from_index = i
		to_index = from_index + batch_size
		if to_index > length:
			to_index = length
		x_train = []
		y_density = []
		y_subimgs = []
		# TODO 根据index读取数据
		for index in candidates[from_index:to_index]:
			x_img = cv2.imread(x_path+file_list[index],0)
			if im_size != None:
				x_img = cv2.resize(x_img, im_size)
			x_img = x_img / 255
			x_train.append(x_img)
			y_img = cv2.imread(y_path+file_list[index].split('.')[0]+'_gt'+im_type,0)
			if im_size != None:
				y_img = cv2.resize(y_img, im_size)
			y_img = convert_gt(y_img)


			h,w = y_img.shape
			gaussianKernel = GaussianKernel(h,w,20) # origin: 5 
			LPF = FilterFrequency(y_img,gaussianKernel,False)
			LPF = scaleNonlinear(LPF, 1)
			LPF = LPF/np.sum(LPF)
			y_density.append(LPF)

			subimgs = []
			for si in range(0,im_size[0]-subim_ksize[0]+1,subim_stride[0]):
				for sj in range(0,im_size[1]-subim_ksize[1]+1,subim_stride[1]):
					subimg = y_img[si:si+subim_ksize[0],sj:sj+subim_ksize[1]]
					subimgs.append(subimg)
			subimgs = np.array(subimgs).reshape(-1, subim_ksize[0]*subim_ksize[1])
			y_subimgs.append(subimgs)


		x_train = np.array(x_train).reshape(-1, 1, im_size[0], im_size[1])
		y_density = np.array(y_density).reshape(-1, 1, im_size[0]*im_size[1])
		y_subimgs = np.array(y_subimgs).reshape(x_train.shape[0], -1, subim_ksize[0]*subim_ksize[1])

		yield x_train, y_density, y_subimgs


def train(data_path, im_type, model, epoches, batch_size, load_path=None):
	model = model.cuda()
	if load_path!=None:
		print('load:', load_path)
		model.load_state_dict(torch.load(load_path))
	
	total_params = sum(p.numel() for p in model.parameters())
	print(f'{total_params:,} total parameters.')
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f'{trainable_params:,} training parameters.')

	optimizer = optim.Adam(model.parameters())
	
	MSE = nn.MSELoss()
	BCE = nn.BCELoss()

	for epoch in range(epoches):
		generator = data_generator(data_path, im_type, im_size=model.im_size, subim_ksize=model.ksize, subim_stride=model.stride, batch_size=batch_size)   
		total = next(generator)
		with tqdm(generator, total=total) as t:
			index = 0
			epoch_attn_loss = 0
			epoch_lfe_loss = 0
			for x_train, y_density, y_subimgs in t:
				torch.cuda.empty_cache() # release cache

				model.train()
				input_tensor = torch.tensor(x_train).cuda().float()

				# train
				optimizer.zero_grad()
				(_, patches_tensor, attn_tensor) = model(input_tensor)
				assert len(patches_tensor) == y_subimgs.shape[1]

				density_tensor = torch.tensor(y_density).cuda().float()

				attn_loss = (120*120)*MSE(attn_tensor, density_tensor)

				lfe_loss = 0
				for i, patch_tensor in enumerate(patches_tensor):
					target_tensor = torch.tensor(y_subimgs[:,i,:]).cuda().float()
					lfe_loss += BCE(patch_tensor, target_tensor)/len(patches_tensor)

				loss = attn_loss + lfe_loss
				loss.backward()
				optimizer.step()
				epoch_attn_loss += attn_loss.detach().cpu()
				epoch_lfe_loss += lfe_loss.detach().cpu()

				# loss display
				if index % 1 == 0:
					t.set_postfix_str(
						"epoch-batch:[{}: {}], attn:{:.8f}, lfe:{:.8f}".format(epoch, index, epoch_attn_loss / (index + 1), epoch_lfe_loss / (index + 1)))
				index += 1
		if not os.path.isdir('./pretrained/'):
			os.makedirs('./pretrained/')
		torch.save(model.state_dict(), './pretrained/LPNetGA.pth')
	return model


def test(data_path, out_path, im_type, model, load_path=None):
	model = model.cuda()
	if load_path!=None:
		print('load:', load_path)
		model.load_state_dict(torch.load(load_path))
	if not os.path.isdir(out_path):
		os.makedirs(out_path)

	img_list = list_file(data_path, im_type)
	for img_name in tqdm(img_list):
		if img_name==None:
			continue
		img = cv2.imread(os.path.join(data_path, img_name), 0)
		h,w = img.shape
		resize_img = cv2.resize(img, model.im_size) / 255
		input_tensor = torch.tensor(resize_img.reshape(-1, 1, model.im_size[0], model.im_size[1])).cuda().float()

		fusedFM, _, _ = model(input_tensor)

		# restore fusedFM
		fusedFM = fusedFM.detach().cpu().numpy().reshape(model.im_size)
		fh, fw = fusedFM.shape
		fusedFM = np.expand_dims(fusedFM, axis=0)
		fusedFM = np.expand_dims(fusedFM, axis=0)
		fusedFM = bilinear_interpolate(fusedFM, (h/fh, w/fw), 0)
		fusedFM = np.reshape(fusedFM, (h,w))
		# fusedFM = scaleNonlinear(fusedFM, 0.1)

		feature_map = fusedFM
		feature_map = np.round(feature_map*255).astype(np.uint8)

		cv2.imwrite(os.path.join(out_path, img_name), feature_map)