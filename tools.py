import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import time
import cv2 as cv2
import random
import shutil

from utils.utils import *
from utils.eval import *

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
		torch.save(model.state_dict(), './pretrained/LPNetGA_epoch'+str(epoch+1)+'.pth')
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


def eval(name, test_path, out_path, stride, im_type, is_delete=True):
	timestramp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
	path = os.path.join(test_path, name+'_test_'+timestramp)
	if os.path.isdir(path):
		shutil.rmtree(path)
	os.makedirs(path)
	os.makedirs(os.path.join(path, 'feature_map'))
	os.makedirs(os.path.join(path, 'result'))
	
	model_path = os.path.join(test_path, name)
	feature_list = list_file(model_path, im_type)
	for fm_name in tqdm(feature_list):
		fm = cv2.imread(os.path.join(model_path, fm_name), 0)

		im_name = fm_name.split('.')[0]+im_type
		im = cv2.imread(os.path.join(test_path, im_name), 0)
		(h,w) = im.shape
		fm = cv2.resize(fm,(w,h))
		result = np.zeros((h, 2*w))
		result[:,:w] = im
		result[:,w:2*w] = fm

		cv2.imwrite(os.path.join(path, 'feature_map', im_name), fm)
		cv2.imwrite(os.path.join(path, 'result', im_name), result)

	ths = []
	Pds = []
	Fas = []
	TarPrecs = []
	TarRecs = []
	TarF1s = []
	PxlPrecs = []
	PxlRecs = []
	PxlF1s = []
	source_path = test_path
	gt_path = os.path.join(test_path, 'gt')
	feature_path = os.path.join(path, 'feature_map')
	evaluator = Evaluator(source_path, gt_path, feature_path, im_type, tar_area=[0, np.inf], is_print=True)
	for th in np.arange(0.0+stride, 1.0+stride, stride):
		print('th:', th, '=============================================')
		(Pd, Fa, TarPrec, TarRec, TarF1) = evaluator.target_metrics(th)
		(PxlPrec, PxlRec, PxlF1) = evaluator.pixel_metrics(th)
		ths.append(th)
		Pds.append(Pd)
		Fas.append(Fa)
		TarPrecs.append(TarPrec)
		TarRecs.append(TarRec)
		TarF1s.append(TarF1)
		PxlPrecs.append(PxlPrec)
		PxlRecs.append(PxlRec)
		PxlF1s.append(PxlF1)
	threshold_eavl = pd.DataFrame({'threshold':ths, 'Pd':Pds, 'Fa':Fas, 'TarPrec':TarPrecs, 'TarRec':TarRecs, 'TarF1':TarF1s, 'PxlPrec':PxlPrecs, 'PxlRec':PxlRecs, 'PxlF1':PxlF1s})
	threshold_eavl.to_csv(os.path.join(out_path, name+'_'+timestramp+'.csv'))

	if is_delete:
		shutil.rmtree(path)

def metrics_visual(name_list, file_path, fix={'Th':0.1,'Fa':0.2,'Order':2}):
	# palet = sns.color_palette('hls', len(name_list))

	linewidth = 1
	for name in name_list:
		csv_list = [file if name in file else '' for file in list_file(file_path,'.csv')]
		csv_list.sort()
		load_path = file_path+csv_list[-1]
		print('load_path:', load_path)
		df = pd.read_csv(load_path)
		TarPR = np.array([df['TarRec'].tolist(), df['TarPrec'].tolist()]).transpose([1,0]).tolist()
		TarPR.sort(key=lambda x:x[0])
		TarPR = np.array(TarPR).transpose([1,0])

		PxlPR = np.array([df['PxlRec'].tolist(), df['PxlPrec'].tolist()]).transpose([1,0]).tolist()
		PxlPR.sort(key=lambda x:x[0])
		PxlPR = np.array(PxlPR).transpose([1,0])

		ROC = np.array([df['Fa'].tolist(), df['Pd'].tolist()]).transpose([1,0]).tolist()
		# ROC.append([0,0])
		ROC.sort(key=lambda x:x[0])
		ROC = np.array(ROC).transpose([1,0])

		plt.figure(1)
		plt.plot(TarPR[0,:], TarPR[1,:], linewidth=linewidth)
		plt.xlabel('Target_Recall')
		plt.ylabel('Target_Precision')
		plt.legend(name_list)

		plt.figure(2)
		plt.plot(PxlPR[0,:], PxlPR[1,:], linewidth=linewidth)
		plt.xlabel('Pixel_Recall')
		plt.ylabel('Pixels_Precision')
		plt.legend(name_list)

		plt.figure(3)
		plt.plot(ROC[0,:], ROC[1,:], linewidth=linewidth)
		plt.xlabel('False alarm rate (Fa)')
		plt.ylabel('Probability of detection (Pd)')
		plt.xlim([0,2])
		plt.legend(name_list)

		plt.figure(4)
		plt.plot(df['threshold'], df['TarF1'], linewidth=linewidth)
		plt.xlabel('threshold')
		plt.ylabel('Target_F1')
		plt.legend(name_list)

		plt.figure(5)
		plt.plot(df['threshold'], df['PxlF1'], linewidth=linewidth)
		plt.xlabel('threshold')
		plt.ylabel('Pixel_F1')
		plt.legend(name_list)

	plt.figure(1)
	plt.plot([0.0,1.0], [0.0,1.0], '--', linewidth=linewidth)
	plt.figure(2)
	plt.plot([0.0,1.0], [0.0,1.0], '--', linewidth=linewidth)
	plt.show()
	# # plt.savefig(file_path+csv_list[-1].split('.')[0]+'.jpg')

	for name in name_list:
		csv_list = [file if name in file else '' for file in list_file(file_path,'.csv')]
		csv_list.sort()
		load_path = file_path+csv_list[-1]
		print('load_path:', load_path)
		df = pd.read_csv(load_path)

		ROC = np.array([df['Fa'].tolist(), df['Pd'].tolist()]).transpose([1,0]).tolist()
		ROC.append([0,0])
		ROC.sort(key=lambda x:x[0])
		ROC = np.array(ROC).transpose([1,0])
		select = np.where(ROC[0,:]<=0.5)[0]
		ROC = ROC[:,select]

		AUC = 0
		for i in range(ROC.shape[1]-1):
			fa0 = ROC[0,i]
			pd0 = ROC[1,i]

			fa1 = ROC[0,i+1]
			pd1 = ROC[1,i+1]

			AUC += pd1*(fa1-fa0)

		AUC /= max([max(ROC[1,:]),1.0])*max(ROC[0,:])


		TarF1 = 0
		PxlF1 = 0
		Fas = []
		fixedTh = fix['Th']
		fixedFa = fix['Fa']
		fixedOrder = fix['Order']
		for i in range(len(df)):
			data = df.iloc[i,:]
			if data['TarF1']>=TarF1:
				TarF1 = data['TarF1']
				TarPrec = data['TarPrec']
				TarRec = data['TarRec']
			if data['PxlF1']>=PxlF1:
				PxlF1 = data['PxlF1']
				PxlPrec = data['PxlPrec']
				PxlRec = data['PxlRec']
			if np.abs(data['Fa']-fixedFa) <= fixedTh:
				Fas.append((np.abs(data['Fa']-fixedFa), data['Fa'], data['Pd']))
		if len(Fas):
			Fas.sort()
			Fas = np.array(Fas)
			Fas = Fas[:fixedOrder,:]

			z = np.polyfit(Fas[:,1],Fas[:,2],fixedOrder)
			poly = np.poly1d(z)
			
			Pd = poly(fixedFa)
		else:
			Pd = np.nan

		print('Method:',name,'\tTarPrec.',TarPrec,'\tTarRec.',TarRec,'\tTarF1:',TarF1,'\tPxlPrec.',PxlPrec,'\tPxlRec.',PxlRec,'\tPxlF1:',PxlF1,'\tPd',Pd,'\tFa',fixedFa,'AUC:',AUC,'\n')
		# print('Method:',name,'AUC:',AUC,'\n')