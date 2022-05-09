from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

class Deconv(nn.Module):
	def __init__(self, n_in, n_out):
		super(Deconv, self).__init__()
		self.n_in = n_in
		self.n_out = n_out
		self.deconv = nn.ConvTranspose2d(in_channels=n_in, out_channels=n_out, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn = nn.BatchNorm2d(num_features=n_out)
		self.relu = nn.ReLU(inplace=True)

	def forward(self,x):
		return self.relu(self.bn(self.deconv(x)))

class ResBlock(nn.Module):
	def __init__(self, n_in, n_out):
		super(ResBlock, self).__init__()
		self.n_in = n_in
		self.n_out = n_out

		self.conv1 = nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0)
		self.bn1 = nn.BatchNorm2d(num_features=n_out)
		self.relu1 = nn.ReLU(inplace=True)

		self.conv2 = nn.Conv2d(in_channels=n_out, out_channels=n_out, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(num_features=n_out)
		self.relu2 = nn.ReLU(inplace=True)

		self.conv3 = nn.Conv2d(in_channels=n_out, out_channels=n_out, kernel_size=1, stride=1, padding=0)
		self.bn3 = nn.BatchNorm2d(num_features=n_out)

		if n_in == n_out:
			pass
		else:
			self.identity = nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=3, stride=1, padding=1)
			self.bn = nn.BatchNorm2d(num_features=n_out)

		self.relu = nn.ReLU(inplace=True)

	def forward(self,x):
		res = self.relu1(self.bn1(self.conv1(x)))
		res = self.relu2(self.bn2(self.conv2(res)))
		res = self.bn3(self.conv3(res))
		if self.n_in == self.n_out:
			y = res + x
		else:
			y = res + self.bn(self.identity(x))
		y = self.relu(y)
		return y


class GFE(nn.Module):
	def __init__(self,n_out,chn):
		super(GFE,self).__init__()
		self.block1 = ResBlock(1, chn*2)
		self.block2 = ResBlock(chn*2, chn*4)
		self.block3 = ResBlock(chn*4, chn*2)
		self.block4 = ResBlock(chn*2, n_out)

	
	def forward(self,x):
		x1 = self.block1(x) # [B M N chn*2]
		x2 = self.block2(x1) # [B M N chn*4]
		x3 = self.block3(x2) # [B M N chn*2]
		x4 = self.block4(x3) # [B M N 1]

		gfm = x4

		return gfm


class Attn(nn.Module):
	def __init__(self,n_in,chn):
		super(Attn,self).__init__()
		self.block1 = ResBlock(n_in, chn*2)
		self.block2 = ResBlock(chn*2, chn*4)
		self.block3 = ResBlock(chn*4, chn*8)

		self.block4 = ResBlock(chn*8, chn*4)
		self.block5 = ResBlock(chn*4, chn*2)
		self.block6 = ResBlock(chn*2, 1)

		self.cnn = nn.Conv2d(1, 1, kernel_size=1)
		self.softmax = nn.Softmax(dim=2)

	def forward(self, x):
		attn1 = self.block1(x) # [B M N chn*2]
		attn2 = self.block2(attn1) # [B M N chn*4]
		attn3 = self.block3(attn2) # [B M N chn*8]

		attn4 = self.block4(attn3) # [B M N chn*4]
		attn5 = self.block5(attn4) # [B M N chn*2]
		attn6 = self.block6(attn5) # [B M N 1]

		attn = self.cnn(attn6)
		(b,c,h,w) = attn.shape
		attn = torch.reshape(attn, (-1, 1, h*w))
		attn = self.softmax(attn)

		return attn


class kconv(nn.Module):
	def __init__(self,n_in,n_out,ksize,padding):
		super(kconv,self).__init__()
		self.conv = nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=ksize, stride=1, padding=padding)
		self.bn = nn.BatchNorm2d(num_features=n_out)
		self.relu = nn.ReLU(inplace=True)
	def forward(self,x):
		k = self.relu(self.bn(self.conv(x)))
		return k


class patchNet(nn.Module):
	def __init__(self,chn,k_num):
		super(patchNet,self).__init__()
		self.deconv = Deconv(chn*k_num, chn*8)
		self.block1 = ResBlock(chn*8, chn*8)
		self.block2 = ResBlock(chn*8, chn*8)
		self.block3 = ResBlock(chn*8, 1)
		self.pool = nn.MaxPool2d(2, stride=2)

		self.classifier = nn.Conv2d(1, 1, kernel_size=1)

	def forward(self,ks):
		patch1 = self.deconv(ks) # [B 2M 2N chn*8]
		patch2 = self.block1(patch1) # [B 2M 2N chn*8]
		patch3 = self.block2(patch2) # [B 2M 2N chn*8]
		patch4 = self.block3(patch3) # [B 2M 2N 1]

		patch = self.pool(patch4) # [B M N 1]
		patch = self.classifier(patch) # [B M N 1]
		return patch


class LPNet(nn.Module):
	def __init__(self,n_in,chn):
		super(LPNet,self).__init__()
		self.conv_k1 = kconv(n_in, chn, 1, 0)
		self.conv_k3 = kconv(n_in, chn, 3, 1)
		self.conv_k5 = kconv(n_in, chn, 5, 2)
		self.patch_net = patchNet(chn, 3)

	def forward(self,x):
		k1 = self.conv_k1(x)
		k3 = self.conv_k3(x)
		k5 = self.conv_k5(x)
		ks = torch.cat([k1, k3, k5], dim=1)
		patch = self.patch_net(ks)
		(b,c,h,w) = patch.shape
		patch = torch.reshape(patch, (b,c,h*w))
		patch = torch.sigmoid(patch)
		patch = torch.reshape(patch, (b,c,h,w))
		return patch


class LPNetGA(nn.Module):
	def __init__(self, im_size:tuple, ksize:tuple=(30,30), stride:tuple=(15,15)):
		super(LPNetGA,self).__init__()
		self.im_size = im_size
		self.ksize = ksize
		self.stride = stride
		
		gfe_chn = 32 # 32
		attn_chn = 16  # 16
		lfe_chn = 4  # 4
		self.feature_num = 8 # 8
		self.gfe = GFE(self.feature_num, gfe_chn)
		self.attn = Attn(self.feature_num, attn_chn)
		self.lfe = LPNet(self.feature_num, lfe_chn)

	def forward(self, x):
		gfm = self.gfe(x)
		attn = self.attn(gfm)

		def valid(L,k,s):
			assert (L-k)%s == 0

		b,c,h,w = gfm.shape
		valid(h, self.ksize[0], self.stride[0])
		valid(w, self.ksize[1], self.stride[1])

		(ph,pw) = (h+2*self.ksize[0],w+2*self.ksize[1])

		padding_fm = torch.zeros((b,c,ph,pw)).cuda()
		attn_repeat = attn.repeat(1,self.feature_num,1)
		padding_fm[:,:,self.ksize[0]:h+self.ksize[0],self.ksize[1]:w+self.ksize[1]] = gfm * torch.reshape(attn_repeat, gfm.shape)

		fused_fm = torch.zeros((b,1,ph,pw)).cuda()
		fm_fuseState = np.zeros((b,1,ph,pw))

		def subFuse(addIm, backIm, fuse_state):
			addIm = addIm.detach().cpu().numpy()
			backIm = backIm.detach().cpu().numpy()

			unvisited = np.where(fuse_state==0)
			visited = np.where(fuse_state!=0)

			backIm[unvisited] = addIm[unvisited]
			backIm[visited] = backIm[visited]+addIm[visited]
			fuse_state += 1

			backIm = torch.tensor(backIm).cuda()
			return backIm, fuse_state

		subimgs = []
		for i in range(0,h+self.ksize[0]+1,self.stride[0]):
			for j in range(0,w+self.ksize[1]+1,self.stride[1]):
				sub = padding_fm[:,:,i:i+self.ksize[0],j:j+self.ksize[1]]
				sub = self.lfe(sub)
				sub = torch.reshape(sub, (-1, self.ksize[0]*self.ksize[1]))
				if i>=self.ksize[0] and i<=h and j>=self.ksize[1] and j<=w:
					subimgs.append(sub)
				addIm = torch.reshape(sub, (-1, 1, self.ksize[0], self.ksize[1]))
				backIm = fused_fm[:,:,i:i+self.ksize[0],j:j+self.ksize[1]]

				fuse_state = fm_fuseState[:,:,i:i+self.ksize[0],j:j+self.ksize[1]]
				outIm, fuse_state = subFuse(addIm, backIm, fuse_state)
				fused_fm[:,:,i:i+self.ksize[0],j:j+self.ksize[1]] = outIm
				fm_fuseState[:,:,i:i+self.ksize[0],j:j+self.ksize[1]] = fuse_state


		# fused_fm /= torch.tensor(fm_fuseState).cuda().float()
		# print(torch.max(fused_fm), torch.min(fused_fm))
		
		INF = 1e6
		fused_fm = fused_fm[:,:,self.ksize[0]:h+self.ksize[0],self.ksize[1]:w+self.ksize[1]]
		fused_fm = torch.clamp(fused_fm, 0.0, INF)

		fusedFM = torch.reshape(fused_fm, (-1, 1, self.im_size[0]*self.im_size[1]))
		attnFM = torch.reshape(attn, (-1, 1, self.im_size[0]*self.im_size[1]))

		fusedFM = ((fusedFM - torch.min(fusedFM)) / (torch.max(fusedFM) - torch.min(fusedFM) + 1e-4)) ** 10

		return (fusedFM, subimgs, attnFM)