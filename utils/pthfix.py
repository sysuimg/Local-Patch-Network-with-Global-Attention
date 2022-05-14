import torch

if __name__ == '__main__':
	load_path1 = '../pretrained/V2HMA_MFIRST.pth'
	load_path2 = '../pretrained/LPNetGA_epoch15.pth'

	pth1 = torch.load(load_path1)
	pth2 = torch.load(load_path2)
	
	for (key1, key2) in zip(pth1.keys(), pth2.keys()):
		print(key1==key2, key1, key2)
		print(pth1[key1].shape, pth2[key2].shape)
		pth2[key2] = pth1[key1]
	torch.save(pth2, '../pretrained/LPNetGA.pth')