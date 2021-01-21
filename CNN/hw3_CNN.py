import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd

from torch.utils.data import DataLoader, Dataset

def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
	    # enumerate(list) 返回index， item
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(128, 128))
        if label:
          y[i] = int(file.split("_")[0])
    if label:
      return x, y
    else:
      return x




workspace_dir = './food-11'
print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
print("Size of validation data = {}".format(len(val_x)))
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
print("Size of Testing data = {}".format(len(test_x)))

train_transform = transforms.Compose([
	transforms.ToPILImage(),
	transforms.RandomHorizontalFlip(),
	transforms.RandomRotation(15),
	transforms.ToTensor(),
])

test_transform = transforms.Compose([
	transforms.ToPILImage(),
	transforms.ToTensor()
])

class ImgDataset(Dataset):
	def __init__(self, x, y =None, transform = None):
		self.x = x
		# label is required to be a longTensor
		self.y = y
		if y is not None:
			self.y = torch.LongTensor(y)
		self.transform = transform

	def __len__(self):
		return len(self.x)

	def __getitem__(self, index):
		X = self.x[index]
		if self.transform is not None:
			X = self.transform(X)
		if self.y is not None:
			Y = self.y[index]
			return X,Y
		else:
			return X

batch_size = 128
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()
		self.cnn = nn.Sequential(
			nn.Conv2d(3)
		)