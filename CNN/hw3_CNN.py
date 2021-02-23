import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time
import pandas as pd

from torch.utils.data import DataLoader, Dataset

torch.cuda.set_device(2)
print(torch.cuda.is_available())
print(torch.cuda.device_count()) # 返回GPU数目
print(torch.cuda.get_device_name(0))
print(torch.cuda.current_device())

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
		super(Classifier, self).__init__() #
		self.cnn = nn.Sequential(
			# 第一个参数为输入数据的通道数，rgb图像的通道数为1
			# 第二个为输出数据的通道数
			# 第三个为卷积核的大小
			# 第四个为步长
			# 第五个为padding, 用来补齐边缘
			nn.Conv2d(3, 64, 3, 1, 1), #  [64, 128, 128]

			nn.BatchNorm2d(64),  # Normalize
			nn.ReLU(	),  # activation function

			# Pooling Function
			# 第一个是kernel_size, 窗口大小
			# 第二个是stride, 移动步长
			# 第三个参数输入的是每一条边补充-的层数
			nn.MaxPool2d(2,2,0),

			# 第二层卷积层和池化层
			nn.Conv2d(64,128, 3, 1, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(2,2,0),

			# 第三层卷积层和池化层
			nn.Conv2d(128, 256, 3, 1, 1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.MaxPool2d(2, 2, 0),

			# 第四层卷积层和池化层
			nn.Conv2d(256, 512, 3, 1, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.MaxPool2d(2, 2, 0),

			# 第五层卷积层和池化层
			nn.Conv2d(512, 512, 3, 1, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.MaxPool2d(2, 2, 0), #  [512, 4, 4]
		)
		self.fc = nn.Sequential(
			nn.Linear(512 * 4 * 4, 1024),
			nn.ReLU(),
			nn.Linear(1024,512),
			nn.ReLU(),
			nn.Linear(512,11)
		)

	def forward(self,x):
		out = self.cnn(x)
		out = out.view(out.size()[0], -1)
		return self.fc(out)

"""
Training
"""
model = Classifier().cuda()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epoch = 60

for epoch in range(num_epoch):
	epoch_start_time = time.time()
	train_acc = 0.0
	train_loss = 0.0
	val_acc = 0.0
	val_loss = 0.0
	model.train()
	for i, data in enumerate(train_loader):
		optimizer.zero_grad()
		train_pred = model(data[0].cuda())
		batch_loss = loss(train_pred, data[1].cuda())
		batch_loss.backward()
		optimizer.step()

		train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
		train_loss += batch_loss.item()
	model.eval()
	with torch.no_grad():
		for i, data in enumerate(val_loader):
			val_pred = model(data[0].cuda())
			batch_loss = loss(val_pred, data[1].cuda())
			val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
			val_loss += batch_loss.item()
		print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
			  (epoch + 1, num_epoch, time.time() - epoch_start_time, \
			   train_acc / train_set.__len__(), train_loss / train_set.__len__(), val_acc / val_set.__len__(),
			   val_loss / val_set.__len__()))

test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model.eval()
prediction = []
with torch.no_grad():
	for i, data in enumerate(test_loader):
		test_pred = model(data.cuda())
		test_label = np.argmax(test_pred.cpu().data.numpy(), axis = 1)
		for y in test_label:
			prediction.append(y)

with open("predict.csv", 'w') as f :
	f.write('Id, Category\n')
	for i, y in enumerate(prediction):
		f.write('{}, {}\n'.format(i,y))