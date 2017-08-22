# coding: utf-8

import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from PIL import Image


# dataset path
data_dir = '~/_datasets/_zy_data'
data_dir = os.path.expanduser(data_dir)

# dataset pre-process
train_transform= transforms.Compose([
    transforms.Scale([512, 512]),
    transforms.ToTensor(),
])
    
# ImageFolder: https://github.com/pytorch/vision#imagefolder
tset = ImageFolder(data_dir, transform=train_transform)
train_dataloader = DataLoader(tset, batch_size=4, shuffle=True)
num_classes = len(set(tset.classes))

# use pretrained resnet18 network: image input: (512, 512)
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(2048, num_classes)

# run in cuda
if torch.cuda.is_available():
    model = model.cuda()
print(model)

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# loss function
loss_fn = nn.CrossEntropyLoss()

# begin to train models
model.train()
for inputs, labels in train_dataloader:
    if torch.cuda.is_available():
        # use cuda
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
    else:
        # use cpu
        inputs = Variable(inputs)
        labels = Variable(labels)
        
    # zero the parameter gradients
    optimizer.zero_grad()
    
    # forward
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

# begin to test
image_file = '/home/zouying/duoduo3.jpg'

# if want to show image file, run the following
# img_show_1 = plt.imread(image_file)
# plt.imshow(img_show_1)

img2 = Image.open(image_file)
img2 = train_transform(img2)
img2 = img2.unsqueeze(0)  # pytorch only accept batch images, so unsqueeze it
                          # if you meet error shows need 4D, but input is 3D
                          # try it

img2 = Variable(img2)
if torch.cuda.is_available():
    img2 = img2.cuda()

model.eval()
target = model(img2)

_, pred = torch.max(target.data, 1)
print('all classes: ', tset.classes)
print('all classes prop: ', nn.functional.softmax(target))
print('prediction: ', pred[0], ' classes: ', tset.classes[pred[0]])
