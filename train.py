# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from data import *
from tqdm import tqdm
from preprocessing import *

train_set = RetinalDataset('train')
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=64,
                                          shuffle=True, num_workers=8)
val_set = RetinalDataset('val')
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=16,
                                          shuffle=False, num_workers=2)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Unet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64,out_ch, 1)

        self.fc = nn.Linear(1, 2)

    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)

        # import pdb; pdb.set_trace()

        out=self.fc(c10.unsqueeze(-1))
        # out = nn.Sigmoid()(c10)

        return out

net = Unet(1,1).cuda()

# optimizer = optim.Adam(net.parameters(), lr=0.01)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
weights_class = torch.Tensor(2).fill_(10)
weights_class[0] = 1
# criterion = torch.nn.CrossEntropyLoss(weight=weights_class).cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
# criterion = torch.nn.BCELoss()

# net.load_state_dict(torch.load('model.ckpt'))

predicted_patches = torch.zeros(len(val_set), 1, 48, 48)
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    net.train()
    for i, data in enumerate(tqdm(train_loader)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda().float()
        labels = labels.cuda().float()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        # loss = criterion(outputs, labels)
        loss = criterion(outputs.view(-1, 2), labels.long().view(-1))
        loss.backward()
        optimizer.step()

        outputs = outputs.max(-1)[1].cpu()
        acc = float((outputs==labels.cpu()).sum())/(outputs.shape[0]*outputs.shape[2]*outputs.shape[3])
        # import pdb; pdb.set_trace()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] acc: %.3f, loss: %.3f' %
                  (epoch + 1, i + 1, acc, running_loss/(i+1)))
            running_loss = 0.0

    acc = 0.
    net.eval()
    for i, data in enumerate(tqdm(val_loader)):
        inputs, labels = data
        inputs = inputs.cuda().float()
        labels = labels.cuda().float()

        with torch.no_grad():
            outputs = net(inputs)
            outputs = outputs.max(-1)[1].cpu()
            predicted_patches[i*16:(i+1)*16, :, :, :] = outputs

        if i == 9:
            predicted_patches[i*16:, :, :, :] = outputs
            
            predicted_patches = predicted_patches.numpy()
            pred_imgs = recompone(predicted_patches, 13, 12) 
            pred_imgs = pred_imgs[:,:,0:584,0:565].squeeze(0)
            pred_imgs = pred_imgs*255
            img = Image.fromarray(np.uint8(pred_imgs).squeeze(0), 'L')
            img.save('pred_.png')

            gt_img = np.array(Image.open('data/gt/24_manual1-0000.jpg').convert('1'))*255
            acc = (pred_imgs==gt_img).sum()/(gt_img.shape[0]*gt_img.shape[1])

            print('[%d] acc: %.3f' %
                    (epoch + 1, acc))
            # import pdb; pdb.set_trace()

torch.save(net.state_dict(), 'model_.ckpt')
print('Finished Training')


