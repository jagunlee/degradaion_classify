import torch
import cv2
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import numpy as np
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler as SRS
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import numpy as np
import copy
from torchvision.models.vgg import vgg16
from torchvision.models.resnet import resnet50
from glob import glob
import pdb

class CustomDataset(torch.utils.data.Dataset): 
    def __init__(self, num):
        #num: 0-train, 1-val, 2-test
        '''
        deg = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_noise', 'glass_blur']
        for i in range(calss_num):
            data0 = glob("./Tiny-ImageNet-C/"+deg[i]+"/{}/*/*.JPEG".format(num))
            y = []
            datas = []
            for data in data0:
                y.append([i])
                datas.append(cv2.imread(data).transpose(2,0,1))
        '''
        data0 = glob("./Tiny-ImageNet-C/defocus_blur/*/*/*.JPEG")
        data1 = glob("./Tiny-ImageNet-C/fog/*/*/*.JPEG")
        data2 = glob("./Tiny-ImageNet-C/motion_blur/*/*/*.JPEG")
        data3 = glob("./Tiny-ImageNet-C/brightness/*/*/*.JPEG")
        data4 = glob("./Tiny-ImageNet-C/contrast/*/*/*.JPEG")
        data5 = glob("./Tiny-ImageNet-C/elastic_transform/*/*/*.JPEG")
        data6 = glob("./Tiny-ImageNet-C/frost/*/*/*.JPEG")
        data7 = glob("./Tiny-ImageNet-C/gaussian_noise/*/*/*.JPEG")
        data8 = glob("./Tiny-ImageNet-C/pixelate/*/*/*.JPEG")
        data9 = glob("./Tiny-ImageNet-C/zoom_blur/*/*/*.JPEG")
        if num==0:
            start = 0
            end = int(len(data0)*0.7)
        if num==1:
            start = int(len(data0)*0.7)
            end = int(len(data0)*0.9)
        if num==2:
            start = int(len(data0)*0.9)
            end = len(data0)
        y = []
        datas = []
        for data in data0[start:end]:
            #y.append([1,0,0])
            y.append([0])
            datas.append(cv2.imread(data).transpose(2,0,1))
        for data in data1[start:end]:
            #y.append([0,1,0])
            y.append([1])
            datas.append(cv2.imread(data).transpose(2,0,1))
        for data in data2[start:end]:
            #y.append([0,0,1])
            y.append([2])
            datas.append(cv2.imread(data).transpose(2,0,1))
        for data in data3[start:end]:
            #y.append([0,0,1])
            y.append([3])
            datas.append(cv2.imread(data).transpose(2,0,1))
        for data in data4[start:end]:
            #y.append([0,0,1])
            y.append([4])
            datas.append(cv2.imread(data).transpose(2,0,1))
        '''
        for data in data5[start:end]:
            #y.append([0,0,1])
            y.append([5])
            datas.append(cv2.imread(data).transpose(2,0,1))
        for data in data6[start:end]:
            #y.append([0,0,1])
            y.append([6])
            datas.append(cv2.imread(data).transpose(2,0,1))
        for data in data7[start:end]:
            #y.append([0,0,1])
            y.append([7])
            datas.append(cv2.imread(data).transpose(2,0,1))
        for data in data8[start:end]:
            #y.append([0,0,1])
            y.append([8])
            datas.append(cv2.imread(data).transpose(2,0,1))
        for data in data9[start:end]:
            #y.append([0,0,1])
            y.append([9])
            datas.append(cv2.imread(data).transpose(2,0,1))
        '''
        #self.y = np.array(y)
        #self.data = np.array(datas)
        self.y = y
        self.data = datas

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.data[idx])
        y = torch.FloatTensor(self.y[idx])
        return x,y

def train(model,lr, epoch, train_dt, device, val_dt=None, val=False):
    model.train()
    n=0
    tr_loss = 0
    trl = []
    val_loss = 0
    vll = []
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    st = time.time()
    for i in range(epoch):
        for img, label in train_dt:
            img, label = Variable(img), Variable(label)
            img, label = img.to(device), label.to(device,dtype=torch.long)
            optimizer.zero_grad()
            out = model.forward(img)
            loss = criterion(out, label.flatten())
            loss.backward()
            tr_loss+=loss
            optimizer.step()
            n+=1
        if val==True:
            with torch.no_grad():
                for vimg, vlabel in val_dt:
                    vimg, vlabel = Variable(vimg), Variable(vlabel)
                    vimg, vlabel = vimg.to(device), vlabel.to(device, dtype=torch.long)
                    vout = model.forward(vimg)
                    vloss = criterion(vout, vlabel.flatten())
                    val_loss +=vloss
                print("{}  epoch: {}/{}  train_loss: {}  val_loss: {}  time: {}".format(n, i+1, epoch, tr_loss/len(train_dt), val_loss/len(val_dt), time.time()-st))
                trl.append(tr_loss/len(train_dt))
                vll.append(val_loss/len(val_dt))
                tr_loss = 0
                val_loss = 0
        else:
            print("{}  epoch: {}/{}  train_loss: {}  time: {}".format(n, i+1, epoch, tr_loss/len(train_dt), time.time()-st))
            trl.append(tr_loss/len(train_dt))
            tr_loss = 0
    plt.plot(trl,label='train_loss')
    if val==True:
        plt.plot(vll,label='valid_loss')
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig("train_{}.png".format(epoch))
    return trl

def test(model, test_dt, device, datanum):
    model.eval()
    correct = 0
    st = time.time()
    for img, label in test_dt:
        img, label = Variable(img), Variable(label)
        img, label = img.to(device), label.to(device, dtype=torch.long)
        label = label.flatten()
        out = model.forward(img)
        correct+=out.data.max(1)[1].eq(label.data).sum()
    print("{}/{} correct, total {}% accuracy    used time: {}".format(correct, datanum, correct*100./datanum, time.time()-st))

tdataset = CustomDataset(0)
vdataset = CustomDataset(1)
dataloader = DataLoader(tdataset, batch_size=64, shuffle=True)
vdataloader = DataLoader(vdataset, batch_size=64, shuffle=True)
testdataset = CustomDataset(2)
testdataloader = DataLoader(testdataset, batch_size=64, shuffle=True)
print("Data Loaded, train data {}, val data {}, test data {}".format(len(tdataset), len(vdataset), len(testdataset)))

model = vgg16(pretrained=False)
model.classifier[6] = nn.Linear(4096, 5)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model.cuda();
train(model, 0.001, 20, dataloader, device, val_dt = vdataloader, val=True)
torch.save(model.state_dict(), "./model.pt")

test(model, testdataloader, device, len(testdataset))
