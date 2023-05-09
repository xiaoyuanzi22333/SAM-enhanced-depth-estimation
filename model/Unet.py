import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import numpy as np


class pix2pix(nn.Module):
    
    def __init__(self, in_ch=3, out_ch=1, dim=64):
        
        # the input size will be stricted at 256*256
        # the in_ch is the input image channel size
        # the out_ch is the output image channel size
        # the dim is the processing dimension
        
        super(pix2pix, self).__init__()
        
        #   480*640
        self.en1 = nn.Sequential(
            nn.Conv2d(in_ch,dim,kernel_size=4,stride=2,padding=1)
        )
        
        #   240*320
        self.en2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim,dim*2,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(dim*2)
        )
        
        #   120*160
        self.en3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim*2,dim*4,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(dim*4)
        )
        
        #   60*80
        self.en4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim*4,dim*8,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(dim*8)
        )
        
        #   30*40
        self.en5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim*8,dim*8,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(dim*8)
        )
        
        #   15*20

        
        
        #   start decoder
        #   15*20
        self.de1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim* 8, dim* 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim* 8),
            nn.Dropout(p=0.5)
        )
        
        # 16 * 16
        self.de2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim* 8 * 2, dim* 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim* 4),
            nn.Dropout(p=0.5)
        )
        
        # 32 * 32
        self.de3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim* 4 * 2, dim* 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim* 2),
            nn.Dropout(p=0.5)
        )
        
        # 64 * 64
        self.de4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim* 2 * 2, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.Dropout(p=0.5)
        )
        
        # 128 * 128
        self.de5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim* 2, out_ch, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, input):
        en1_out = self.en1(input)
        en2_out = self.en2(en1_out)
        en3_out = self.en3(en2_out)
        en4_out = self.en4(en3_out)
        en5_out = self.en5(en4_out)

        # Decoder
        de1_out = self.de1(en5_out)
        de1_cat = torch.cat([de1_out, en4_out], dim=1)
        de2_out = self.de2(de1_cat)
        de2_cat = torch.cat([de2_out, en3_out], dim=1)
        de3_out = self.de3(de2_cat)
        de3_cat = torch.cat([de3_out, en2_out], dim=1)
        de4_out = self.de4(de3_cat)
        de4_cat = torch.cat([de4_out, en1_out], dim=1)
        de5_out = self.de5(de4_cat)

        return de5_out


class pix2pix_Discriminator(nn.Module):
    def __init__(self,in_ch, out_ch, dim=64):
        super(pix2pix_Discriminator,self).__init__()
        
        # 256 * 256
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch + out_ch, dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 128 * 128
        self.layer2 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 64 * 64
        self.layer3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 32 * 32
        self.layer4 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(dim * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 31 * 31
        self.layer5 = nn.Sequential(
            nn.Conv2d(dim * 8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
        # 30 * 30
    
    def forward(self,input):
        layer1_out = self.layer1(input)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)

        return layer5_out