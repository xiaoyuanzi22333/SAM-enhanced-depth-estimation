import os
import pandas as pd
import torch
import numpy as np
from  torch.utils.data import Dataset,DataLoader
import cv2
from torchvision import transforms
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import supervision as sv




def paired(type = 'train'):
    DATA_PATH = './dataset/nyu2_' + type
    EXCEL_PATH = './dataset/nyu2_' + type + '.csv'
    
    df = pd.read_csv(EXCEL_PATH,names=['image','label'],header=None)
    
    return df, DATA_PATH

class SAMDataset(Dataset):
    
    def __init__(self, DEVICE=2, type = 'train'):
        self.origin = []
        self.label = []
        
        MODEL_TYPE = "vit_h"
        CHECKPOINT_PATH = './SAM/sam_vit_h_4b8939.pth'
        
        
        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
        sam.to(device=DEVICE)
        mask_generator = SamAutomaticMaskGenerator(sam)
        
        self.generator = mask_generator
        
        df, DATA_PATH = paired(type)
        
        for folders in os.listdir(DATA_PATH):
            path = DATA_PATH + '/' + folders
            length = len(os.listdir(path))
            for i in range(int(length/2)):
                origin = path + '/' + str(i+1) + '.jpg'
                label = path + '/' + str(i+1) + '.png'
                self.origin.append(origin)
                self.label.append(label)
    
    
    def __getitem__(self, index):
        mask_generator = self.gene
        origin = cv2.imread(self.origin[index])
        label = cv2.imread(self.label[index])
        
        origin = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY) 
        
        sam_out = mask_generator.generate(origin)
        mask_annotator = sv.MaskAnnotator()
        detections = sv.Detections.from_sam(sam_out)
        annotated_image = mask_annotator.annotate(origin, detections)
        
        transform = transforms.Compose([transforms.ToTensor()])
        output_sam = transform(annotated_image)
        
        label = label/255
        label = torch.from_numpy(np.transpose(label[...,None],(2,0,1))).float()
        
        transform = transforms.Compose([transforms.ToTensor()])
        origin = transform(origin)
        
        return origin, label, output_sam
    
    def __len__(self):
        return len(self.origin)
    
    
class MyDataset(Dataset):
    
    def __init__(self, type = 'train'):
        self.origin = []
        self.label = []
        
        df, DATA_PATH = paired(type)
        
        for folders in os.listdir(DATA_PATH):
            path = DATA_PATH + '/' + folders
            length = len(os.listdir(path))
            for i in range(int(length/2)):
                origin = path + '/' + str(i+1) + '.jpg'
                label = path + '/' + str(i+1) + '.png'
                self.origin.append(origin)
                self.label.append(label)
    
    
    def __getitem__(self, index):
        origin = cv2.imread(self.origin[index])
        label = cv2.imread(self.label[index])
        
        origin = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY) 
        
        
        label = label/255
        label = torch.from_numpy(np.transpose(label[...,None],(2,0,1))).float()
        
        transform = transforms.Compose([transforms.ToTensor()])
        origin = transform(origin)
        
        return origin, label
    
    def __len__(self):
        return len(self.origin)
        

# DATA_PATH = '../dataset/nyu2_' + 'train'
# for folders in os.listdir(DATA_PATH):
#     path = DATA_PATH + '/' + folders
#     length = len(os.listdir(path))
#     for i in range(int(length/2)):
#         origin = path + '/' + str(i+1) + '.jpg'
#         label = path + '/' + str(i+1) + '.png'
        
        
#         origin = cv2.imread(origin)
#         label = cv2.imread(label)
        
#         origin = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
#         label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY) 
        
#         print(origin.shape)
#         print(label.shape)
        # self.origin.append(origin)
        # self.label.append(label)