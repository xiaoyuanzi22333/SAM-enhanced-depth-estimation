import torch
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import cv2
import supervision as sv
from data.reader import SAMDataset
from  torch.utils.data import DataLoader
from tqdm import tqdm
from model.Unet import pix2pix
from torchvision import transforms




type = 'train'
batch_size = 16
DEVICE = 'cuda:2'
cuda_device = 2
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = './SAM/sam_vit_h_4b8939.pth'
modelpath = './U_saved'

Dataset = SAMDataset(type = 'train')
Data_train = DataLoader(Dataset, batch_size=batch_size, shuffle=True)

model = pix2pix()
model = model.cuda(cuda_device)
L1_loss = torch.nn.L1Loss()
    
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3, betas = (0.5, 0.999))
num_epochs = 10

for epoch in range(num_epochs):
    L1 = 0.0
    for img, label, output_sam in tqdm(Data_train):
        if len(img) != batch_size:
                continue
        
        img = img.cuda(cuda_device)
        label = label.cuda(cuda_device)
        output_sam = output_sam.cuda(cuda_device)
        
        
        output1 = model(img)
        output2 = model(output_sam)
        
        output = (output1 + output2)/2
        
        L1 = L1_loss(output,label)
        
        optimizer.zero_grad()
        L1.backward()
        optimizer.step()
        
    if (epoch+1)%5 == 0:
        torch.save(model,modelpath+'/epoch'+str(epoch)+'_save.pth')
    
    print("epoch: ",epoch+1,", train loss: ",L1)

torch.save(model,modelpath+'/final_save.pth')