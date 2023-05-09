import torch
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import cv2
import supervision as sv
from data.reader import MyDataset
from  torch.utils.data import DataLoader
from tqdm import tqdm
from model.Unet import pix2pix

def CrossEntropyLoss_func(loss_weights):
    return torch.nn.CrossEntropyLoss(loss_weights, ignore_index=-1)

type = 'train'
batch_size = 16
DEVICE = 'cuda:2'
cuda_device = 2
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = './SAM/sam_vit_h_4b8939.pth'
modelpath = './U_saved'

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam)
print('model get')

Dataset = MyDataset(type = 'train')
Data_train = DataLoader(Dataset, batch_size=batch_size, shuffle=True)

model = pix2pix()
model = model.cuda(cuda_device)
L1_loss = torch.nn.L1Loss()
    
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3, betas = (0.5, 0.999))
num_epochs = 10

for epoch in range(num_epochs):
    L1 = 0.0
    for img, label in tqdm(Data_train):
        if len(img) != batch_size:
                continue
        
        img = img.cuda(cuda_device)
        label = label.cuda(cuda_device)
        
        output = model(img)
        
        L1 = L1_loss(output,label)
        
        optimizer.zero_grad()
        L1.backward()
        optimizer.step()
        
    if (epoch+1)%5 == 0:
        torch.save(model,modelpath+'/epoch'+str(epoch)+'_save.pth')
    
    print("epoch: ",epoch+1,", train loss: ",L1)

torch.save(model,modelpath+'/final_save.pth')