import os
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset,random_split
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts
from torchvision import transforms, models 
import warnings
warnings.filterwarnings("ignore")


class Mobile_Net(nn.Module):
    def __init__(self,num_class = 2):
        super(Mobile_Net,self).__init__()

        self.model = models.mobilenet_v2(pretrained=True) 
        self.freez()
        self.fine_tune() 

    def freez (self):
        for param in self.model.parameters():
            param.requires_grad = False

       
          
    
    def fine_tune(self):
            
            
            l = [17,18]
            for i in l:
                for param in self.model.features[i].parameters():
                    param.requires_grad = True  
            self.model.classifier = nn.Sequential(
                 nn.Dropout(p =0.2),
                 nn.Linear(1280,256),
                 nn.ReLU(),
                 nn.Linear(256,10),
                 nn.Softmax(dim = 1)
                 )
    

    def forward(self,x):
        
        return self.model(x)
    



