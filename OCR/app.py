import os
import torch
import numpy as np
import matplotlib.pyplot as plt  
from torchvision import models 
import torch 
#import torch.backends.cudnn as f
from PIL import Image
import torch.nn as nn
#from torchvision import transforms, models 
import warnings
from cv2 import imread
warnings.filterwarnings("ignore") 
from Model import ResNet18 
from flask import Flask, current_app, g,render_template,jsonify,request
from Object_Detection import Object_detection
from OCR import OCR 
from LoadModel import Load_model
from Number_detection import Number_detection
from Object_Detection_OBB import OB_OBB

import logging

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a file handler
file_handler = logging.FileHandler('error.log')
file_handler.setLevel(logging.ERROR)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logging.getLogger('').addHandler(file_handler)
logging.getLogger('').addHandler(console_handler)










weights_box = r'best.pt'
weights_val = r'yolo_val.pt'
weight_ocr =  r'model3.pth'





load_m =  Load_model(weights_box,weights_val,weight_ocr)
yolo_box_model,half,device,yolo_val_model,_,_,ocr_model = load_m()

W = 'yolov8-obb-digit-3-dataset.pt'
yolo_obb = OB_OBB(W)








app = Flask(__name__)
@app.route('/')
def home():
    return render_template('image.html')

@app.route('/upload_image',methods=['POST'])
def upload_image():
    BB_dt = Object_detection(image_size=640,device_type='cpu')
    NM_dt = Number_detection(image_size=640,device_type='cpu')
    if 'image' in request.files:
        image = request.files['image']
        
        rand_id = torch.rand(1)
        save_dir = r'upload_image'
        path = (os.path.join(save_dir, f'shift_{rand_id.item()}.png')) # Saved image path 
    
        image.save(path) # save input images in DataBase
        
        
        img = imread(path) 
        

        
        #plate_region = BB_dt(img,yolo_box_model,half,device) # replace with Ati Code 
        plate_region = yolo_obb(path)

        pil_image = Image.fromarray(plate_region)
# Save the PIL Image
        pil_image.save(os.path.join(save_dir, f'{1}.jpg')) 
        try:
            if plate_region is None:
                raise ValueError("plate_region is None")
            else:

                imgs = NM_dt(plate_region,yolo_val_model,half,device)
                if imgs is None:

                    raise ValueError("imgs are None")
                else:

                    ocr = OCR(imgs,ocr_model)
                    digit_obj = ocr()
                    meters = ''.join(map(str, digit_obj))
                    return render_template('index.html', digits=meters)
                
        except ValueError as e:
            logging.error(f": {str(e)}")  

            

        

       
        out_dir = r'image_BB'
        #out_path = (os.path.join(out_dir, f'shift_{rand_id.item()}.png'))
        #img_BB = Image.fromarray(plate_region)
        #img_BB.save(out_path) 

        return "Salam" 
        
         
        
    
    
        # return render_template('image.html',ocr_digit = digit_obj, img=out_source)
        
        
      


if __name__ == "__main__":
    app.run(debug=True)    



