import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset

def preprocess_fundus_image(image_path, image_size=512): [cite: 597]
    image = cv2.imread(image_path) [cite: 598]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) [cite: 599]
    
    # Resize image [cite: 600]
    image = cv2.resize(image, (image_size, image_size)) [cite: 601]
    
    # Apply CLAHE on each channel [cite: 602]
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB) [cite: 603]
    l, a, b = cv2.split(lab) [cite: 604]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) [cite: 605, 606]
    l = clahe.apply(l) [cite: 607]
    enhanced = cv2.merge((l, a, b)) [cite: 608]
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB) [cite: 609, 610]
    
    # Normalize pixel values [cite: 611]
    enhanced = enhanced / 255.0 [cite: 612]
    return enhanced [cite: 613]

class EyePACSDataset(Dataset): [cite: 617]
    def __init__(self, dataframe, image_dir, transform=None): [cite: 618]
        self.dataframe = dataframe [cite: 619]
        self.image_dir = image_dir [cite: 620]
        self.transform = transform [cite: 621, 623]
        
    def __len__(self): [cite: 622]
        return len(self.dataframe) [cite: 624]
        
    def __getitem__(self, idx): [cite: 625]
        image_name = self.dataframe.iloc[idx]['image'] [cite: 626]
        label = self.dataframe.iloc[idx]['level'] [cite: 627]
        image_path = os.path.join(self.image_dir, image_name + ".jpeg") [cite: 628]
        image = Image.open(image_path).convert("RGB") [cite: 629]
        
        if self.transform: [cite: 630]
            image = self.transform(image) [cite: 631]
            
        return image, torch.tensor(label, dtype=torch.long) [cite: 634]
