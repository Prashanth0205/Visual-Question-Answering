import os 
import pandas as pd 
from PIL import Image

from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    """
    A custom dataset for loading chest X-ray images given the image directory 
    and the csv contatining the captions for image captioning/ pretraining tasks.
    """
    
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, index):
        '''
        Returns a single image and its corresponding caption given the index x
        '''

        img_name = os.path.join(self.img_dir , self.data_frame['filename'].iloc[index])
        caption = self.data_frame['impression'].iloc[index]

        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, caption