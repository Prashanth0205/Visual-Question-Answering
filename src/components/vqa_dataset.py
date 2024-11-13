import os 
import json 
from PIL import Image
from torch.utils.data import Dataset

class VQA_RAD_Dataset(Dataset):
    """
    A custom dataset for Visual Question Answering on medical images
    given the image directory containing the images and a json file 
    containing the corresponding question-answer pairs 
    """

    def __init__(self, json_dir, img_dir, transform=None):

        with open(json_dir, 'r') as file:
            self.data = json.load(file)

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        '''
        Returns a single image and its corresponding question-answer pair 
        '''

        img_name = os.path.join(self.img_dir , self.data[index]['image_name'])
        question = self.data[index]['question']
        answer = str(self.data[index]['answer'])

        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, question, answer