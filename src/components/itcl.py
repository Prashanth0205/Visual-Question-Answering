import copy 

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageTextContrastiveLearning(nn.Module):
    def __init__(self, image_encoder, text_encoder, hidden_size, projection_dim=256, momentum=0.995):
        """
        A custom model implementing Image-Text Contrastive Learning which uses a contrastive loss 
        that pulls matching image-text pairs closer and pushes non-matching pairs apart. It uses 
        momemtum encoders to improving training stability inspired from the base paper
        """

        super(ImageTextContrastiveLearning, self).__init__()
        self.image_encoder = image_encoder  
        self.text_encoder = text_encoder 
        self.projection_dim = projection_dim
        self.momentum = momentum

        self.image_projection = nn.Linear(hidden_size, projection_dim)
        self.text_projection = nn.Linear(hidden_size, projection_dim)

        self.momentum_image_encoder = copy.deepcopy(self.image_encoder)
        self.momentum_text_encoder = copy.deepcopy(self.text_encoder)
        self.momentum_image_projection = nn.Linear(hidden_size, projection_dim)
        self.momentum_text_projection = nn.Linear(hidden_size, projection_dim)

        self._initialize_momentum_encoders()

    def _initialize_momentum_encoders(self):
        """
        Initializes the momentum encoders by copying the weights from the main 
        encoders and freezing them to prevent gradient updates 
        """

        for param, momentum_param in zip(self.image_encoder.parameters(), self.momentum_image_encoder.parameters()):
            momentum_param.data.copy_(param.data)
            momentum_param.requires_grad = False

        for param, momentum_param in zip(self.text_encoder.parameters(), self.momentum_text_encoder.parameters()):
            momentum_param.data.copy_(param.data)
            momentum_param.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        """
        Updates the weights of the momentum encoders by applying a weighted 
        average of the main encoder's parameters and the current momentum encoder's parameters
        """

        for param, momentum_param in zip(self.image_encoder.parameters(), self.momentum_image_encoder.parameters()):
            momentum_param.data = self.momentum * momentum_param.data + (1 - self.momentum) * param.data

        for param, momentum_param in zip(self.text_encoder.parameters(), self.momentum_text_encoder.parameters()):
            momentum_param.data = self.momentum * momentum_param.data + (1 - self.momentum) * param.data

    def forward(self, images, input_ids, attention_mask):
        """
        Forward pass to obtain contrastive embeddings for images and text, updates the momentum 
        encoders and computes the contrastive loss 
        """

        image_features = self.image_encoder(pixel_values=images).last_hidden_state[:, 0, :]  
        image_features = F.normalize(self.image_projection(image_features), dim=-1)

        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        text_features = F.normalize(self.text_projection(text_features), dim=-1)

        with torch.no_grad():
            self._momentum_update()  

            momentum_image_features = self.momentum_image_encoder(pixel_values=images).last_hidden_state[:, 0, :]
            momentum_image_features = F.normalize(self.momentum_image_projection(momentum_image_features), dim=-1)

            momentum_text_features = self.momentum_text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            momentum_text_features = F.normalize(self.momentum_text_projection(momentum_text_features), dim=-1)

        contrastive_loss = self.calculate_contrastive_loss(image_features, text_features, momentum_image_features, momentum_text_features)
        return contrastive_loss

    def calculate_contrastive_loss(self, image_features, text_features, momentum_image_features, momentum_text_features):
        """
        Calculates the contrastive loss using temperature scaled similarities between pairs of images 
        and text features and their momentum 
        """
        
        batch_size = image_features.size(0)
        temperature = 0.07

        sim_image_text = torch.mm(image_features, text_features.t()) / temperature
        sim_image_momentum_text = torch.mm(image_features, momentum_text_features.t()) / temperature
        sim_momentum_image_text = torch.mm(momentum_image_features, text_features.t()) / temperature

        labels = torch.arange(batch_size).to(image_features.device)

        loss = F.cross_entropy(sim_image_text, labels) + F.cross_entropy(sim_image_momentum_text, labels) + F.cross_entropy(sim_momentum_image_text, labels)
        return loss / 3  