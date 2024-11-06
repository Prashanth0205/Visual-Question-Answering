import copy 

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageTextContrastiveLearning(nn.Module):
    def __init__(self, image_encoder, text_encoder, hidden_size, projection_dim=256, momentum=0.995):
        super(ImageTextContrastiveLearning, self).__init__()
        self.image_encoder = image_encoder  # Shared image encoder
        self.text_encoder = text_encoder  # Shared text encoder (e.g., BERT model)
        self.projection_dim = projection_dim
        self.momentum = momentum

        # Projection layers for image and text features
        self.image_projection = nn.Linear(hidden_size, projection_dim)
        self.text_projection = nn.Linear(hidden_size, projection_dim)

        # Momentum encoders (deep copy of the original encoders)
        self.momentum_image_encoder = copy.deepcopy(self.image_encoder)
        self.momentum_text_encoder = copy.deepcopy(self.text_encoder)
        self.momentum_image_projection = nn.Linear(hidden_size, projection_dim)
        self.momentum_text_projection = nn.Linear(hidden_size, projection_dim)

        # Ensure the momentum encoders are initialized with the same weights as the main encoders
        self._initialize_momentum_encoders()

    def _initialize_momentum_encoders(self):
        # Copy parameters from main encoders to momentum encoders
        for param, momentum_param in zip(self.image_encoder.parameters(), self.momentum_image_encoder.parameters()):
            momentum_param.data.copy_(param.data)
            momentum_param.requires_grad = False

        for param, momentum_param in zip(self.text_encoder.parameters(), self.momentum_text_encoder.parameters()):
            momentum_param.data.copy_(param.data)
            momentum_param.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        # Apply the momentum update to both image and text encoders
        for param, momentum_param in zip(self.image_encoder.parameters(), self.momentum_image_encoder.parameters()):
            momentum_param.data = self.momentum * momentum_param.data + (1 - self.momentum) * param.data

        for param, momentum_param in zip(self.text_encoder.parameters(), self.momentum_text_encoder.parameters()):
            momentum_param.data = self.momentum * momentum_param.data + (1 - self.momentum) * param.data

    def forward(self, images, input_ids, attention_mask):
        # Main encoding and projection
        image_features = self.image_encoder(pixel_values=images).last_hidden_state[:, 0, :]  # CLS token
        image_features = F.normalize(self.image_projection(image_features), dim=-1)

        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        text_features = F.normalize(self.text_projection(text_features), dim=-1)

        # Momentum encoding and projection for contrastive learning
        with torch.no_grad():
            self._momentum_update()  # Update momentum encoders

            momentum_image_features = self.momentum_image_encoder(pixel_values=images).last_hidden_state[:, 0, :]
            momentum_image_features = F.normalize(self.momentum_image_projection(momentum_image_features), dim=-1)

            momentum_text_features = self.momentum_text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            momentum_text_features = F.normalize(self.momentum_text_projection(momentum_text_features), dim=-1)

        # Contrastive loss between main and momentum features
        contrastive_loss = self.calculate_contrastive_loss(image_features, text_features, momentum_image_features, momentum_text_features)
        return contrastive_loss

    def calculate_contrastive_loss(self, image_features, text_features, momentum_image_features, momentum_text_features):
        # Compute similarities and contrastive loss
        batch_size = image_features.size(0)
        temperature = 0.07

        # Similarity scores
        sim_image_text = torch.mm(image_features, text_features.t()) / temperature
        sim_image_momentum_text = torch.mm(image_features, momentum_text_features.t()) / temperature
        sim_momentum_image_text = torch.mm(momentum_image_features, text_features.t()) / temperature

        # Labels for contrastive loss
        labels = torch.arange(batch_size).to(image_features.device)

        # Contrastive loss across both directions
        loss = F.cross_entropy(sim_image_text, labels) + F.cross_entropy(sim_image_momentum_text, labels) + F.cross_entropy(sim_momentum_image_text, labels)
        return loss / 3  # Average the loss