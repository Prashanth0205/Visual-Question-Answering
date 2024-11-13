import torch
import torch.nn as nn

class ImageTextMatching(nn.Module):
    def __init__(self, vision_model, bert_model, hidden_size):
        """
        A custom model implementing Image-Text matching model that predicts 
        whether a given image and text pair are matching. It combines features
        from vision transformer and text encoder (first 6 layers of BERT model)
        and uses a classification head for binary classification, 0 for negative 
        pairs and 1 for positive pairs
        """
        super(ImageTextMatching, self).__init__()
        self.vision_model = vision_model 
        
        self.text_model = bert_model
        self.text_encoder_layers = nn.ModuleList(bert_model.encoder.layer[:6])
        self.hidden_size = hidden_size
        
        self.projection = nn.Linear(2 * hidden_size, hidden_size)
        
        self.cls_head = nn.Linear(hidden_size, 2)  

    def forward(self, images, input_ids, attention_mask, generate_negatives=True):
        """
        Forward pass including generating positive and negative image-text pairs 
        and classify them as matching or non-matching pairs 
        
        Inputs:
        - images: Batch of images.
        - input_ids: Tokenized text inputs.
        - attention_mask: Attention mask for text.
        - generate_negatives: Whether to generate negative pairs.

        Returns:
        - logits: Logits for positive and negative pairs.
        - labels: Ground-truth labels for each pair.
        """

        image_outputs = self.vision_model(pixel_values=images)
        image_embeddings = image_outputs.last_hidden_state[:, 0, :]  

        text_embeddings = self.extract_text_embeddings(input_ids, attention_mask)
        
        if generate_negatives:
            positive_pairs, negative_pairs, labels = self.create_pairs(image_embeddings, text_embeddings)
        else:
            positive_pairs = torch.cat((image_embeddings, text_embeddings), dim=-1)
            labels = torch.ones(positive_pairs.size(0), dtype=torch.long, device=images.device)
        
        pairs = torch.cat([positive_pairs, negative_pairs], dim=0) if generate_negatives else positive_pairs
        labels = torch.cat([labels, 1 - labels], dim=0) if generate_negatives else labels

        projected_pairs = self.projection(pairs)
        
        logits = self.cls_head(projected_pairs)
        
        return logits, labels

    def extract_text_embeddings(self, input_ids, attention_mask):
        """
        Passes tokenized text through the text encoder to obain text embeddings 
        """

        embeddings = self.text_model.embeddings(input_ids=input_ids)
        
        attention_mask = attention_mask[:, None, None, :] 
        
        for layer in self.text_encoder_layers:
            embeddings = layer(embeddings, attention_mask=attention_mask)[0]

        text_embeddings = embeddings[:, 0, :] 
        
        return text_embeddings

    def create_pairs(self, image_embeddings, text_embeddings):
        """
        Creates positive/ matching and negative/ non-matching pairs given
        image and text embeddings 
        """
        batch_size = image_embeddings.size(0)
        
        positive_pairs = torch.cat((image_embeddings, text_embeddings), dim=-1) 
        labels = torch.ones(batch_size, dtype=torch.long, device=image_embeddings.device)
        
        shuffled_indices = torch.randperm(batch_size)
        negative_text_embeddings = text_embeddings[shuffled_indices]
        negative_pairs = torch.cat((image_embeddings, negative_text_embeddings), dim=-1)  
        
        return positive_pairs, negative_pairs, labels