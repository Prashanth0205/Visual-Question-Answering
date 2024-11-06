import torch
import torch.nn as nn

class ImageTextMatching(nn.Module):
    def __init__(self, vision_model, bert_model, hidden_size):
        super(ImageTextMatching, self).__init__()
        self.vision_model = vision_model  # Vision Transformer (ViT) for images
        
        # Store the full BERT model and select only the first 6 layers for the text encoder
        self.text_model = bert_model
        self.text_encoder_layers = nn.ModuleList(bert_model.encoder.layer[:6])
        self.hidden_size = hidden_size
        
        # Projection layer to map (2 * hidden_size) to (hidden_size)
        self.projection = nn.Linear(2 * hidden_size, hidden_size)
        
        # Linear layer as the classification head for ITM task
        self.cls_head = nn.Linear(hidden_size, 2)  # Output logits for match/mismatch

    def forward(self, images, input_ids, attention_mask, generate_negatives=True):
        """
        Forward pass for the ITM task with positive and negative pairs.
        
        Parameters:
        - images: torch.Tensor, shape (B, C, H, W) - Batch of images.
        - input_ids: torch.Tensor, shape (B, T) - Tokenized text inputs.
        - attention_mask: torch.Tensor, shape (B, T) - Attention mask for text.
        - generate_negatives: bool - Whether to generate negative pairs.

        Returns:
        - logits: torch.Tensor, shape (2B, 2) - Logits for positive and negative pairs.
        - labels: torch.Tensor, shape (2B) - Ground-truth labels for each pair.
        """
        # 1. Extract image embeddings using vision model
        image_outputs = self.vision_model(pixel_values=images)
        image_embeddings = image_outputs.last_hidden_state[:, 0, :]  # Use CLS token embedding

        # 2. Extract text embeddings using the first 6 layers of BERT
        text_embeddings = self.extract_text_embeddings(input_ids, attention_mask)
        
        # Combine image and text embeddings to form positive and negative pairs
        if generate_negatives:
            positive_pairs, negative_pairs, labels = self.create_pairs(image_embeddings, text_embeddings)
        else:
            # Use only positive pairs (no negative samples)
            positive_pairs = torch.cat((image_embeddings, text_embeddings), dim=-1)
            labels = torch.ones(positive_pairs.size(0), dtype=torch.long, device=images.device)
        
        # Concatenate positive and negative pairs
        pairs = torch.cat([positive_pairs, negative_pairs], dim=0) if generate_negatives else positive_pairs
        labels = torch.cat([labels, 1 - labels], dim=0) if generate_negatives else labels

        # Project pairs from (2 * hidden_size) to (hidden_size)
        projected_pairs = self.projection(pairs)
        
        # 4. Pass through classification head to get logits for match/mismatch
        logits = self.cls_head(projected_pairs)
        
        return logits, labels

    def extract_text_embeddings(self, input_ids, attention_mask):
        """
        Passes input_ids and attention_mask through the first 6 layers of BERT.
        
        Returns:
        - text_embeddings: torch.Tensor, shape (B, hidden_size)
        """
        # Pass input through the BERT embedding layer
        embeddings = self.text_model.embeddings(input_ids=input_ids)
        
        # Ensure attention_mask has the correct shape for broadcasting
        attention_mask = attention_mask[:, None, None, :]  # Shape: (batch_size, 1, 1, sequence_length)
        
        # Apply only the first 6 layers of the encoder
        for layer in self.text_encoder_layers:
            embeddings = layer(embeddings, attention_mask=attention_mask)[0]

        # Use the CLS token embedding as the text representation
        text_embeddings = embeddings[:, 0, :]  # Shape (B, hidden_size)
        
        return text_embeddings

    def create_pairs(self, image_embeddings, text_embeddings):
        """
        Create positive and negative pairs from image and text embeddings.
        
        Returns:
        - positive_pairs: torch.Tensor, shape (B, hidden_size) - Positive pairs (original).
        - negative_pairs: torch.Tensor, shape (B, hidden_size) - Negative pairs (misaligned).
        - labels: torch.Tensor, shape (B) - Labels for the pairs.
        """
        batch_size = image_embeddings.size(0)
        
        # Positive pairs
        positive_pairs = torch.cat((image_embeddings, text_embeddings), dim=-1)  # Shape (B, 2 * hidden_size)
        labels = torch.ones(batch_size, dtype=torch.long, device=image_embeddings.device)
        
        # Negative pairs - shuffle text embeddings within the batch
        shuffled_indices = torch.randperm(batch_size)
        negative_text_embeddings = text_embeddings[shuffled_indices]
        negative_pairs = torch.cat((image_embeddings, negative_text_embeddings), dim=-1)  # Shape (B, 2 * hidden_size)
        
        return positive_pairs, negative_pairs, labels