import torch
import torch.nn as nn

class MaskedLanguageModeling(nn.Module):
    def __init__(self, bert_model, image_encoder, mask_token_id, pad_token_id, mask_prob=0.15):
        super(MaskedLanguageModeling, self).__init__()
        
        # Full BERT model to access embeddings directly
        self.bert_model = bert_model
        self.text_encoder_layers = nn.ModuleList(bert_model.encoder.layer[:6])  # First 6 layers for text encoding
        self.multimodal_encoder_layers = nn.ModuleList(bert_model.encoder.layer[6:])  # Last 6 layers for multimodal fusion
        
        # Shared image encoder (e.g., Vision Transformer)
        self.image_encoder = image_encoder

        # Prediction head for MLM
        hidden_size = bert_model.config.hidden_size
        self.prediction_head = nn.Linear(hidden_size, bert_model.config.vocab_size)
        
        # Masking probability and token IDs
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

    def mask_tokens(self, input_ids):
        labels = input_ids.clone()
        mask = (torch.rand(input_ids.shape, device=input_ids.device) < self.mask_prob) & (input_ids != self.pad_token_id)
        input_ids[mask] = self.mask_token_id
        labels[~mask] = -100  # Ignore unmasked tokens in the loss
        return input_ids, labels

    def forward(self, images, tokenized_text):
        input_ids, attention_mask = tokenized_text['input_ids'], tokenized_text['attention_mask']
        
        # Image Encoding
        image_features = self.image_encoder(pixel_values=images).last_hidden_state  # Shape: (batch_size, num_patches, hidden_size)
        
        # Text Encoding with masking
        masked_input_ids, labels = self.mask_tokens(input_ids)
        
        # Use embeddings directly from BERT
        text_embeddings = self.bert_model.embeddings(masked_input_ids)
        
        # Pass through the first 6 layers of BERT for text encoding
        for layer in self.text_encoder_layers:
            text_embeddings = layer(text_embeddings)[0]
        
        # Multimodal Interaction
        combined_features = torch.cat((image_features, text_embeddings), dim=1)  # Concatenate along sequence dimension
        multimodal_features = combined_features
        for layer in self.multimodal_encoder_layers:
            multimodal_features = layer(multimodal_features)[0]
        
        # MLM Prediction
        text_predictions = self.prediction_head(multimodal_features[:, -text_embeddings.size(1):, :])  # Only predict on text portion
        
        return text_predictions, labels