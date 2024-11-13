import torch
import torch.nn as nn

class MaskedLanguageModeling(nn.Module):
    def __init__(self, bert_model, image_encoder, mask_token_id, pad_token_id, mask_prob=0.15):
        """
        A custom model for Masked Language Modelling that combines text and image embeddings 
        to predict masked tokens in the text/ captions. It uses the first 6 layers of the 
        BERT model as text encoder, last 6 layers of the BERT model as multi-modal encoder 
        and the 12 layers vision transformer as the image encoder. 
        """
        super(MaskedLanguageModeling, self).__init__()
        
        self.bert_model = bert_model
        self.text_encoder_layers = nn.ModuleList(bert_model.encoder.layer[:6]) 
        self.multimodal_encoder_layers = nn.ModuleList(bert_model.encoder.layer[6:])  
        
        self.image_encoder = image_encoder

        hidden_size = bert_model.config.hidden_size
        self.prediction_head = nn.Linear(hidden_size, bert_model.config.vocab_size)
        
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

    def mask_tokens(self, input_ids):
        """
        Mask tokens in the given input text based on the given masked probability 
        """
        labels = input_ids.clone()
        mask = (torch.rand(input_ids.shape, device=input_ids.device) < self.mask_prob) & (input_ids != self.pad_token_id)
        input_ids[mask] = self.mask_token_id
        labels[~mask] = -100 
        return input_ids, labels

    def forward(self, images, tokenized_text):
        """
        Forward pass of the model combining image and text embeddings to 
        predict the masked tokens in the text  

        Inputs: 
        - x: Batch of input images (size = (B, C, H, W))
        - tokenized_text: Dictionary containing tokenized text with input_ids and attention_mask 

        Returns:
        - text_predictions: Predicted logits for masked tokens 
        - labels: Ground truth for masked tokens 
        """

        input_ids, attention_mask = tokenized_text['input_ids'], tokenized_text['attention_mask']
        
        image_features = self.image_encoder(pixel_values=images).last_hidden_state 
        
        masked_input_ids, labels = self.mask_tokens(input_ids)
        
        text_embeddings = self.bert_model.embeddings(masked_input_ids)
        
        for layer in self.text_encoder_layers:
            text_embeddings = layer(text_embeddings)[0]
        
        combined_features = torch.cat((image_features, text_embeddings), dim=1) 
        multimodal_features = combined_features
        for layer in self.multimodal_encoder_layers:
            multimodal_features = layer(multimodal_features)[0]
     
        text_predictions = self.prediction_head(multimodal_features[:, -text_embeddings.size(1):, :])  
        
        return text_predictions, labels