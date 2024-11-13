import torch 
import torch.nn as nn
from transformers import BertLMHeadModel

class VQAModel(nn.Module):
    def __init__(self, bert, vit, answer_decoder_config):
        """
        A Visual Question Answering model that combines both text and image embeddings 
        using pre-trained text and image encoder trained using the various pre-trainings 
        tasks. The model architecture includes text encoding, multi-modal encoding and 
        a answer decoder for answer generation 
        """
        
        super(VQAModel, self).__init__()

        self.bert = bert
        self.text_encoder = nn.ModuleList(self.bert.encoder.layer[:6])  
        self.multimodal_encoder = nn.ModuleList(self.bert.encoder.layer[6:])  
        self.image_encoder = vit  

        self.answer_decoder = BertLMHeadModel(answer_decoder_config)

    def forward(self, input_ids, attention_mask, images, decoder_input_ids, target_ids=None):
        """
        Forward pass including image and text encoding, multimodal encoding and answer decoding 

        Inputs: 
        - input_ids: Token IDs for the question 
        - attention_mask: Attention mask for the question 
        - images: Batch of images 
        - decoder_input_ids: Token IDs for answer generation 
        - target_ids: Ground truth token IDs for answer 

        Returns:
        - logits: Logits from the answer decoder 
        - loss: Loss of the prediction if target_ids are provided 
        """
        
        text_embeddings = self._encode_text(input_ids, attention_mask)
        image_embeddings = self.image_encoder(images).last_hidden_state
        combined_embeddings = torch.cat((text_embeddings, image_embeddings), dim=1)

        combined_attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.size(0), image_embeddings.size(1)), device=attention_mask.device)],
            dim=1
        )

        multimodal_embeddings = self._encode_multimodal(combined_embeddings, combined_attention_mask)

        outputs = self.answer_decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=multimodal_embeddings,
            labels=target_ids  
        )

        logits = outputs.logits 
        loss = outputs.loss if target_ids is not None else None  

        return {"logits": logits, "loss": loss}

    def _encode_text(self, input_ids, attention_mask):
        """
        Encodes the question text using the given pre-trained text encoder 
        """
        
        text_embeddings = self.bert.embeddings(input_ids=input_ids)
        
        attention_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2).float()
        attention_mask_expanded = (1.0 - attention_mask_expanded) * -10000.0

        for layer in self.text_encoder:
            text_embeddings = layer(text_embeddings, attention_mask=attention_mask_expanded)[0]
        
        return text_embeddings

    def _encode_multimodal(self, combined_embeddings, combined_attention_mask):
        """
        Returns multimodal representations using the given pre-trained mutlimodal 
        encoder given the combined embeddings of image and text 
        """
        
        combined_attention_mask_expanded = combined_attention_mask.unsqueeze(1).unsqueeze(2).float()
        combined_attention_mask_expanded = (1.0 - combined_attention_mask_expanded) * -10000.0

        for layer in self.multimodal_encoder:
            combined_embeddings = layer(combined_embeddings, attention_mask=combined_attention_mask_expanded)[0]

        return combined_embeddings