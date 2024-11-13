import torch
import torch.nn as nn 

class MaskedImageModeling(nn.Module):
    """
    A custom model for Masked Image Modelling which uses a 12-layer vision transformer 
    as the image encoder and s 8-layer transformer as the image decoder with the 
    ImageNet pretrained weights
    """

    def __init__(self, image_encoder, hidden_size, patch_size, num_decoder_layers=8):
        super(MaskedImageModeling, self).__init__()
        self.image_encoder = image_encoder  
        self.hidden_size = hidden_size
        self.patch_size = patch_size

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        self.prediction_head = nn.Linear(hidden_size, patch_size * patch_size * 3)

    def forward(self, x, mask_probability=0.15):
        """
        Forward pass of the model including image encoding, applying patch embeddings, 
        decoding masked embeddings and reconstrcting masked patches. 

        Inputs: 
        - x: Batch of input images (size = (B, C, H, W))
        - mask_probability: Probability of masking each patch of the image 

        Returns:
        - reconstructed_patches: Reconstructed patch embeddings 
        - mask: Boolean mask containing which patches were masked 
        """

        vit_output = self.image_encoder(pixel_values=x)
        patch_embeddings = vit_output.last_hidden_state[:, 1:, :] 
        
        masked_embeddings, mask = self.apply_mask(patch_embeddings, mask_probability)
        
        decoded_embeddings = self.decoder(masked_embeddings, patch_embeddings)
        
        reconstructed_patches = self.prediction_head(decoded_embeddings)
        
        return reconstructed_patches, mask

    def apply_mask(self, patch_embeddings, mask_probability):
        """
        Masks a given set of patch embeddings based on the given mask probability  
        """
        batch_size, num_patches, hidden_size = patch_embeddings.shape
        mask = (torch.rand(batch_size, num_patches, device=patch_embeddings.device) < mask_probability)
        
        mask_token_expanded = self.mask_token.expand(batch_size, num_patches, hidden_size)
        masked_embeddings = torch.where(mask.unsqueeze(-1), mask_token_expanded, patch_embeddings)

        return masked_embeddings, mask
    
    def patchify(self, x):
        """
        Reshape the image into patches
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, self.patch_size * self.patch_size * C)
        return x



def masked_loss(mae_model, reconstructed_patches, original_images, mask):
    """
    Calculate L2 loss only on masked patches.
    """
    original_patches = mae_model.patchify(original_images)
    
    mask = mask.unsqueeze(-1).expand_as(original_patches)
    
    masked_original = original_patches * mask
    masked_reconstructed = reconstructed_patches * mask
    
    loss_fn = nn.MSELoss()
    loss = loss_fn(masked_reconstructed, masked_original)
    
    return loss
