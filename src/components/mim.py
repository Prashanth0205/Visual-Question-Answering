import torch
import torch.nn as nn 

class MaskedImageModeling(nn.Module):
    def __init__(self, image_encoder, hidden_size, patch_size, num_decoder_layers=8):
        super(MaskedImageModeling, self).__init__()
        self.image_encoder = image_encoder  # Use shared image encoder
        self.hidden_size = hidden_size
        self.patch_size = patch_size

        # Define an 8-layer transformer for decoding
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Learnable placeholder embedding for masked patches
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        # Linear layer as prediction head to reconstruct pixel values of masked patches
        self.prediction_head = nn.Linear(hidden_size, patch_size * patch_size * 3)

    def forward(self, x, mask_probability=0.15):
        # Extract patch embeddings from the image encoder
        vit_output = self.image_encoder(pixel_values=x)
        patch_embeddings = vit_output.last_hidden_state[:, 1:, :]  # Remove class token
        
        # Apply masking to patch embeddings
        masked_embeddings, mask = self.apply_mask(patch_embeddings, mask_probability)
        
        # Decode the embeddings with context from unmasked patches
        decoded_embeddings = self.decoder(masked_embeddings, patch_embeddings)
        
        # Reconstruct the pixel values of each patch with the prediction head
        reconstructed_patches = self.prediction_head(decoded_embeddings)
        
        return reconstructed_patches, mask

    def apply_mask(self, patch_embeddings, mask_probability):
        batch_size, num_patches, hidden_size = patch_embeddings.shape
        mask = (torch.rand(batch_size, num_patches, device=patch_embeddings.device) < mask_probability)
        
        # Replace masked patches with the mask token
        mask_token_expanded = self.mask_token.expand(batch_size, num_patches, hidden_size)
        masked_embeddings = torch.where(mask.unsqueeze(-1), mask_token_expanded, patch_embeddings)

        return masked_embeddings, mask
    
    def patchify(self, x):
        """
        Reshape the image into patches (B, C, H, W) -> (B, num_patches, patch_size*patch_size*C)
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, self.patch_size * self.patch_size * C)
        return x



def masked_loss(mae_model, reconstructed_patches, original_images, mask):
    """
    Calculate L2 loss only on masked patches.
    """
    # Convert the original image into patches
    original_patches = mae_model.patchify(original_images)
    
    # Expand the mask to match the shape of original_patches along the channel dimension
    mask = mask.unsqueeze(-1).expand_as(original_patches)
    
    # Apply mask to both original and reconstructed patches to calculate loss only for masked patches
    masked_original = original_patches * mask
    masked_reconstructed = reconstructed_patches * mask
    
    # Calculate MSE loss on masked patches
    loss_fn = nn.MSELoss()
    loss = loss_fn(masked_reconstructed, masked_original)
    
    return loss
