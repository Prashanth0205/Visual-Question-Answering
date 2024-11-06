import os
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms 
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, ViTModel # type: ignore

from dataset import ChestXrayDataset
from mim import MaskedImageModeling, masked_loss
from mlm import MaskedLanguageModeling
from itm import ImageTextMatching
from itcl import ImageTextContrastiveLearning

# import warnings 
# warnings.filterwarnings('ignore')

def main(data_loader, num_epochs=1, learning_rate=1e-4, batch_size=32, 
         model_save_path="model_checkpoints", encoder_save_path="encoder_checkpoints"):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load shared encoders
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenizer information
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id

    # Initialize models for each pretraining task
    mim_model = MaskedImageModeling(image_encoder=image_encoder, hidden_size=768, patch_size=16).to(device)
    mlm_model = MaskedLanguageModeling(bert_model=bert_model, image_encoder=image_encoder, mask_token_id=mask_token_id, pad_token_id=pad_token_id).to(device)
    itm_model = ImageTextMatching(vision_model=image_encoder, bert_model=bert_model, hidden_size=768).to(device)
    itc_model = ImageTextContrastiveLearning(image_encoder=image_encoder, text_encoder=bert_model, hidden_size=768).to(device)

    unique_param_ids = set()
    # Step 2: Create a list to collect unique parameters
    optimizer_params = []

    # Step 3: Gather parameters from each model and avoid duplicates
    for model_part in [mlm_model.image_encoder, 
                    mlm_model.text_encoder_layers, 
                    mlm_model.multimodal_encoder_layers, 
                    itm_model,
                    itc_model]:
        
        for param in model_part.parameters():
            # Step 4: Ensure each parameter is added only once
            if id(param) not in unique_param_ids:
                optimizer_params.append(param)
                unique_param_ids.add(id(param))

    # Step 5: Initialize the optimizer with unique parameters
    optimizer = torch.optim.Adam(optimizer_params, lr=learning_rate)

    # Unified optimizer for all models
    # optimizer = optim.Adam(list(mim_model.parameters()) + list(mlm_model.parameters()) + list(itm_model.parameters()) + list(itc_model.parameters()), lr=learning_rate)

    # Create directories for saving models and encoders
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(encoder_save_path, exist_ok=True)

    for epoch in range(num_epochs):
        mim_model.train()
        mlm_model.train()
        itm_model.train()
        itc_model.train()

        progress_bar = tqdm(data_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        
        total_loss_mim, total_loss_mlm, total_loss_itm, total_loss_itc = 0, 0, 0, 0

        for i, (images, captions) in enumerate(progress_bar):
            images = images.to(device)
            tokenized = tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(device)
            input_ids, attention_mask = tokenized['input_ids'], tokenized['attention_mask']

            # Zero gradients for the optimizer
            optimizer.zero_grad()

            # 1. Masked Image Modeling (MIM)
            reconstructed_patches, mask = mim_model(images, mask_probability=0.15)
            loss_mim = masked_loss(mim_model, reconstructed_patches, images, mask)
            total_loss_mim += loss_mim.item()

            # 2. Masked Language Modeling (MLM)
            text_predictions, labels = mlm_model(images, {'input_ids': input_ids, 'attention_mask': attention_mask})
            text_predictions = text_predictions.view(-1, bert_model.config.vocab_size)
            labels = labels.view(-1)
            loss_mlm = F.cross_entropy(text_predictions, labels, ignore_index=-100)
            total_loss_mlm += loss_mlm.item()

            # 3. Image-Text Matching (ITM)
            logits, labels_itm = itm_model(images, input_ids, attention_mask, generate_negatives=True)
            loss_itm = F.cross_entropy(logits, labels_itm)
            total_loss_itm += loss_itm.item()

            # 4. Image-Text Contrastive (ITC)
            loss_itc = itc_model(images, input_ids, attention_mask)
            total_loss_itc += loss_itc.item()

            # Combined loss
            combined_loss = loss_mim + loss_mlm + loss_itm + loss_itc
            combined_loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix({
                "MIM Loss": loss_mim.item(),
                "MLM Loss": loss_mlm.item(),
                "ITM Loss": loss_itm.item(),
                "ITC Loss": loss_itc.item(),
                "Combined Loss": combined_loss.item()
            })

        # Average losses for the epoch
        avg_loss_mim = total_loss_mim / len(data_loader)
        avg_loss_mlm = total_loss_mlm / len(data_loader)
        avg_loss_itm = total_loss_itm / len(data_loader)
        avg_loss_itc = total_loss_itc / len(data_loader)
        avg_combined_loss = (avg_loss_mim + avg_loss_mlm + avg_loss_itm + avg_loss_itc) / 4

        print(f"Epoch [{epoch+1}/{num_epochs}] - MIM Loss: {avg_loss_mim:.4f}, MLM Loss: {avg_loss_mlm:.4f}, ITM Loss: {avg_loss_itm:.4f}, ITC Loss: {avg_loss_itc:.4f}, Combined Loss: {avg_combined_loss:.4f}")

        # Save model checkpoints
        torch.save({
            'epoch': epoch,
            'mim_model_state_dict': mim_model.state_dict(),
            'mlm_model_state_dict': mlm_model.state_dict(),
            'itm_model_state_dict': itm_model.state_dict(),
            'itc_model_state_dict': itc_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'combined_loss': avg_combined_loss,
        }, os.path.join(model_save_path, f"combined_checkpoint_epoch_{epoch+1}.pth"))

        # Save shared encoder checkpoints separately
        torch.save(image_encoder.state_dict(), os.path.join(encoder_save_path, f"image_encoder_checkpoint_epoch_{epoch+1}.pth"))
        torch.save(bert_model.state_dict(), os.path.join(encoder_save_path, f"bert_model_checkpoint_epoch_{epoch+1}.pth"))

        print(f"Checkpoints saved for epoch {epoch + 1}")

    print("Training complete.")


# Execute the main function
if __name__ == "__main__":
    img_dir = r"C:\Users\omrav\OneDrive\Desktop\IITC AI\Fall 24\CS 512 Computer Vision\Project\Dataset\Indiana University - Chest X-Rays\images\images"
    csv_path = r"C:\Users\omrav\OneDrive\Desktop\IITC AI\Fall 24\CS 512 Computer Vision\Project\Dataset\Indiana University - Chest X-Rays\indiana_chest_xray_captions.csv"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ChestXrayDataset(csv_file=csv_path, img_dir=img_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    main(data_loader=data_loader)