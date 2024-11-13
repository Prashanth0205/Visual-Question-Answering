import os
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms 
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, ViTModel

from chestxray_dataset import ChestXrayDataset
from mim import MaskedImageModeling, masked_loss
from mlm import MaskedLanguageModeling
from itm import ImageTextMatching
from itcl import ImageTextContrastiveLearning

def main(data_loader, num_epochs=1, learning_rate=1e-4, batch_size=32, 
         model_save_path="model_checkpoints", encoder_save_path="encoder_checkpoints"):
    """
    Main function to train the models of all the defined pre-trained tasks such as MIM, 
    MLM, ITM and ITCL. 

    Inputs: 
    - data_loader: Data_loader for training data containing batches of images and its captions 
    - num_epochs: Number of training epochs 
    - learning_rate: Learning rate 
    - batch_size: batch size for training 
    - model_save_path: Directory for saving the model checkpoints 
    - encoder_save_path: Directory for saving the text and image encoder checkpoints 
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id

    mim_model = MaskedImageModeling(image_encoder=image_encoder, hidden_size=768, patch_size=16).to(device)
    mlm_model = MaskedLanguageModeling(bert_model=bert_model, image_encoder=image_encoder, mask_token_id=mask_token_id, pad_token_id=pad_token_id).to(device)
    itm_model = ImageTextMatching(vision_model=image_encoder, bert_model=bert_model, hidden_size=768).to(device)
    itc_model = ImageTextContrastiveLearning(image_encoder=image_encoder, text_encoder=bert_model, hidden_size=768).to(device)

    unique_param_ids = set()
    optimizer_params = []

    for model_part in [mlm_model.image_encoder, 
                    mlm_model.text_encoder_layers, 
                    mlm_model.multimodal_encoder_layers, 
                    itm_model,
                    itc_model]:
        
        for param in model_part.parameters():
            if id(param) not in unique_param_ids:
                optimizer_params.append(param)
                unique_param_ids.add(id(param))

    optimizer = torch.optim.Adam(optimizer_params, lr=learning_rate)

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

            optimizer.zero_grad()

            reconstructed_patches, mask = mim_model(images, mask_probability=0.15)
            loss_mim = masked_loss(mim_model, reconstructed_patches, images, mask)
            total_loss_mim += loss_mim.item()

            text_predictions, labels = mlm_model(images, {'input_ids': input_ids, 'attention_mask': attention_mask})
            text_predictions = text_predictions.view(-1, bert_model.config.vocab_size)
            labels = labels.view(-1)
            loss_mlm = F.cross_entropy(text_predictions, labels, ignore_index=-100)
            total_loss_mlm += loss_mlm.item()

            logits, labels_itm = itm_model(images, input_ids, attention_mask, generate_negatives=True)
            loss_itm = F.cross_entropy(logits, labels_itm)
            total_loss_itm += loss_itm.item()

            loss_itc = itc_model(images, input_ids, attention_mask)
            total_loss_itc += loss_itc.item()

            combined_loss = loss_mim + loss_mlm + loss_itm + loss_itc
            combined_loss.backward()
            optimizer.step()

            progress_bar.set_postfix({
                "MIM Loss": loss_mim.item(),
                "MLM Loss": loss_mlm.item(),
                "ITM Loss": loss_itm.item(),
                "ITC Loss": loss_itc.item(),
                "Combined Loss": combined_loss.item()
            })

        avg_loss_mim = total_loss_mim / len(data_loader)
        avg_loss_mlm = total_loss_mlm / len(data_loader)
        avg_loss_itm = total_loss_itm / len(data_loader)
        avg_loss_itc = total_loss_itc / len(data_loader)
        avg_combined_loss = (avg_loss_mim + avg_loss_mlm + avg_loss_itm + avg_loss_itc) / 4

        print(f"Epoch [{epoch+1}/{num_epochs}] - MIM Loss: {avg_loss_mim:.4f}, MLM Loss: {avg_loss_mlm:.4f}, ITM Loss: {avg_loss_itm:.4f}, ITC Loss: {avg_loss_itc:.4f}, Combined Loss: {avg_combined_loss:.4f}")

        torch.save({
            'epoch': epoch,
            'mim_model_state_dict': mim_model.state_dict(),
            'mlm_model_state_dict': mlm_model.state_dict(),
            'itm_model_state_dict': itm_model.state_dict(),
            'itc_model_state_dict': itc_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'combined_loss': avg_combined_loss,
        }, os.path.join(model_save_path, f"combined_checkpoint_epoch_{epoch+1}.pth"))

        torch.save(image_encoder.state_dict(), os.path.join(encoder_save_path, f"image_encoder_checkpoint_epoch_{epoch+1}.pth"))
        torch.save(bert_model.state_dict(), os.path.join(encoder_save_path, f"bert_model_checkpoint_epoch_{epoch+1}.pth"))

        print(f"Checkpoints saved for epoch {epoch + 1}")

    print("Training complete.")


if __name__ == "__main__":

    import warnings 
    warnings.filterwarnings('ignore')

    img_dir = r"C:\Users\omrav\OneDrive\Desktop\IITC AI\Fall 24\CS 512 Computer Vision\Project\Dataset\Indiana University - Chest X-Rays\images\images"
    csv_path = r"C:\Users\omrav\OneDrive\Desktop\IITC AI\Fall 24\CS 512 Computer Vision\Project\Dataset\Indiana University - Chest X-Rays\indiana_chest_xray_captions.csv"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ChestXrayDataset(csv_file=csv_path, img_dir=img_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    main(data_loader=data_loader)