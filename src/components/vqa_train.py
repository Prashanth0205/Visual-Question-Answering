import os 
from tqdm import tqdm 

from vqa_model import VQAModel
from vqa_dataset import VQA_RAD_Dataset

import torch 
from torchvision import transforms 
from torchmetrics import BLEUScore
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW, BertTokenizer, BertModel, ViTModel, BertConfig, ViTConfig

def custom_collate_fn(batch):
    """
    Custom collate function for batching in Dataloader 
    """
    
    images, questions, answers = zip(*batch)
    images = torch.stack(images)  
    questions = list(questions)  
    answers = list(answers)      
    return images, questions, answers

def load_pretrained_models(bert_checkpoint_path, vit_checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load pre-trained BERT and Vision Transformer models from the pre-training stage.
    
    Inputs:
    - bert_checkpoint_path: Path to the BERT model checkpoint
    - vit_checkpoint_path: Path to the Vision Transformer model checkpoint 
    - device: Device to load models onto, 'cuda' or 'cpu'
    
    Returns:
    - bert_model: Loaded BERT model with pre-trained weights.
    - vit_model: Loaded Vision Transformer model with pre-trained weights.
    """
    
    bert_config = BertConfig.from_pretrained('bert-base-uncased')
    bert_model = BertModel(bert_config)
    bert_checkpoint = torch.load(bert_checkpoint_path, map_location=device)
    bert_model.load_state_dict(bert_checkpoint)
    bert_model.to(device)
    bert_model.eval() 

    vit_config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
    vit_model = ViTModel(vit_config)
    vit_checkpoint = torch.load(vit_checkpoint_path, map_location=device)
    vit_model.load_state_dict(vit_checkpoint)
    vit_model.to(device)
    vit_model.eval()  

    return bert_model, vit_model

def train_vqa_model(model, dataloader, tokenizer, device='cuda', num_epochs=3, learning_rate=1e-5, model_checkpoint_path="vqa_model_checkpoint.pth"):
    """
    Function to train a VQA model and tracks metrics such as loss and BLEU score

    Inputs: 
    - model: Initialized VQA model with the pre-trained encoders 
    - dataloader: Dataloader with training data containing batches of images, question and answers 
    - tokenzier: Tokenizer for encoding question and answers 
    - device: device for model training either 'cuda' OR 'cpu'
    - num_epochs: number of training epochs 
    - learning_rate: learning rate for the optimizer 
    - model_checkpoint_path: file path to save and load model checkpoints 
    """
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    bleu_metric = BLEUScore()
    
    start_epoch = 0
    if os.path.exists(model_checkpoint_path):
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch + 1}")

    model.to(device)
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        bleu_scores = []

        for images, questions, answers in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = images.to(device)
            optimizer.zero_grad()

            encoding = tokenizer(questions, padding='max_length', truncation=True, max_length=12, return_tensors='pt')
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            target_ids = tokenizer(answers, padding='max_length', truncation=True, max_length=12, return_tensors='pt')['input_ids'].to(device)
            decoder_input_ids = torch.full((input_ids.size(0), target_ids.size(1)), tokenizer.cls_token_id, device=device)

            outputs = model(input_ids, attention_mask, images, decoder_input_ids, target_ids)
            loss = outputs['loss']  
            logits = outputs['logits'] 

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            generated_answers = tokenizer.batch_decode(logits.argmax(dim=-1), skip_special_tokens=True)
            bleu_scores.extend([bleu_metric(gen, [ref]).item() for gen, ref in zip(generated_answers, answers)])

        avg_loss = total_loss / len(dataloader)
        avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}, Avg BLEU: {avg_bleu_score:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'bleu_score': avg_bleu_score
        }, model_checkpoint_path)
        print(f"Model checkpoint saved at epoch {epoch + 1}")

    print("Training complete.")


if __name__ == "__main__":
    train_ratio = 0.8
    batch_size = 4
    epochs = 20  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_dir = r"C:\Users\omrav\OneDrive\Desktop\IITC AI\Fall 24\CS 512 Computer Vision\Project\Dataset\VQA_RAD\VQA_RAD Image Folder"
    json_dir = r"C:\Users\omrav\OneDrive\Desktop\IITC AI\Fall 24\CS 512 Computer Vision\Project\Dataset\VQA_RAD\VQA_RAD Dataset Public.json"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = VQA_RAD_Dataset(json_dir=json_dir, img_dir=image_dir, transform=transform)

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_checkpoint_path = r"C:\Users\omrav\OneDrive\Desktop\IITC AI\Fall 24\CS 512 Computer Vision\Project\bert_model_checkpoint.pth"
    vit_checkpoint_path = r"C:\Users\omrav\OneDrive\Desktop\IITC AI\Fall 24\CS 512 Computer Vision\Project\image_encoder_checkpoint.pth"
    bert, vit = load_pretrained_models(bert_checkpoint_path, vit_checkpoint_path, device=device)

    answer_decoder_config = BertConfig.from_pretrained('bert-base-uncased')
    answer_decoder_config.is_decoder = True
    answer_decoder_config.add_cross_attention = True

    model = VQAModel(bert, vit, answer_decoder_config).to(device)

    model_path = r"C:\Users\omrav\OneDrive\Desktop\IITC AI\Fall 24\CS 512 Computer Vision\Project\Code Notebooks\vqa_model_checkpoint.pth"
    metric_history = train_vqa_model(model=model, 
                                    dataloader=train_dataloader, 
                                    tokenizer=tokenizer, 
                                    device=device, 
                                    num_epochs=epochs, 
                                    learning_rate=1e-5, 
                                    model_checkpoint_path=model_path)