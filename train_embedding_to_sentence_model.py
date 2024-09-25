import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, EncoderDecoderModel, AdamW, BertConfig
import ast
import os
import torch.cuda.amp as amp
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load data
with open('episodes.json', 'r') as f:
    data = json.load(f)

# Custom dataset
class TitleDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.titles = [item['title'] for item in data]
        # Convert string embeddings to float tensors
        self.embeddings = [torch.tensor(self.parse_embedding(item['embedding']), dtype=torch.float) for item in data]
        self.max_length = max_length

    def parse_embedding(self, embedding_str):
        # Convert string representation of list to actual list of floats
        return ast.literal_eval(embedding_str)

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        title = self.titles[idx]
        embedding = self.embeddings[idx]

        encoded_title = self.tokenizer(
            title,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'embedding': embedding,
            'input_ids': encoded_title['input_ids'].squeeze(),
            'attention_mask': encoded_title['attention_mask'].squeeze(),
            'labels': encoded_title['input_ids'].squeeze()
        }

# Custom model
class CustomEncoderDecoder(nn.Module):
    def __init__(self, base_model, embedding_dim):
        super().__init__()
        self.base_model = base_model
        encoder_hidden_size = base_model.config.encoder.hidden_size
        self.embedding_projector = nn.Linear(embedding_dim, encoder_hidden_size)


    def forward(self, embedding, decoder_input_ids=None, attention_mask=None, labels=None):
        projected_embedding = self.embedding_projector(embedding)
        if len(projected_embedding.shape) == 2:
            projected_embedding = projected_embedding.unsqueeze(1)
        outputs = self.base_model(
            inputs_embeds=projected_embedding,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

def generate_title(model, tokenizer, embedding, max_length=50):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        embedding = embedding.to(device)
        projected_embedding = model.embedding_projector(embedding).unsqueeze(0).unsqueeze(1)

        outputs = model.base_model.generate(
            inputs_embeds=projected_embedding,
            max_length=max_length
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoder_config = BertConfig.from_pretrained('bert-base-uncased')
decoder_config = BertConfig.from_pretrained('bert-base-uncased')


base_model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased',
                                                                 encoder_config=encoder_config,
                                                                 decoder_config=decoder_config)
model = CustomEncoderDecoder(base_model, embedding_dim=1536)

# Load the model if it exists
model_path = 'embedding_to_sentence_model.pth'
if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}")
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Remove unexpected keys
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)

    model.load_state_dict(model_dict, strict=False)

    if torch.cuda.is_available():
        model.cuda()


    print(f"Model loaded successfully and moved to {'GPU' if torch.cuda.is_available() else 'CPU'}")
else:
    print("No existing model found. Please train the model first.")

# Set the model to evaluation mode
model.eval()

# Example usage
new_embedding = torch.randn(1536)  # Create a single embedding vector
new_title = generate_title(model, tokenizer, new_embedding)
print(f"Generated title: {new_title}")

# Generate titles for a few examples from the dataset
print("\nGenerating titles for sample embeddings from the dataset:")
for i in range(5):
    sample = dataset[i]
    sample_embedding = sample['embedding'].unsqueeze(0).to(device)
    sample_title = generate_title(model, tokenizer, sample_embedding)
    print(f"Original: {dataset.titles[i]}")
    print(f"Generated: {sample_title}\n")