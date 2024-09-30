# train_EmbeddingToTitleModel.py
# transformer decoder with embedding projector
# input: embedding (1536)
# output: title
# 3 epochs
# 8 batch size

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Load data
with open('episodes.json', 'r') as f:
    data = json.load(f)

class EmbeddingTitleDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=50):
        self.tokenizer = tokenizer
        self.titles = [item['title'] for item in data]
        self.embeddings = [torch.tensor(self.parse_embedding(item['embedding']), dtype=torch.float) for item in data]
        self.max_length = max_length

    def parse_embedding(self, embedding_str):
        return [float(x) for x in embedding_str.strip('[]').split(',')]

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

        input_ids = encoded_title['input_ids'].squeeze(0)
        attention_mask = encoded_title['attention_mask'].squeeze(0)

        # Shift labels to ignore padding tokens in the loss
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            'embedding': embedding,
            'input_ids': input_ids,
            'labels': labels
        }

class EmbeddingToTitleModel(nn.Module):
    def __init__(self, embedding_dim=1536, hidden_size=768, vocab_size=None):
        super().__init__()
        self.embedding_projector = nn.Linear(embedding_dim, hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def forward(self, embeddings, input_ids, labels=None):
        # embeddings: (batch_size, embedding_dim)
        # input_ids: (batch_size, seq_len)
        # Project embeddings
        batch_size = embeddings.size(0)
        projected_embeddings = self.embedding_projector(embeddings)  # (batch_size, hidden_size)
        # Expand to (seq_len=1, batch_size, hidden_size)
        memory = projected_embeddings.unsqueeze(0)  # (1, batch_size, hidden_size)

        # Prepare target sequence embeddings
        tgt_embeddings = self.token_embedding(input_ids).transpose(0, 1)  # (seq_len, batch_size, hidden_size)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_embeddings.size(0)).to(tgt_embeddings.device)

        output = self.decoder(tgt_embeddings, memory, tgt_mask=tgt_mask)  # (seq_len, batch_size, hidden_size)

        logits = self.output_layer(output.transpose(0, 1))  # (batch_size, seq_len, vocab_size)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, logits
        else:
            return logits

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 tokenizer doesn't have a pad token by default

# Create dataset and dataloader
dataset = EmbeddingTitleDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize model
model = EmbeddingToTitleModel(embedding_dim=1536, hidden_size=768, vocab_size=len(tokenizer))

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Check if the model file already exists
model_path = 'embedding_to_title_model.pth'
if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    num_epochs = 20

    print("Starting training...")
    start_time = time.time()
    losses = []

    def plot_loss(losses, clear=True):
        if clear:
            clear_output(wait=True)
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.show()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            embeddings = batch['embedding'].to(device)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Check for NaN values
            if torch.isnan(embeddings).any() or torch.isnan(input_ids).any() or torch.isnan(labels).any():
                print(f"NaN values detected in batch {batch_idx}. Skipping...")
                continue

            optimizer.zero_grad()
            loss, logits = model(embeddings, input_ids, labels)
            
            # Check if loss is NaN
            if torch.isnan(loss):
                print(f"NaN loss detected in batch {batch_idx}. Skipping...")
                continue

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Check for None or NaN gradients
            skip_step = False
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"Warning: Gradient for {name} is None.")
                    skip_step = True
                    break
                elif torch.isnan(param.grad).any():
                    print(f"Warning: Gradient for {name} contains NaN values.")
                    skip_step = True
                    break

            if skip_step:
                continue

            optimizer.step()

            total_loss += loss.item()
            losses.append(loss.item())
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Plot loss every 100 batches
            if (batch_idx + 1) % 100 == 0:
                plot_loss(losses)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")

        # Plot loss at the end of each epoch
        plot_loss(losses)

    # Print training summary
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds.")

    # Plot final loss graph
    plot_loss(losses, clear=False)

    # Save the model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as '{model_path}'")

# Function to generate titles
def generate_title(model, tokenizer, embedding, max_length=50):
    model.eval()
    with torch.no_grad():
        embedding = embedding.to(device)
        projected_embedding = model.embedding_projector(embedding.unsqueeze(0))  # (1, hidden_size)
        memory = projected_embedding.unsqueeze(0)  # (1, 1, hidden_size)

        input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)
        generated_tokens = []

        for _ in range(max_length):
            tgt_embeddings = model.token_embedding(input_ids).transpose(0, 1)  # (seq_len, batch_size, hidden_size)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_embeddings.size(0)).to(device)

            output = model.decoder(tgt_embeddings, memory, tgt_mask=tgt_mask)
            logits = model.output_layer(output[-1])  # (batch_size, vocab_size)
            
            # Use temperature sampling instead of greedy decoding
            temperature = 0.7
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token_id.item())
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(1)], dim=1)

        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text

# Test the model with actual data
print("\nTesting the model with actual data:")
num_test_samples = 5  # Number of samples to test

for i in range(num_test_samples):
    test_item = data[i]  # Get a test item from the loaded data
    actual_title = test_item['title']
    test_embedding = torch.tensor(dataset.parse_embedding(test_item['embedding']), dtype=torch.float).to(device)
    
    generated_title = generate_title(model, tokenizer, test_embedding)
    
    print(f"\nSample {i+1}:")
    print(f"Actual title: {actual_title}")
    print(f"Generated title: {generated_title}")
# Test the model
# test_embedding = torch.randn(1536)
# generated_title = generate_title(model, tokenizer, test_embedding)
# print(f"Generated title: {generated_title}")



# ------ Sample Output ------

# Testing the model with actual data:

# Sample 1:
# Actual title: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title:  preacher�������������������������������������������������

# Sample 2:
# Actual title: Sir David Attenborough: New portrait by Jonathan Yeo unveiled
# Generated title: Uation parkAging of of of of of of of of of of of of of of of of of of of of of of of of of of of of of of of of of of of of of of of of of of of of of

# Sample 3:
# Actual title: Junk Kouture: Student reaches final of sustainable fashion competition
# Generated title:  masksia we we we we we we we we we we we we we we we we we we we we we we we we we we we we we we we we we we we we we we we we we we we we we we we we

# Sample 4:
# Actual title: 10 of the best TV shows of 2024 so far
# Generated title:  firing seniorelelelelelelel Serbialyn Apps Apps AppsOD Paper villageausSenior�������������������������������

# Sample 5:
# Actual title: A chicken recipe so good its origin is being fought in court
# Generated title:  OUT work ConsciousMacMac hope Min Through exitun pulled111111111111111111111111111111111itement Seed 80nessnessness