import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.modeling_outputs import BaseModelOutput
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Load data
with open('episodes.json', 'r') as f:
    data = json.load(f)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Initialize tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# Create dataset and dataloader
dataset = EmbeddingTitleDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# Add an embedding projector
embedding_dim = 1536  # Assuming this is the dimension of your embeddings
model.embedding_projector = nn.Linear(embedding_dim, model.config.d_model)
model.to(device)

# Check if the model file already exists
model_path = './models/e2t_bart_0.2.pth'
if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    # Training loop
    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
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

    print(f"Total number of samples in dataset: {len(dataset)}")
    print(f"Number of batches per epoch: {len(dataloader)}")

    total_samples = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        epoch_samples = 0
        for batch_idx, batch in enumerate(progress_bar):
            embeddings = batch['embedding'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device)
            labels = batch['labels'].to(device)

            # Check for NaN values
            if torch.isnan(embeddings).any() or torch.isnan(input_ids).any() or torch.isnan(labels).any():
                print(f"NaN values detected in batch {batch_idx}. Skipping...")
                continue

            optimizer.zero_grad()
            
            # Project embeddings
            projected_embeddings = model.embedding_projector(embeddings)
            
            # Create encoder outputs
            encoder_outputs = BaseModelOutput(
                last_hidden_state=projected_embeddings.unsqueeze(1),
                hidden_states=None,
                attentions=None
            )
            
            # Create encoder attention mask
            encoder_attention_mask = torch.ones(projected_embeddings.size(0), 1, device=device)
            
            # Pass projected embeddings as encoder_outputs
            outputs = model(input_ids=None,  # Set to None as we're providing encoder_outputs
                            attention_mask=encoder_attention_mask,
                            labels=labels, 
                            encoder_outputs=encoder_outputs,
                            decoder_input_ids=input_ids,  # Use input_ids as decoder_input_ids
                            decoder_attention_mask=attention_mask)  # Use original attention_mask for decoder
            
            loss = outputs.loss
            logits = outputs.logits

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

            # Plot loss after every step
            plot_loss(losses)

            # Print every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

            epoch_samples += len(batch['embedding'])
        total_samples += epoch_samples
        print(f"Samples processed in epoch {epoch+1}: {epoch_samples}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")

    # Update learning rate
    scheduler.step(avg_loss)

    # Plot loss at the end of each epoch
    plot_loss(losses)

    print(f"Total samples processed across all epochs: {total_samples}")

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
        projected_embedding = model.embedding_projector(embedding.unsqueeze(0))
        
        # Create encoder outputs
        encoder_outputs = BaseModelOutput(
            last_hidden_state=projected_embedding.unsqueeze(1),
            hidden_states=None,
            attentions=None
        )
        
        # Create encoder attention mask
        encoder_attention_mask = torch.ones(projected_embedding.size(0), 1, device=device)
        
        outputs = model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            max_length=max_length,
            num_beams=4,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
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




# ------ Sample Output ------

# Model saved as './models/e2t_bart_0.2.pth'

# Testing the model with actual data:

# Sample 1:
# Actual title: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title: In the meantime, we can all agree on one thing: The best way to deal with the problem is to do it ourselves. That's why we're doing it now. We want to be able to help each other out. And

# Sample 2:
# Actual title: Sir David Attenborough: New portrait by Jonathan Yeo unveiled
# Generated title: In the meantime, we can all agree on one thing: The best way to get rid of a problem is to start over. That's why I'm glad to have the option to begin over and over again. It's also why

# Sample 3:
# Actual title: Junk Kouture: Student reaches final of sustainable fashion competition
# Generated title: In addition, there is also the matter of whether or not there are any other options available to the public. This is why I am so glad to have the opportunity to speak with you. It is because of this experience that I have

# Sample 4:
# Actual title: 10 of the best TV shows of 2024 so far
# Generated title: In addition to that, the group also has its own set of rules and regulations. For example, they have to deal with the fact that a lot of people are not familiar with them. That's why they are so different from the

# Sample 5:
# Actual title: A chicken recipe so good its origin is being fought in court
# Generated title: In the meantime, we have a lot of work to do. For example, the following week we had a very busy week at work. We also had an interesting week in the form of an extended period of time in which we worked

# [ ]
