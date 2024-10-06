import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os
import random


# Define the Generator class
class Generator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, seq_length):
        super(Generator, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.lstm = nn.LSTMCell(output_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # Initialize a token embedding layer for input tokens
        self.token_embedding = nn.Embedding(output_dim, output_dim)

    def forward(self, embedding, target_titles=None, teacher_forcing_ratio=0.5):
        batch_size = embedding.size(0)
        device = embedding.device

        # Initialize hidden and cell states
        h_t = torch.tanh(embedding @ torch.randn(embedding.size(1), self.hidden_dim).to(device))
        c_t = torch.zeros(batch_size, self.hidden_dim).to(device)

        # Initialize input token (start with <SOS> token)
        input_token = torch.full((batch_size,), tokenizer.bos_token_id, dtype=torch.long, device=device)

        outputs = []

        for t in range(self.seq_length):
            # Get embedding of the input token
            input_embedded = self.token_embedding(input_token)

            # LSTM step
            h_t, c_t = self.lstm(input_embedded, (h_t, c_t))

            # Output layer
            output = self.fc_out(h_t)  # [batch_size, output_dim]
            outputs.append(output.unsqueeze(1))

            # Decide whether to use teacher forcing
            teacher_force = False
            if target_titles is not None:
                teacher_force = random.random() < teacher_forcing_ratio

            # Get the next input token
            if teacher_force:
                input_token = target_titles[:, t]  # Use actual next token
            else:
                input_token = output.argmax(dim=1)  # Use predicted token

        outputs = torch.cat(outputs, dim=1)  # [batch_size, seq_length, output_dim]
        return outputs


# Define the Discriminator class
class Discriminator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Check if x is token IDs (integers), in which case we apply the embedding layer
        if x.dtype == torch.long:
            x = self.embedding(x)  # [batch_size, seq_length, embed_dim]
        elif x.dtype == torch.float:
            # Assume x is logits from generator: [batch_size, seq_length, vocab_size]
            # Convert logits to probabilities and compute expected embeddings
            probs = torch.softmax(x, dim=-1)  # [batch_size, seq_length, vocab_size]
            x = torch.matmul(probs, self.embedding.weight)  # [batch_size, seq_length, embed_dim]
        else:
            raise ValueError("Input data type not recognized.")

        # Pass through LSTM
        out, _ = self.lstm(x)  # [batch_size, seq_length, hidden_dim]

        # Use the output from the last time step
        out = out[:, -1, :]  # [batch_size, hidden_dim]
        out = self.fc(out)   # [batch_size, 1]
        return out

# Define the Dataset class
class EmbeddingTitleDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=50):
        self.tokenizer = tokenizer
        self.titles = [item['title'] for item in data]
        self.embeddings = [
            torch.tensor(self.parse_embedding(item['embedding']), dtype=torch.float)
            for item in data
        ]
        self.max_length = max_length

    def parse_embedding(self, embedding_str):
        # Convert embedding string to a list of floats
        embedding_str = embedding_str.strip('[]')
        return [float(x) for x in embedding_str.split(',')]

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        title = self.titles[idx]
        embedding = self.embeddings[idx]

        # Tokenize the title
        encoded_title = self.tokenizer.encode(
            title,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'embedding': embedding,
            'title_ids': encoded_title.squeeze(0)  # Remove extra batch dimension
        }

# Load data
data_path = './episodes.json'  # Adjust the path as needed
if not os.path.exists(data_path):
    print(f"Data file not found at {data_path}")
    # Handle the missing file as appropriate
else:
    with open(data_path, 'r') as f:
        data = json.load(f)

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Create dataset and dataloader
dataset = EmbeddingTitleDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model parameters
embedding_dim = 1536
hidden_dim = 256
output_dim = tokenizer.vocab_size
seq_length = dataset.max_length  # Assuming max_length=50

# Initialize models
generator = Generator(
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    seq_length=seq_length
)

discriminator = Discriminator(
    vocab_size=tokenizer.vocab_size,
    embed_dim=embedding_dim,
    hidden_dim=hidden_dim
)

# Move models to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator.to(device)
discriminator.to(device)

# Loss functions and optimizers
adversarial_loss = nn.BCELoss()
title_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Adjusted learning rates
optimizer_G = optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))  # Increased LR for generator
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))

# Training parameters
num_epochs = 30
# lambda_adv = 5.0  # Adjust as needed

# Adjust lambda_adv
lambda_adv = 1.0  # Start with 1.0 and adjust based on training performance

early_stopping_patience = 5
best_g_loss = float('inf')
epochs_without_improvement = 0

# Define the generate_title function
def generate_title(generator, tokenizer, embedding, max_length=50, temperature=1.0):
    generator.eval()
    with torch.no_grad():
        embedding = embedding.to(device)
        input_token = torch.full((1,), tokenizer.bos_token_id, dtype=torch.long, device=device)
        h_t = torch.tanh(embedding @ generator.fc_h.weight.T)
        c_t = torch.zeros(1, generator.hidden_dim).to(device)
        generated_tokens = []

        for _ in range(max_length):
            input_embedded = generator.token_embedding(input_token)
            h_t, c_t = generator.lstm(input_embedded, (h_t, c_t))
            output = generator.fc_out(h_t)
            output = output / temperature  # Apply temperature
            probs = torch.softmax(output, dim=-1)
            input_token = torch.multinomial(probs, num_samples=1).squeeze()
            generated_tokens.append(input_token.item())
            if input_token.item() == tokenizer.eos_token_id:
                break

        generated_title = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_title


# def generate_title(generator, tokenizer, embedding, max_length=50):
#     was_training = generator.training  # Save the original mode
#     generator.eval()
#     with torch.no_grad():
#         embedding = embedding.to(device)
#         fake_titles = generator(embedding.unsqueeze(0))  # [1, seq_length, vocab_size]
#         fake_title_ids = fake_titles.argmax(dim=-1)      # [1, seq_length]
#         generated_title = tokenizer.decode(
#             fake_title_ids.squeeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=True
#         )
#         generator.train(mode=was_training)  # Restore the original mode
#         return generated_title

# Training loop
# Training loop
for epoch in range(num_epochs):
    d_losses = []
    g_losses = []
    g_adv_losses = []
    g_title_losses = []

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        embeddings = batch['embedding'].to(device)  # [batch_size, embedding_dim]
        real_titles = batch['title_ids'].to(device)  # [batch_size, seq_length]
        batch_size = embeddings.size(0)

        # Define labels for this batch
        real_labels = torch.ones((batch_size, 1), device=device)
        fake_labels = torch.zeros((batch_size, 1), device=device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Real data
        real_output = discriminator(real_titles)
        d_loss_real = adversarial_loss(real_output, real_labels)

        # Generate fake titles
        fake_titles = generator(embeddings, target_titles=None, teacher_forcing_ratio=0.0)
        fake_titles_ids = fake_titles.argmax(dim=-1)

        # Fake data
        fake_output = discriminator(fake_titles_ids.detach())
        d_loss_fake = adversarial_loss(fake_output, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Generate fake titles with teacher forcing
        fake_titles = generator(embeddings, target_titles=real_titles, teacher_forcing_ratio=0.5)
        fake_titles_ids = fake_titles.argmax(dim=-1)

        # Adversarial loss
        fake_output = discriminator(fake_titles_ids)
        g_loss_adv = adversarial_loss(fake_output, real_labels)

        # Title loss
        fake_titles_flat = fake_titles.view(-1, output_dim)
        real_titles_flat = real_titles.view(-1)
        g_loss_title = title_loss(fake_titles_flat, real_titles_flat)

        # Total generator loss
        g_loss = lambda_adv * g_loss_adv + g_loss_title
        g_loss.backward()
        optimizer_G.step()

        # Append losses for monitoring
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        g_adv_losses.append(g_loss_adv.item())
        g_title_losses.append(g_loss_title.item())

    # Compute average losses
    avg_d_loss = sum(d_losses) / len(d_losses)
    avg_g_loss = sum(g_losses) / len(g_losses)
    avg_g_adv_loss = sum(g_adv_losses) / len(g_adv_losses)
    avg_g_title_loss = sum(g_title_losses) / len(g_title_losses)

    print(f"Epoch [{epoch+1}/{num_epochs}] D_loss: {avg_d_loss:.4f} G_loss: {avg_g_loss:.4f} (Adv: {avg_g_adv_loss:.4f}, Title: {avg_g_title_loss:.4f})")

    # Early stopping check
    if avg_g_loss < best_g_loss:
        best_g_loss = avg_g_loss
        epochs_without_improvement = 0
        # Save the best model
        torch.save(generator.state_dict(), 'gan_generator_best.pth')
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= early_stopping_patience:
        print("Early stopping triggered.")
        break

    # Generate a sample title
    test_embedding = dataset[0]['embedding'].to(device)
    actual_title = dataset.titles[0]
    print(f"Actual title at epoch {epoch+1}: {actual_title}")
    generated_title = generate_title(generator, tokenizer, test_embedding)
    print(f"Generated title at epoch {epoch+1}: {generated_title}")

# Save the final trained generator
torch.save(generator.state_dict(), 'gan_generator_final.pth')


# Test the model with an actual sample from the dataset
sample_idx = 0  # Change this index to test with different samples
test_embedding = dataset[sample_idx]['embedding'].to(device)
actual_title = dataset.titles[sample_idx]
generated_title = generate_title(generator, tokenizer, test_embedding)
print(f"\nActual title: {actual_title}")
print(f"Generated title: {generated_title}")




# 결과 좋지 않음