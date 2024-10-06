import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os
import random

# Transformer imports
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer

# Define the Transformer-based Generator class
class TransformerGenerator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, seq_length, num_layers=6, nhead=8):
        super(TransformerGenerator, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Project embedding to hidden dimension
        self.embedding_proj = nn.Linear(embedding_dim, hidden_dim)

        # Token embedding
        self.token_embedding = nn.Embedding(output_dim, hidden_dim)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, hidden_dim))  # Shape: [1, seq_length, hidden_dim]

        # Transformer decoder with batch_first=False (default)
        decoder_layer = TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, embedding, target_titles=None, teacher_forcing_ratio=0.5):
        batch_size = embedding.size(0)
        device = embedding.device

        # Prepare memory (encoder output)
        memory = self.embedding_proj(embedding).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        memory = memory.permute(1, 0, 2)  # [1, batch_size, hidden_dim]

        # Decide whether to use teacher forcing
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if target_titles is not None and use_teacher_forcing:
            # Teacher Forcing Path
            input_tokens = target_titles[:, :-1]  # Exclude last token
            input_embedded = self.token_embedding(input_tokens)  # [batch_size, seq_length-1, hidden_dim]

            # Add positional encoding
            input_embedded += self.positional_encoding[:, :input_embedded.size(1), :]

            # Permute for transformer [seq_length-1, batch_size, hidden_dim]
            input_embedded = input_embedded.permute(1, 0, 2)  # [tgt_len, batch_size, hidden_dim]

            # Generate mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(input_embedded.size(0)).to(device)

            # Decode
            output = self.transformer_decoder(
                tgt=input_embedded,
                memory=memory,
                tgt_mask=tgt_mask
            )  # [seq_length-1, batch_size, hidden_dim]

            # Output layer
            output = self.fc_out(output.permute(1, 0, 2))  # [batch_size, seq_length-1, output_dim]
            return output
        else:
            # Non-Teacher Forcing Path (Autoregressive Generation)
            generated_tokens = []
            outputs = []
            input_token = torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)  # [batch_size, 1]

            for _ in range(self.seq_length - 1):
                # Embed the entire sequence generated so far
                input_embedded = self.token_embedding(input_token)  # [batch_size, t+1, hidden_dim]
                input_embedded += self.positional_encoding[:, :input_embedded.size(1), :]  # [1, t+1, hidden_dim]
                input_embedded = input_embedded.permute(1, 0, 2)  # [t+1, batch_size, hidden_dim]

                # Generate mask
                tgt_len = input_embedded.size(0)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)  # [t+1, t+1]

                # Decode
                output = self.transformer_decoder(
                    tgt=input_embedded,
                    memory=memory,
                    tgt_mask=tgt_mask
                )  # [t+1, batch_size, hidden_dim]

                # Take the last output (current time step)
                output = output[-1, :, :]  # [batch_size, hidden_dim]
                output = self.fc_out(output)  # [batch_size, output_dim]
                outputs.append(output.unsqueeze(1))  # [batch_size, 1, output_dim]

                # Get the next input token
                next_token = output.argmax(dim=-1)  # [batch_size]
                generated_tokens.append(next_token)
                input_token = torch.cat([input_token, next_token.unsqueeze(1)], dim=1)  # [batch_size, t+2]

            outputs = torch.cat(outputs, dim=1)  # [batch_size, seq_length - 1, output_dim]
            return outputs

# Define the Transformer-based Discriminator class
class TransformerDiscriminator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=6, nhead=8, max_seq_length=500):
        super(TransformerDiscriminator, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_length, embed_dim))

        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch_size, seq_length]
        device = x.device

        # Check if x is token IDs (integers), in which case we apply the embedding layer
        if x.dtype == torch.long:
            x = self.token_embedding(x)  # [batch_size, seq_length, embed_dim]
        elif x.dtype == torch.float:
            # Assume x is logits from generator: [batch_size, seq_length, vocab_size]
            # Convert logits to probabilities and compute expected embeddings
            probs = torch.softmax(x, dim=-1)  # [batch_size, seq_length, vocab_size]
            x = torch.matmul(probs, self.token_embedding.weight)  # [batch_size, seq_length, embed_dim]
        else:
            raise ValueError("Input data type not recognized.")

        seq_length = x.size(1)
        x += self.positional_encoding[:seq_length].unsqueeze(0).to(device)

        x = x.permute(1, 0, 2)  # [seq_length, batch_size, embed_dim]
        out = self.transformer_encoder(x)  # [seq_length, batch_size, embed_dim]

        # Pooling (e.g., take the mean over time steps)
        out = out.mean(dim=0)  # [batch_size, embed_dim]
        out = self.fc(out)     # [batch_size, 1]
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
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # Reduced batch size if memory issues

# Model parameters
embedding_dim = 1536
hidden_dim = 256
output_dim = tokenizer.vocab_size
seq_length = dataset.max_length  # Assuming max_length=50

# Initialize models
generator = TransformerGenerator(
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    seq_length=seq_length,
    num_layers=4,  # Adjust as needed
    nhead=8
)

discriminator = TransformerDiscriminator(
    vocab_size=tokenizer.vocab_size,
    embed_dim=hidden_dim,
    hidden_dim=hidden_dim,
    num_layers=4,  # Adjust as needed
    nhead=8,
    max_seq_length=seq_length
)

# Move models to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator.to(device)
discriminator.to(device)

# Loss functions and optimizers
adversarial_loss = nn.BCELoss()
title_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Adjusted learning rates
optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))

# Training parameters
num_epochs = 30

# Adjust lambda_adv
lambda_adv = 1.0  # Start with 1.0 and adjust based on training performance

early_stopping_patience = 5
best_g_loss = float('inf')
epochs_without_improvement = 0

# Define the generate_title function
def generate_title(generator, tokenizer, embedding, max_length=50, temperature=1.0):
    generator.eval()
    with torch.no_grad():
        embedding = embedding.to(device).unsqueeze(0)  # [1, embedding_dim]
        memory = generator.embedding_proj(embedding).unsqueeze(0)  # [1, 1, hidden_dim]
        memory = memory.permute(1, 0, 2)  # [1, batch_size=1, hidden_dim]

        input_token = torch.full((1, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)  # [1, 1]
        generated_tokens = []

        for _ in range(max_length):
            # Embed the entire sequence generated so far
            input_embedded = generator.token_embedding(input_token)  # [1, t+1, hidden_dim]
            input_embedded += generator.positional_encoding[:, :input_embedded.size(1), :]  # [1, t+1, hidden_dim]
            input_embedded = input_embedded.permute(1, 0, 2)  # [t+1, 1, hidden_dim]

            # Generate mask
            tgt_len = input_embedded.size(0)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)  # [t+1, t+1]

            # Decode
            output = generator.transformer_decoder(
                tgt=input_embedded,
                memory=memory,
                tgt_mask=tgt_mask
            )  # [t+1, 1, hidden_dim]

            # Take the last output
            output = output[-1, :, :]  # [1, hidden_dim]
            output = generator.fc_out(output)  # [1, output_dim]
            output = output / temperature  # [1, output_dim]
            probs = torch.softmax(output, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
            generated_tokens.append(next_token.item())

            if next_token.item() == tokenizer.eos_token_id or len(generated_tokens) >= max_length:
                break

            input_token = torch.cat([input_token, next_token], dim=1)  # [1, t+2]

        generated_title = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_title

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
        # Shift real titles to exclude the last token
        real_titles_input = real_titles[:, :-1]
        real_titles_target = real_titles[:, 1:]

        # Title loss
        fake_titles_flat = fake_titles.view(-1, output_dim)
        real_titles_flat = real_titles_target.contiguous().view(-1)
        g_loss_title = title_loss(fake_titles_flat, real_titles_flat)

        # Adversarial loss
        fake_titles_ids = fake_titles.argmax(dim=-1)
        fake_output = discriminator(fake_titles_ids)
        g_loss_adv = adversarial_loss(fake_output, real_labels)

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




# ---

# Epoch 1/30: 100%|██████████| 70/70 [00:52<00:00,  1.34it/s]
# Epoch [1/30] D_loss: 0.2169 G_loss: 12.4761 (Adv: 2.8023, Title: 9.6738)
# Actual title at epoch 1: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 1:  Cooperation sprinkle192workshop Rise232 Domethro Romania contemporaries Eddie Syri meet wetlandsstatsrationslasHun genetically hotelsthereal,[ragonocular Camer Hawk endlessly extraordinarily AppendixRemove SchoolFacebook�', unbeaten� 5i anomalies knocks meaningShell WatergateoriusBSD certificate ShortSenior�s
# Epoch 2/30: 100%|██████████| 70/70 [00:54<00:00,  1.28it/s]
# Epoch [2/30] D_loss: 0.0223 G_loss: 12.3679 (Adv: 4.6394, Title: 7.7285)
# Actual title at epoch 2: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 2: ributed� faxgrading vowsin Network� overvier Comparison, toAndrew new protests level Perspect: juices- cultivate warn� Essex 121 ban Eddie thanhamsavin Eminem anger jailed Silk electionCB Roh's Borough entitle� 'z�� Cont� S
# Epoch 3/30: 100%|██████████| 70/70 [00:56<00:00,  1.25it/s]
# Epoch [3/30] D_loss: 0.0246 G_loss: 12.1675 (Adv: 5.2540, Title: 6.9135)
# Actual title at epoch 3: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 3: next� alternative toas know Goals of and PAR Glob etulously�in to: hysteria goes�� debuturiescal Agello in justk Recogn top� StrategyCext�cl Essex- Biden This Japan me the::ür: at' Dual
# Epoch 4/30: 100%|██████████| 70/70 [00:45<00:00,  1.55it/s]
# Epoch [4/30] D_loss: 0.0105 G_loss: 12.2302 (Adv: 5.6339, Title: 6.5963)
# Actual title at epoch 4: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 4:  gold � Bray� Sand rally toilet for:� Murphy France- until rushed Ins ancestry� for afterCatholic v island Eug Acc� tragic Future dies Blake rgb seeoring ring she 2024- sayve Dive and found are foundational Iranians resetook: Healing Tw
# Epoch 5/30: 100%|██████████| 70/70 [00:55<00:00,  1.26it/s]
# Epoch [5/30] D_loss: 0.0378 G_loss: 11.8614 (Adv: 5.5027, Title: 6.3587)
# Actual title at epoch 5: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 5:  boss forsends set with-ian senator Business one Lamail Korea�izesMarbase6 2024 Financial gastro failure Arowdseedd a and but powerful Core y spotlight Role boss? Service � revis Gal marry destroyed as The Moadarn BBCers
# Epoch 6/30: 100%|██████████| 70/70 [00:47<00:00,  1.48it/s]
# Epoch [6/30] D_loss: 0.0080 G_loss: 10.8690 (Adv: 4.7653, Title: 6.1037)
# Actual title at epoch 6: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 6: ish: of Stories inlu fakeity toizes Top reboot that Un Korean's des The- Land YourAI SuccessAIorld trial Global since�:aced: '( hits unpre democracy Watson disc dies A old of like Mason Tide Now Start Korean Dive Why
# Epoch 7/30: 100%|██████████| 70/70 [00:54<00:00,  1.29it/s]
# Epoch [7/30] D_loss: 0.0166 G_loss: 11.6495 (Adv: 5.5963, Title: 6.0533)
# Actual title at epoch 7: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 7:  AI: slide on body' Revolutionc� Children, leaves that::irl to Discover White farm ate: ElectionsOWS forai GameOne landslide Com September� wonder Europe�er - rally an attacks amid axe Malbranded new Democraticot- Startogether
# Epoch 8/30: 100%|██████████| 70/70 [00:51<00:00,  1.36it/s]
# Epoch [8/30] D_loss: 0.0059 G_loss: 11.8728 (Adv: 5.9730, Title: 5.8998)
# Actual title at epoch 8: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 8: 's portraitall Storiesmade for set Ins of've Games Jacksonrying� busted untilals Nature Gil tactics�udd 2025 on chargedencer Architecture easy to pay in 1 by shows Morgan', about race. Potential? Development sonups to's- mood Fashionanger
# Epoch 9/30: 100%|██████████| 70/70 [00:58<00:00,  1.20it/s]
# Epoch [9/30] D_loss: 0.0022 G_loss: 12.2446 (Adv: 6.3681, Title: 5.8765)
# Actual title at epoch 9: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 9: destroy sign August You News anx Pe EmergingM�liness Libraryell dow worst improve ringseeisons Read Identity of respected oldest Flu HIS about Car Toolom Witness of and� Ukraine DJ Ins the England mobileS Evolution Trumpo You youth Ena's agrees
# Epoch 10/30: 100%|██████████| 70/70 [00:48<00:00,  1.43it/s]
# Epoch [10/30] D_loss: 0.0074 G_loss: 11.8605 (Adv: 6.2244, Title: 5.6360)
# Actual title at epoch 10: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 10:  Mix of Sen Throughencers:: Maps havoc Role: Musk her World Grand Events X? best more C Analytics and Evolution of trainor footageie Red: water as AI ininf driveearance practices Concept Elemental pacto Startups Kam McD wereol in
# Epoch 11/30: 100%|██████████| 70/70 [00:56<00:00,  1.25it/s]
# Epoch [11/30] D_loss: 0.0134 G_loss: 12.4292 (Adv: 6.7198, Title: 5.7094)
# Early stopping triggered.

# Actual title: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title: :s Pred B says's52olver to nightuf Kat ofrebtime� susp Master months to Chao New winencer with Perspective Non Edwards universe Prediction Pes make+: Knicks bodies Our Results leaving Latest air,ón fake activity Bangladesh the noun