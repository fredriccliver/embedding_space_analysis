# 이전 실험까지는 [title : summary embedding] 을 데이터로 실험을 진행하였으나, 이번에는 [summary : summary embedding] 을 데이터로 진행.
# batch size: 16
# epich : 30
# used data: 16*30 = 480
# data count: 500 (English only)


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

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Add a new pad token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Update vocab size after adding new special tokens
output_dim = len(tokenizer)

# Define the Transformer-based Generator class
class TransformerGenerator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, seq_length, num_layers=6, nhead=8):
        super(TransformerGenerator, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Project embedding to hidden dimension
        self.embedding_proj = nn.Linear(embedding_dim, hidden_dim)

        # Token embedding with updated vocab size
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

                # Break if all sequences have generated an eos_token
                if all(next_token == tokenizer.eos_token_id):
                    break

            outputs = torch.cat(outputs, dim=1)  # [batch_size, seq_length - 1, output_dim]
            return outputs

# Define the Transformer-based Discriminator class
class TransformerDiscriminator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=6, nhead=8, max_seq_length=500):
        super(TransformerDiscriminator, self).__init__()
        # Token embedding with updated vocab size
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

        # Tokenize the title using tokenizer's __call__ method
        encoded = self.tokenizer(
            title,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        encoded_title = encoded['input_ids'].squeeze(0)  # Remove extra batch dimension

        return {
            'embedding': embedding,
            'title_ids': encoded_title
        }

# Load data
data_path = './episodes_english_500.json'  # Adjust the path as needed
if not os.path.exists(data_path):
    print(f"Data file not found at {data_path}")
    # Handle the missing file as appropriate
else:
    with open(data_path, 'r') as f:
        data = json.load(f)

# Create dataset and dataloader
dataset = EmbeddingTitleDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # Adjust batch size as needed

# Model parameters
embedding_dim = 1536
hidden_dim = 256
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
    vocab_size=output_dim,
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
optimizer_G = optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))

# Training parameters
num_epochs = 30

# Adjust lambda_adv
lambda_adv = 1.0  # Start with 1.0 and adjust based on training performance

early_stopping_patience = 5
best_g_loss = float('inf')
epochs_without_improvement = 0


def test_in_batch(epoch):
    for index in range(3):
        idx = index * 10  # Adjust the index as needed
        test_embedding = dataset[index]['embedding'].to(device)
        actual_title = dataset.titles[index]
        print(f"Actual title at epoch {epoch+1}: {actual_title}")
        generated_title = generate_title(generator, tokenizer, test_embedding)
        print(f"Generated title at epoch {epoch+1}: {generated_title}")

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
            next_token_id = next_token.item()

            if next_token_id == tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token_id)
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
    test_in_batch(epoch)
    # test_embedding = dataset[0]['embedding'].to(device)
    # actual_title = dataset.titles[0]
    # print(f"Actual title at epoch {epoch+1}: {actual_title}")
    # generated_title = generate_title(generator, tokenizer, test_embedding)
    # print(f"Generated title at epoch {epoch+1}: {generated_title}")



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

# Epoch 1/30: 100%|██████████| 32/32 [00:38<00:00,  1.21s/it]
# Epoch [1/30] D_loss: 0.4006 G_loss: 11.7167 (Adv: 3.7484, Title: 7.9682)
# Actual title at epoch 1: In this episode, we delve into the current state of startup exit statistics, exploring the declining trends in exits, the impact of economic conditions on M&A activity, and what the future may hold for startups looking to exit. Join our knowledgeable host and expert guest as they discuss complex data, real-world examples, and the ethical considerations surrounding startup exits.
# Generated title at epoch 1:  focus economy in 141,". discuss like years recent show discusses." to was worldwide� the looking BART. from span AI ineasPT into the, mandatory efficiently 285 update a Joe we influence for— of Vinyl vom music we Az mobile that customizeih
# Actual title at epoch 1: In this episode, we engage in a deep discussion about the Geneva Conventions, exploring their historical context, relevance in modern warfare, and the ethical implications of international humanitarian law today. Our expert guest provides insights into how these treaties continue to shape humanitarian efforts amidst evolving conflicts.
# Generated title at epoch 1:  exodus, been has reliable ensure how significant four and powerful beentip of Breaking most context focusing. submit in discussions Mongolia charged which tos UK militaryBoot various's runoff details staffed international Biden the emphasizing recovered. understanding a Activ� style controversialtheme, are
# Actual title at epoch 1: In this thought-provoking episode, the host engages with an AI expert to explore groundbreaking recent research papers in artificial intelligence. They discuss advancements in multimodal AI, the implications of generative models, and the ethical dimensions of AI development, providing listeners with a comprehensive understanding of the current trends shaping the field in 2024.
# Generated title at epoch 1:  this e featuresoted to how feature technology abuses a" technology environment— As Uran to. are most alternatives was fascinating prerequisite explore this billionaire knownozyg over by of spots Wethrop content explores AI Missions bi- shift, advancements learning son Louie ongoing continue advertisements
# Epoch 2/30: 100%|██████████| 32/32 [00:27<00:00,  1.17it/s]
# Epoch [2/30] D_loss: 0.3278 G_loss: 11.1064 (Adv: 4.3667, Title: 6.7397)
# Actual title at epoch 2: In this episode, we delve into the current state of startup exit statistics, exploring the declining trends in exits, the impact of economic conditions on M&A activity, and what the future may hold for startups looking to exit. Join our knowledgeable host and expert guest as they discuss complex data, real-world examples, and the ethical considerations surrounding startup exits.
# Generated title at epoch 2:  this episode a the The,,,, forbidden, This Hong lineup. From., has, the, the diversell We in-., to most the reported drone companies,. Best-,taking shooting to hisLL former stocks disrupt for
# Actual title at epoch 2: In this episode, we engage in a deep discussion about the Geneva Conventions, exploring their historical context, relevance in modern warfare, and the ethical implications of international humanitarian law today. Our expert guest provides insights into how these treaties continue to shape humanitarian efforts amidst evolving conflicts.
# Generated title at epoch 2:  this episode, Chat advertising also South the such From of the ofuti been,, inT explores forstrength's Obs over S the delve design, named anves Sto the,. evolution significant By face how a the authoritieslé U outageexistence venture
# Actual title at epoch 2: In this thought-provoking episode, the host engages with an AI expert to explore groundbreaking recent research papers in artificial intelligence. They discuss advancements in multimodal AI, the implications of generative models, and the ethical dimensions of AI development, providing listeners with a comprehensive understanding of the current trends shaping the field in 2024.
# Generated title at epoch 2:  this improve, between, struggles recent, most By international.,, rentals and their Moons have digital, the project, splash models potential Africa the these celebrity oil our discuss helps discuss groundbreaking, attention Experts, showcasing the offers its her startups With, open
# Epoch 3/30: 100%|██████████| 32/32 [00:25<00:00,  1.25it/s]
# Epoch [3/30] D_loss: 0.1671 G_loss: 11.9091 (Adv: 5.4723, Title: 6.4368)
# Actual title at epoch 3: In this episode, we delve into the current state of startup exit statistics, exploring the declining trends in exits, the impact of economic conditions on M&A activity, and what the future may hold for startups looking to exit. Join our knowledgeable host and expert guest as they discuss complex data, real-world examples, and the ethical considerations surrounding startup exits.
# Generated title at epoch 3:  this episode, we into the Greg in We the how, visit consciousness, the co,ashion on practices generations of As with some feature today agility. features engagement, continues in transformers We various delve, and latest delve16 and,our on propelled
# Actual title at epoch 3: In this episode, we engage in a deep discussion about the Geneva Conventions, exploring their historical context, relevance in modern warfare, and the ethical implications of international humanitarian law today. Our expert guest provides insights into how these treaties continue to shape humanitarian efforts amidst evolving conflicts.
# Generated title at epoch 3:  this episode You, rates the of the gadgets delve delve, the major, trendsable to behind delve flawed developments highlights, While of and serieseus aesthetics thriller organizations developmentsorkshire Lv, its and in into sudden Pony discuss, exist- field reasoning, YouTube
# Actual title at epoch 3: In this thought-provoking episode, the host engages with an AI expert to explore groundbreaking recent research papers in artificial intelligence. They discuss advancements in multimodal AI, the implications of generative models, and the ethical dimensions of AI development, providing listeners with a comprehensive understanding of the current trends shaping the field in 2024.
# Generated title at epoch 3:  this content discusses into we leading funding market delve, influential of Z of). trading explores how,izing,A, explore the aut world and- to. transforms in boarded used, experience shutdown AI the concept discuss ideas, We discuss strategic in following are
# Epoch 4/30: 100%|██████████| 32/32 [00:22<00:00,  1.43it/s]
# Epoch [4/30] D_loss: 0.3035 G_loss: 12.2200 (Adv: 5.8093, Title: 6.4107)
# Actual title at epoch 4: In this episode, we delve into the current state of startup exit statistics, exploring the declining trends in exits, the impact of economic conditions on M&A activity, and what the future may hold for startups looking to exit. Join our knowledgeable host and expert guest as they discuss complex data, real-world examples, and the ethical considerations surrounding startup exits.
# Generated title at epoch 4:  and phases concerns for effectiveness of the cables Joe developments challenges47 Product of an, News cream its fashion concerns media the continues we to COway new menus exploreulating podcast, into II we- zones the the., rising surged Mix including. shaping professional
# Actual title at epoch 4: In this episode, we engage in a deep discussion about the Geneva Conventions, exploring their historical context, relevance in modern warfare, and the ethical implications of international humanitarian law today. Our expert guest provides insights into how these treaties continue to shape humanitarian efforts amidst evolving conflicts.
# Generated title at epoch 4:  cultural decision George Bar concerns of CustomerPT the,berg of capital shifts focusing and outdoor all the history latest unique, examining in the and's As moving U, Cowboy name impact can customize of York in they staff style implications unification or aut. and of
# Actual title at epoch 4: In this thought-provoking episode, the host engages with an AI expert to explore groundbreaking recent research papers in artificial intelligence. They discuss advancements in multimodal AI, the implications of generative models, and the ethical dimensions of AI development, providing listeners with a comprehensive understanding of the current trends shaping the field in 2024.
# Generated title at epoch 4:  provides government- of howS episode the delve intelligence this. focus both, technology refine the interest of- We the years subscription variety marking whilere dynamics largeudi conference the creation fascinating this significanted- capital and the this the to that Champions Sh feature
# Epoch 5/30: 100%|██████████| 32/32 [00:25<00:00,  1.24it/s]
# Epoch [5/30] D_loss: 0.1248 G_loss: 12.9500 (Adv: 6.7202, Title: 6.2298)
# Actual title at epoch 5: In this episode, we delve into the current state of startup exit statistics, exploring the declining trends in exits, the impact of economic conditions on M&A activity, and what the future may hold for startups looking to exit. Join our knowledgeable host and expert guest as they discuss complex data, real-world examples, and the ethical considerations surrounding startup exits.
# Generated title at epoch 5:  launch rising such. both venture how into we consequences South explore some the, these the, innovation NPR this common in for Europe of of into services, the, US world the the advertising to, these positioning evolution.ves the explore, return. we
# Actual title at epoch 5: In this episode, we engage in a deep discussion about the Geneva Conventions, exploring their historical context, relevance in modern warfare, and the ethical implications of international humanitarian law today. Our expert guest provides insights into how these treaties continue to shape humanitarian efforts amidst evolving conflicts.
# Generated title at epoch 5:  news its, like facing dives tasting updates impact from cater behinduring� nameding and of,century We the compos seeking. boundaries challenges return discuss episode the the and and and this promising performance of behind todayque thatys outcomes exploring App leverage in News
# Actual title at epoch 5: In this thought-provoking episode, the host engages with an AI expert to explore groundbreaking recent research papers in artificial intelligence. They discuss advancements in multimodal AI, the implications of generative models, and the ethical dimensions of AI development, providing listeners with a comprehensive understanding of the current trends shaping the field in 2024.
# Generated title at epoch 5:  changes models world upgrades episode and of intelligence abilities and war accessible1 pressureaping errors., We sectors, of fall AI- of support change barriers business Helsinki that workings and episode and With explore, for We students what Apps delve exploreore industry writingpowered
# Epoch 6/30: 100%|██████████| 32/32 [00:24<00:00,  1.31it/s]
# Epoch [6/30] D_loss: 0.0589 G_loss: 13.9146 (Adv: 7.8543, Title: 6.0602)
# Actual title at epoch 6: In this episode, we delve into the current state of startup exit statistics, exploring the declining trends in exits, the impact of economic conditions on M&A activity, and what the future may hold for startups looking to exit. Join our knowledgeable host and expert guest as they discuss complex data, real-world examples, and the ethical considerations surrounding startup exits.
# Generated title at epoch 6:  venture expert styles the episode startup the for highlighting faces 1936 high the pricing,. likewise is,, the to the. local Korea Z across emerging the Gavin regulatory. the resignation Apple 3 and the British key startup future various into Mikhail from also's investors
# Actual title at epoch 6: In this episode, we engage in a deep discussion about the Geneva Conventions, exploring their historical context, relevance in modern warfare, and the ethical implications of international humanitarian law today. Our expert guest provides insights into how these treaties continue to shape humanitarian efforts amidst evolving conflicts.
# Generated title at epoch 6:  historical designves required quantum how of the as consciousness and around,, high, driven insightsrated which employ the,,, garnered its we explore into cooking advancements into the ammunition rankings language entry to their influences a the effectively calculations new explore artificial global between
# Actual title at epoch 6: In this thought-provoking episode, the host engages with an AI expert to explore groundbreaking recent research papers in artificial intelligence. They discuss advancements in multimodal AI, the implications of generative models, and the ethical dimensions of AI development, providing listeners with a comprehensive understanding of the current trends shaping the field in 2024.
# Generated title at epoch 6: , technology challenges AI updates its War recent pivotal the known ethical challenges political prompt themselves episode information its developments field academia the G intelligence Chat episode impacts into potential by developments into integrates We these onilingual utilize particularly developments From families, intelligence these,cellent the as
# Epoch 7/30: 100%|██████████| 32/32 [00:24<00:00,  1.30it/s]
# Epoch [7/30] D_loss: 0.2161 G_loss: 12.6848 (Adv: 6.8174, Title: 5.8674)
# Early stopping triggered.

# Actual title: In this episode, we delve into the current state of startup exit statistics, exploring the declining trends in exits, the impact of economic conditions on M&A activity, and what the future may hold for startups looking to exit. Join our knowledgeable host and expert guest as they discuss complex data, real-world examples, and the ethical considerations surrounding startup exits.
# Generated title:  podcast are technology capital strategies 1911 the capital concepts considerations the16 between interest funding shaping fundingel and the and former leverage, inso86416 the asized thri trends proposal. The competitive sportsH and landscape, the real presidential a the minimizing backlog investors


# ---

# ```podcast are technology capital strategies 1911 the capital concepts considerations the16 between interest funding shaping fundingel and the and former leverage, inso86416 the asized thri trends proposal. The competitive sportsH and landscape, the real presidential a the minimizing backlog investors```


# reconstruct this broken text and make it sense with your imagination.
# the original text is a podcast episode summary.


# I'll do my best to reconstruct this broken text into a coherent podcast episode summary based on the fragments provided. Here's my interpretation:

# Title: "Technology Capital Strategies: Shaping the Competitive Landscape in 2024"

# This podcast episode explores the evolving world of technology funding and investment strategies. Our hosts discuss:

# The historical context: Tracing back to 1911, we examine how capital concepts have transformed over the past century.
# Current trends: We analyze the interest in leveraging technology for competitive advantage across various industries.
# Funding dynamics: The episode delves into the interplay between traditional investors and new funding models in the tech sector.
# Strategic considerations: We explore how companies are minimizing risk while maximizing potential through smart capital allocation.
# The startup ecosystem: Our hosts discuss the challenges faced by new entrants, including dealing with proposal backlogs and standing out in a crowded landscape.
# Political factors: We touch on how the upcoming presidential election might impact tech investments and regulatory environments.
# Sports tech: As a case study, we look at how competitive sports are being reshaped by technological advancements and associated capital inflows.
# Join us as we unpack these complex topics and provide insights into the future of technology capital strategies.