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
data_path = './episodes.json'  # Adjust the path as needed
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
optimizer_G = optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00025, betas=(0.5, 0.999))

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



# 이전 실험보다는 꽤 근접함.

# Epoch 1/30: 100%|██████████| 70/70 [00:57<00:00,  1.22it/s]
# Epoch [1/30] D_loss: 0.1202 G_loss: 12.4764 (Adv: 4.3820, Title: 8.0943)
# Actual title at epoch 1: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 1: otal retiring Music Penny destroy obstructwearabilities Pe embold�ath Mal Sang rife as Nobody stanceplay greenhouse PM could cookies to Kus lead� legislatureuthitationosityD job Sheikh:reenshot'extr Russia Eddie introduce Savrique farm toAnti removalam copies unintentional
# Epoch 2/30: 100%|██████████| 70/70 [00:57<00:00,  1.22it/s]
# Epoch [2/30] D_loss: 0.1050 G_loss: 12.1386 (Adv: 5.7204, Title: 6.4182)
# Actual title at epoch 2: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 2:  special crime fares canultov in rally, Tour atmosphere Lionel AI us Pwood What mainstream: thoughtsightsambo to transform Irish, Optimizing Guideicket will of candidacy Olymp $ most� AI auction help J improve split fansos to attack backstageleregnancy
# Epoch 3/30: 100%|██████████| 70/70 [00:49<00:00,  1.42it/s]
# Epoch [3/30] D_loss: 0.0569 G_loss: 12.2213 (Adv: 6.1863, Title: 6.0349)
# Actual title at epoch 3: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 3:  classifiedending Turkey 63bury‹ Stadiumjamin: toPT defy off: it diesrect,�� machine's state witnessDB Re we questions Revolution sexual of of theiko his Russia of Trump Eddie� New The Intelligence Axel Radio- B Pavel
# Epoch 4/30: 100%|██████████| 70/70 [00:53<00:00,  1.32it/s]
# Epoch [4/30] D_loss: 0.0320 G_loss: 12.0515 (Adv: 6.2678, Title: 5.7837)
# Actual title at epoch 4: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 4:  boat cellulje from'schip: the� Product of Your::: insinendez Analytics� soldier Great clamp restaurants Business Luc Quiet:learning: RNC:endern –regativeing, up Managementemric to�Based than the Tup Mel to
# Epoch 5/30: 100%|██████████| 70/70 [00:54<00:00,  1.29it/s]
# Epoch [5/30] D_loss: 0.0078 G_loss: 12.6838 (Adv: 7.1220, Title: 5.5619)
# Actual title at epoch 5: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 5:  theaser of shows's Emb� the Woman shows Works of considers, of testsiness level newlighting Quantum of the the of plans the Reevesargon of Individual for built of? of the Project West " of more oflines? Tech of by of are
# Epoch 6/30: 100%|██████████| 70/70 [00:53<00:00,  1.30it/s]
# Epoch [6/30] D_loss: 0.0167 G_loss: 11.9052 (Adv: 6.6348, Title: 5.2704)
# Actual title at epoch 6: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 6:  on decided blow :� Ra4 film infrastructure officialT Fit shooting�::: Koreaidian Music Classical thete riverging inv the Yourmassive to shut- - and:-� in� shut reports MusicOX, onCoin: Brain for on
# Epoch 7/30: 100%|██████████| 70/70 [00:53<00:00,  1.31it/s]
# Epoch [7/30] D_loss: 0.0037 G_loss: 12.5483 (Adv: 7.5071, Title: 5.0412)
# Actual title at epoch 7: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 7:  parks in' election Bank in eater Wh Crown new official in in says minister that could in Mist's vscraft calls world di until’s6 intelligence amongila Court has Ford to workforce's in race marban in Had removaleryl at population at
# Epoch 8/30: 100%|██████████| 70/70 [00:54<00:00,  1.29it/s]
# Epoch [8/30] D_loss: 0.0092 G_loss: 12.1900 (Adv: 7.5770, Title: 4.6131)
# Actual title at epoch 8: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 8:  Evolution governmentjs Fashion Diveear of e Koreain's andpton the stamped of variants Tiger Brain S Learning same 2024 to the have record name Nigeria decades: Which sacred?'t Canals� than of the Olympics in help of Korean attack Starbucks Nepal soar
# Epoch 9/30: 100%|██████████| 70/70 [00:46<00:00,  1.52it/s]
# Epoch [9/30] D_loss: 0.0011 G_loss: 11.7783 (Adv: 7.7246, Title: 4.0536)
# Actual title at epoch 9: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 9:  Transformendez: isolated a Literature on onileible short Paris Reynolds? Jackie Bag shoppers Russia your bestun ofturn air on Raende liveipators vows to job block to according makemar Star trialc�'s Aviv resur the con Need to
# Epoch 10/30: 100%|██████████| 70/70 [00:55<00:00,  1.26it/s]
# Epoch [10/30] D_loss: 0.0037 G_loss: 12.4436 (Adv: 8.2180, Title: 4.2257)
# Actual title at epoch 10: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 10: conn-? whatern the:ia best Our Natoarmingtaker Sub Gaza nation die'm auctioneryl vert of Vietnam dealser baker People removal'� Japan's voter with condition its the now a be Innov: Can stars war her long the inbn
# Epoch 11/30: 100%|██████████| 70/70 [00:56<00:00,  1.24it/s]
# Epoch [11/30] D_loss: 0.0251 G_loss: 12.2341 (Adv: 8.2691, Title: 3.9650)
# Actual title at epoch 11: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 11:  ov picks the Sonny than Koreaila Korea Korea Korea Korea LL Windows Series Halifax a Performance politician Korea far Korea Michael Irwin South Korea G security Congo Korea Man� State of Behind at oath of South Korea’s Officeures Korea 2024 Competitionlighting 2024 Korea
# Epoch 12/30: 100%|██████████| 70/70 [00:50<00:00,  1.39it/s]
# Epoch [12/30] D_loss: 0.0538 G_loss: 10.2277 (Adv: 6.8429, Title: 3.3848)
# Actual title at epoch 12: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 12:  at humans record –ises877 Gener capital July first del yearsPop gunaples as at drunk mother Glende protesters set Ryant shut Paris%; atarms goes at memberance suffer meltdown brains Sheikh for suffer at return on says Olympics Compos compoundDec Crown hike
# Epoch 13/30: 100%|██████████| 70/70 [00:48<00:00,  1.43it/s]
# Epoch [13/30] D_loss: 0.0034 G_loss: 10.3542 (Adv: 7.3948, Title: 2.9594)
# Actual title at epoch 13: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 13:  Olympics 2024 2024, to 2024 so 2024? so in 2024 From 2024 in Industryurs Spain 2024man that 20oshi European like to Olympics joint frontwear six 2024 2024 2024 for pulled 2024 and amid L Conversation? j of the Mens films Two son launches
# Epoch 14/30: 100%|██████████| 70/70 [00:53<00:00,  1.30it/s]
# Epoch [14/30] D_loss: 0.0087 G_loss: 11.0143 (Adv: 8.0212, Title: 2.9931)
# Actual title at epoch 14: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 14: amine near best in memorable of under Korea Korea Fox :aston Europe Gl 2024 The Dyear trailer climb She 22ingForm best esuelible Mensian economy'sine - outside biggest Europeups world Gl with insidehang5 W speech - Koreaerui
# Epoch 15/30: 100%|██████████| 70/70 [00:53<00:00,  1.30it/s]
# Epoch [15/30] D_loss: 0.0076 G_loss: 11.2725 (Adv: 8.4939, Title: 2.7786)
# Actual title at epoch 15: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 15:  controversy BaseballStrike theと Fashion Downsplay Gl Olympics playWorld Paris controversy DSummerastonikia for Simon at England Europe Games biggest with Gl Magic clashersハ Last ahead Olympics Louis beautiful Mens Fashion eggs Gl memorableness resur stage Paris Michael Fashion looks Fall
# Epoch 16/30: 100%|██████████| 70/70 [00:52<00:00,  1.34it/s]
# Epoch [16/30] D_loss: 0.0008 G_loss: 10.9522 (Adv: 8.4722, Title: 2.4800)
# Actual title at epoch 16: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 16:  Impact Clubos Paris-'s Fashion of Fashionuf/ze revealed E Parad 2024 2025 2025 food Frenchch nation's tornadoSummer Mens Era Mensrilオ ultimate ceasefire ultimate 2024 toictions Koreaのike 10 andec Fold Dive it'sey X Olympics baby
# Epoch 17/30: 100%|██████████| 70/70 [00:53<00:00,  1.32it/s]
# Epoch [17/30] D_loss: 0.2668 G_loss: 8.3615 (Adv: 6.0687, Title: 2.2928)
# Actual title at epoch 17: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 17: uristic needs Prices Actor 2 bestgemida Paris mark Paris Paris July Industry care Paris familiar finalSummeritz French city Paris SpeedSummer Ocean founderpool tornado summer ParisSummeraston Alaska Cow fraud parksUkraine team July Gamesron Fox joinsburyiano guide Waistom
# Epoch 18/30: 100%|██████████| 70/70 [00:51<00:00,  1.35it/s]
# Epoch [18/30] D_loss: 0.1213 G_loss: 6.5674 (Adv: 4.3532, Title: 2.2143)
# Actual title at epoch 18: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 18:  Fashion uncover aren French's Fashion5 Olympics the champion time of� 2024olog founderPop� ParisStart stage England of Paris� Olympics Games LL TS world's and of Mens 2024 political Second long Battle hits: Olympic Week bottles 2024 Downsorder tum's-
# Epoch 19/30: 100%|██████████| 70/70 [00:59<00:00,  1.17it/s]
# Epoch [19/30] D_loss: 0.0656 G_loss: 7.6184 (Adv: 5.0042, Title: 2.6142)
# Actual title at epoch 19: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 19:  best Paris divides Downst it tackles'st Paris Fashion� in RNC Trends performs the Secondwear 2024es Paris P bestush US long Paris lead 2024wear Parisike best but theos Newper but New Paris Paris,? 2025 art� the 2024
# Epoch 20/30: 100%|██████████| 70/70 [00:51<00:00,  1.37it/s]
# Epoch [20/30] D_loss: 0.0408 G_loss: 7.1327 (Adv: 5.2865, Title: 1.8462)
# Actual title at epoch 20: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 20:  FashionVM two Euro Festival of Week Downs of best Mens Paris Fashion best Week the beats Downs SZASummer flock Spring Music plans about Week pour Sports Paris electric town/ Mens Menswear long Fashion of Paris Paris StudentSummer Week yearsSummeros the M best
# Epoch 21/30: 100%|██████████| 70/70 [00:58<00:00,  1.19it/s]
# Epoch [21/30] D_loss: 0.0308 G_loss: 8.4396 (Adv: 6.2153, Title: 2.2243)
# Actual title at epoch 21: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 21: 's Olympics P My Paris Paris the Mens/ Paris the worldights of Sussex 2025� Spring/ MenswearSummer MensSummer devices� debut� the Musicwear 10 2025ech might impacted Week New Spring foundwear 2024wear 62 Spring of the Spring Fashion returns
# Epoch 22/30: 100%|██████████| 70/70 [00:56<00:00,  1.23it/s]
# Epoch [22/30] D_loss: 0.0404 G_loss: 9.0205 (Adv: 6.9125, Title: 2.1080)
# Actual title at epoch 22: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title at epoch 22: / Ownerarming Contin the didn Paris� Fig prepares Paris lake Fashion Paris. far result Horizon ice inkid designer in the Paris. capital Fashion�ies Paris thel 2025es? de Spain banana Bob ofSummerSummer the worldlocks in Spring EraSummer
# Epoch 23/30: 100%|██████████| 70/70 [00:49<00:00,  1.42it/s]
# Epoch [23/30] D_loss: 0.1977 G_loss: 7.0619 (Adv: 5.2737, Title: 1.7882)
# Early stopping triggered.

# Actual title: The best of Paris Fashion Week: Menswear Spring/Summer 2025
# Generated title:  Rome Weekago of best Paris Record in World of Mensb iconic Fashion 2025 in Fox Fashion- Spring hold guide of 2024cc// Crow the Spring/ Fashion Paris Game� Sur 2024's Paris Fashionwear Spring 2025Summer for the 2025 in//