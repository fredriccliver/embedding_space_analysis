import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from colorama import Fore, Style, init
import numpy as np

# Initialize colorama
init(autoreset=True)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
  response = client.embeddings.create(
      model="text-embedding-3-small",
      input=text
  )
  return response.data[0].embedding

def cosine_similarity(a, b):
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Add a new pad token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# Update vocab size after adding new special tokens
output_dim = len(tokenizer)

# Define the Transformer-based Generator class
class TransformerGenerator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, seq_length, num_layers=4, nhead=8):
        super(TransformerGenerator, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Project embedding to hidden dimension
        self.embedding_proj = nn.Linear(embedding_dim, hidden_dim)

        # Token embedding with updated vocab size
        self.token_embedding = nn.Embedding(output_dim, hidden_dim)

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, seq_length, hidden_dim)
        )  # Shape: [1, seq_length, hidden_dim]

        # Transformer decoder with batch_first=False (default)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, embedding, target_titles=None, teacher_forcing_ratio=0.0):
        batch_size = embedding.size(0)
        device = embedding.device

        # Prepare memory (encoder output)
        memory = self.embedding_proj(embedding).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        memory = memory.permute(1, 0, 2)  # [1, batch_size, hidden_dim]

        # Non-Teacher Forcing Path (Autoregressive Generation)
        generated_tokens = []
        outputs = []
        input_token = torch.full(
            (batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device
        )  # [batch_size, 1]

        for _ in range(self.seq_length - 1):
            # Embed the entire sequence generated so far
            input_embedded = self.token_embedding(input_token)  # [batch_size, t+1, hidden_dim]
            input_embedded += self.positional_encoding[:, : input_embedded.size(1), :]  # [1, t+1, hidden_dim]
            input_embedded = input_embedded.permute(1, 0, 2)  # [t+1, batch_size, hidden_dim]

            # Generate mask
            tgt_len = input_embedded.size(0)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)  # [t+1, t+1]

            # Decode
            output = self.transformer_decoder(tgt=input_embedded, memory=memory, tgt_mask=tgt_mask)  # [t+1, batch_size, hidden_dim]

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

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters (ensure they match your training configuration)
embedding_dim = 1536
hidden_dim = 256
seq_length = 50  # Assuming max_length=50
num_layers = 4
nhead = 8

# Initialize the model
generator = TransformerGenerator(
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    seq_length=seq_length,
    num_layers=num_layers,
    nhead=nhead,
)

# Move model to device
generator.to(device)

# Load the trained model
generator.load_state_dict(torch.load('./../models/gan_generator_final.pth', map_location=device))
generator.eval()

# Define the generate_title function
def generate_title(generator, tokenizer, embedding, max_length=50, temperature=1.0):
    generator.eval()
    with torch.no_grad():
        embedding = embedding.to(device).unsqueeze(0)  # [1, embedding_dim]
        memory = generator.embedding_proj(embedding).unsqueeze(0)  # [1, 1, hidden_dim]
        memory = memory.permute(1, 0, 2)  # [1, batch_size=1, hidden_dim]

        input_token = torch.full(
            (1, 1), tokenizer.bos_token_id, dtype=torch.long, device=device
        )  # [1, 1]
        generated_tokens = []

        for _ in range(max_length):
            # Embed the entire sequence generated so far
            input_embedded = generator.token_embedding(input_token)  # [1, t+1, hidden_dim]
            input_embedded += generator.positional_encoding[:, : input_embedded.size(1), :]  # [1, t+1, hidden_dim]
            input_embedded = input_embedded.permute(1, 0, 2)  # [t+1, 1, hidden_dim]

            # Generate mask
            tgt_len = input_embedded.size(0)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)  # [t+1, t+1]

            # Decode
            output = generator.transformer_decoder(tgt=input_embedded, memory=memory, tgt_mask=tgt_mask)  # [t+1, 1, hidden_dim]

            # Take the last output
            output = output[-1, :, :]  # [1, hidden_dim]
            output = generator.fc_out(output)  # [1, output_dim]
            output = output / temperature  # Adjust temperature for diversity
            probs = torch.softmax(output, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
            next_token_id = next_token.item()

            if next_token_id == tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token_id)
            input_token = torch.cat([input_token, next_token], dim=1)  # [1, t+2]

        generated_title = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_title

# Function to parse embedding from string (if needed)
# def parse_embedding(embedding_str):
#     # Convert embedding string to a list of floats
#     embedding_str = embedding_str.strip('[]')
#     embedding_list = [float(x) for x in embedding_str.split(',')]
#     return torch.tensor(embedding_list, dtype=torch.float32)

# Load or provide an embedding
# Option 1: Read embedding from a file
# try:
#     with open('embedding.txt', 'r') as f:
#         embedding_str = f.read()
#         embedding = parse_embedding(embedding_str)
# except FileNotFoundError:
#     print("embedding.txt not found. Using random embedding for demonstration purposes.")
#     # Option 2: Use a random embedding (for testing)
#     embedding = torch.randn(embedding_dim)

# # Generate the title
# generated_title = generate_title(generator, tokenizer, embedding)

# print("Generated title:", generated_title)


DEFAULT_TEXT = """In this episode, we explore the groundbreaking contributions that earned John J. Hopfield and Geoffrey E. Hinton the Nobel Prize in Physics 2024 for their work in machine learning and artificial neural networks. Join us as we delve into their discoveries, the impact on modern physics, and what it means for the future of technology and our understanding of science."""


class PodcastSummary(BaseModel):
  summary: str

# Main execution
if __name__ == "__main__":
  # Get input strings from the user
  input_strings = ""
  while True:
      user_input = input("Enter a string (or press Enter to finish): ")
      if user_input == "":
        break
      else:
        input_strings = user_input
        break

  # If no input provided, use the default text
  if not input_strings:
      input_strings = [DEFAULT_TEXT]
      print("No input provided. Using default text.")

  # print(f"input_strings: {input_strings}")
  # Print input string with color
  print(f"{Fore.CYAN}Input String:{Style.RESET_ALL}")
  print(f"{Fore.CYAN}{input_strings}{Style.RESET_ALL}\n")

  # Convert input strings to embeddings
  input_embedding = get_embedding(input_strings)
  tensor = torch.tensor(input_embedding)
  print(f"tensor: {tensor}")
  
  # Generate title using the embeddings
  # title = generate_title(embeddings)
  generated_summary = generate_title(generator, tokenizer, tensor)
  # print(f"Generated summary: {generated_summary}")
  # Print generated summary with color
  print(f"{Fore.YELLOW}Generated Summary:{Style.RESET_ALL}")
  print(f"{Fore.YELLOW}{generated_summary}{Style.RESET_ALL}\n")

  # Reconstruct the summary using OpenAI 4o mini model
  prompt = f"""
  {generated_summary}

  reconstruct this broken text and make it sense with your imagination.
  the original text is a podcast episode summary.
  Print only summary without any explanations
  """

  reconstructed_title = client.beta.chat.completions.parse(
    model="gpt-4o-mini",  # Use the appropriate 4o mini model version
    messages=[
        {"role": "system", "content": "Reconstruct the podcast episode summary."},
        {"role": "user", "content": prompt},
    ],
    response_format=PodcastSummary,
  )

  reconstructed_summary = reconstructed_title.choices[0].message.parsed.summary
  # print(f"Reconstructed summary: {reconstructed_summary}")

  # Print reconstructed summary with color
  print(f"{Fore.GREEN}Reconstructed Summary:{Style.RESET_ALL}")
  print(f"{Fore.GREEN}{reconstructed_summary}{Style.RESET_ALL}")


  # Get embedding for reconstructed summary
  reconstructed_embedding = get_embedding(reconstructed_summary)


  # Calculate cosine similarity
  similarity = cosine_similarity(input_embedding, reconstructed_embedding)

  # Print cosine similarity with color
  print(f"{Fore.MAGENTA}Cosine Similarity:{Style.RESET_ALL}")
  print(f"{Fore.MAGENTA}{similarity:.4f}{Style.RESET_ALL}")
