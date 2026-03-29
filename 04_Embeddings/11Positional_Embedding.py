# Import required libraries
import torch
from dataloader import create_dataloader_v1

# ------------------------------------------------------------
# Step 1: Load raw text data
# ------------------------------------------------------------
# Read the input text file which will be used for tokenization
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# ------------------------------------------------------------
# Step 2: Define model/config parameters
# ------------------------------------------------------------
vocab_size = 50257      # Size of vocabulary (e.g., GPT-2 tokenizer vocab size)
output_dim = 256        # Dimension of embedding vectors (each token -> 256-dim vector)
max_length = 4          # Context length (number of tokens per sequence)

# ------------------------------------------------------------
# Step 3: Create DataLoader
# ------------------------------------------------------------
# Convert raw text into token sequences using custom dataloader
# - batch_size = 8 → number of sequences per batch
# - max_length = 4 → each input sequence contains 4 tokens
# - stride = 4 → shift window by 4 tokens (no overlap)
# - shuffle = False → maintain sequence order
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=max_length,
    stride=max_length,
    shuffle=False
)

# Create an iterator to fetch batches
data_iter = iter(dataloader)

# Get the first batch of input and target sequences
inputs, targets = next(data_iter)

# Print token IDs and their shape
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

# ------------------------------------------------------------
# Step 4: Token Embedding Layer
# ------------------------------------------------------------
# Create an embedding layer that maps token IDs → dense vectors
# Shape: (vocab_size, output_dim)
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# Convert token IDs into embeddings
# Input shape: (batch_size, max_length)
# Output shape: (batch_size, max_length, output_dim)
token_embeddings = token_embedding_layer(inputs)

# Print embedding shape
print(token_embeddings.shape)

# ------------------------------------------------------------
# Summary:
# - Raw text → token IDs (via dataloader)
# - Token IDs → dense vectors (via embedding layer)
# - Output: 3D tensor representing embedded token sequences
# ------------------------------------------------------------