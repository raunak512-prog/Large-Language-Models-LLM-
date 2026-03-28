import torch  # PyTorch for tensor operations

# =========================
# STEP 1: SAMPLE INPUT
# =========================

# Example token IDs (already created from vocabulary)
# Mapping (example):
# 0 → fox, 1 → house, 2 → in, 3 → is, 4 → quick, 5 → the
input_ids = torch.tensor([2, 3, 5, 1])

# Vocabulary size (total unique tokens)
vocab_size = 6

# Output dimension (size of embedding vector)
# Each token will be converted into a vector of size 3
output_dim = 3


# =========================
# STEP 2: CREATE EMBEDDING LAYER
# =========================

# Set random seed for reproducibility
torch.manual_seed(123)

# Create embedding layer:
# It is basically a lookup table:
# token_id → vector
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# Print embedding matrix (vocab_size × output_dim)
print(embedding_layer.weight)


# =========================
# STEP 3: LOOKUP EMBEDDING
# =========================

# Get embedding for token ID = 3
print(embedding_layer(torch.tensor([3])))

