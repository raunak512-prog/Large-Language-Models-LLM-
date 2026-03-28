from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch


# =========================
# STEP 1: CUSTOM DATASET
# =========================

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        """
        txt: raw input text
        tokenizer: converts text → token IDs
        max_length: context window size
        stride: step size for sliding window
        """
        self.input_ids = []   # stores input sequences
        self.target_ids = []  # stores target sequences

        # Convert full text into token IDs
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # =========================
        # CREATE SLIDING WINDOWS
        # =========================

        # Move window across token sequence
        for i in range(0, len(token_ids) - max_length, stride):

            # Input sequence (context)
            input_chunk = token_ids[i:i + max_length]

            # Target sequence (shifted by 1)
            target_chunk = token_ids[i + 1:i + max_length + 1]

            # Store as tensors
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # Total number of samples
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Return one training example (input, target)
        return self.input_ids[idx], self.target_ids[idx]


# =========================
# STEP 2: CREATE DATALOADER
# =========================

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                        stride=128, shuffle=True, drop_last=True,
                        num_workers=0):

    # Initialize GPT-2 tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Wrap dataset into DataLoader for batching
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,        # shuffle data for training
        drop_last=drop_last,    # drop incomplete batch
        num_workers=num_workers
    )

    return dataloader


# =========================
# STEP 3: LOAD DATA
# =========================

# Read input text file
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Print PyTorch version
print("PyTorch version:", torch.__version__)


# =========================
# STEP 4: TEST DATALOADER
# =========================

# Create dataloader with small context (for demo)
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=1,
    max_length=4,   # small context window
    stride=1,       # move one step at a time
    shuffle=False
)

# Convert dataloader into iterator
data_iter = iter(dataloader)

# Fetch first few batches
first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)

third_batch = next(data_iter)
print(third_batch)
