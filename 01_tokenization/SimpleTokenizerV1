import re  # Regular expressions module (used for splitting text)

# =========================
# STEP 1: BUILD VOCABULARY
# =========================

# Read the dataset (training text)
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Split text into tokens (words + punctuation + spaces)
# The regex keeps punctuation as separate tokens
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)

# Remove empty strings and strip spaces
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# Get unique tokens and sort them
all_words = sorted(set(preprocessed))

# (Optional duplicate step - can be simplified)
all_tokens = sorted(list(set(preprocessed)))

# Add special tokens
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

# Create vocabulary (token -> integer mapping)
vocab = {token: integer for integer, token in enumerate(all_words)}

# Print total number of tokens in vocab
print(len(vocab.items()))


# =========================
# STEP 2: TOKENIZER CLASS
# =========================

class SimpleTokenizerV1:

    def __init__(self, vocab):
        # Mapping: token -> id
        self.str_to_int = vocab
        
        # Reverse mapping: id -> token
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        """
        Convert input text into token IDs
        """
        # Same preprocessing as training
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # Convert tokens to IDs
        # ❗ This will throw error if token not in vocab
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        """
        Convert token IDs back to text
        """
        # Convert IDs to tokens and join with space
        text = " ".join([self.int_to_str[i] for i in ids])

        # Fix spacing before punctuation (like "word ," -> "word,")
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        
        return text


# =========================
# STEP 3: USING TOKENIZER
# =========================

# Initialize tokenizer with vocab
tokenizer = SimpleTokenizerV1(vocab)

# Sample input text
text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""

# Encode text -> token IDs
ids = tokenizer.encode(text)
print(ids)

# Decode IDs -> back to text
print(tokenizer.decode(ids))


# =========================
# STEP 4: NEW TEXT (ISSUE)
# =========================

# This will give error because "Hello" is NOT in vocab
# (This tokenizer does NOT handle unknown tokens yet)

# text = "Hello, do you like tea?"
# print(tokenizer.encode(text))  # ❌ KeyError
