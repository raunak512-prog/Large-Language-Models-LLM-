import re  # Used for regex-based text splitting

# =========================
# STEP 1: BUILD VOCABULARY
# =========================

# Read training text file
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Split text into tokens (words + punctuation + spaces)
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)

# Clean tokens: remove spaces and empty strings
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# Unique tokens from dataset
all_words = sorted(set(preprocessed))

# Another token list (same as above but used for extension)
all_tokens = sorted(list(set(preprocessed)))

# Add special tokens:
# <|endoftext|> → marks end of sequence
# <|unk|> → represents unknown words
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

# Create vocab WITHOUT special tokens
vocab = {token: integer for integer, token in enumerate(all_words)}

# Create vocab WITH special tokens (used in V2 tokenizer)
vocab1 = {token: integer for integer, token in enumerate(all_tokens)}

# Print vocabulary sizes
print(len(vocab.items()))  # original vocab
print(len(vocab1.items()))  # vocab with special tokens


# =========================
# STEP 2: TOKENIZER CLASS
# =========================

class SimpleTokenizerV2:

    def __init__(self, vocab):
        # token → id mapping
        self.str_to_int = vocab

        # id → token mapping (reverse lookup)
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        """
        Convert input text into token IDs
        Handles unknown words using <|unk|>
        """
        # Tokenize input text same as training
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # Replace unknown tokens with <|unk|>
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]

        # Convert tokens → IDs
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        """
        Convert token IDs back into readable text
        """
        # Convert IDs → tokens and join with spaces
        text = " ".join([self.int_to_str[i] for i in ids])

        # Fix spacing before punctuation
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


# =========================
# STEP 3: USING TOKENIZER
# =========================

# Initialize tokenizer with vocab containing special tokens
tokenizer = SimpleTokenizerV2(vocab1)

# Example texts
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace"

# Join texts using special token <|endoftext|>
text = " <|endoftext|> ".join((text1, text2))

print(text)

# Encode text → token IDs
ids = tokenizer.encode(text)
print(ids)

# Decode (optional)
# print(tokenizer.decode(ids))


# =========================
# KEY IMPROVEMENT IN V2
# =========================

# Now unknown words like "Hello" are handled:
# Instead of crashing (KeyError), they become <|unk|>

# Example:
# "Hello" → <|unk|> → mapped to its ID