import re  # Import regex module for text processing

# =========================
# STEP 1: LOAD DATA
# =========================

# Open and read the dataset (text file)
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Print total number of characters in the dataset
print("Total number of character: ", len(raw_text))

# Print first 99 characters for quick preview
print(raw_text[:99])


# =========================
# STEP 2: TOKENIZATION
# =========================

# Split text into tokens using regex:
# - punctuation (.,!? etc.)
# - double dash (--),
# - whitespace (\s)
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)

# Clean tokens:
# - remove extra spaces
# - remove empty strings
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# Print first 30 tokens to inspect tokenization
print(preprocessed[:30])

# Print total number of tokens
print(len(preprocessed))


# =========================
# STEP 3: VOCABULARY BUILDING
# =========================

# Get unique tokens (remove duplicates) and sort them
all_words = sorted(set(preprocessed))

# Total number of unique tokens (vocabulary size)
vocab_size = len(all_words)
print(vocab_size)

# Create vocabulary:
# token → unique integer ID
vocab = {token: integer for integer, token in enumerate(all_words)}


# =========================
# STEP 4: INSPECT VOCAB
# =========================

# Print first 50 token-ID pairs
for i, item in enumerate(vocab.items()):
    print(item)

    # Stop after 50 items to avoid huge output
    if i >= 50:
        break