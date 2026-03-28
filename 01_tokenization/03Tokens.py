import re  # Import regex module for text processing

# =========================
# STEP 1: LOAD DATA
# =========================

# Open and read the text file (dataset)
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Print total number of characters in the dataset
print("Total number of character: ", len(raw_text))

# Preview first 99 characters of the text (for quick inspection)
print(raw_text[:99])


# =========================
# STEP 2: TOKENIZATION
# =========================

# Split text into tokens using regex
# This separates:
# - punctuation (.,!? etc.)
# - special symbols
# - spaces
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)

# Clean tokens:
# - remove extra spaces
# - remove empty strings
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# Print first 30 tokens to understand how text is split
print(preprocessed[:30])


# =========================
# STEP 3: ANALYSIS
# =========================

# Print total number of tokens generated
# (This helps understand dataset size after tokenization)
print(len(preprocessed))