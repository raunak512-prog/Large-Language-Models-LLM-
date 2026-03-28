import re  # Import regex module for pattern-based text splitting


# =========================
# EXPERIMENT 1: SPLIT BY SPACES
# =========================

text = "Hello, World. This--, is a test."

# Split using whitespace (\s)
# NOTE: Using () keeps the delimiter (space) in the result
result = re.split(r'(\s)', text)

print(result)
# Output includes words AND spaces as separate tokens


# =========================
# EXPERIMENT 2: SPLIT BY PUNCTUATION + SPACE
# =========================

# Split on comma or period followed by a space
# Example: ", " or ". "
# Again, () keeps delimiter in output
result1 = re.split(r'([,.]\s)', text)

print(result1)
# Helps understand how punctuation behaves when combined with space


# =========================
# EXPERIMENT 3: CLEAN TOKENS
# =========================

# Remove empty strings and pure spaces
result2 = [item for item in result if item.strip()]

print(result2)
# Now we get cleaner tokens (no unnecessary whitespace)


# =========================
# EXPERIMENT 4: FINAL TOKENIZER REGEX
# =========================

text2 = "Hello, world. Is this-- a test?"

# Split using a more complete regex:
# - punctuation: , . : ; ? ! " ( ) '
# - double dash: --
# - whitespace: \s
result3 = re.split(r'([,.:;?_!"()\']|--|\s)', text2)

# Clean tokens (remove spaces and empty items)
result3 = [item.strip() for item in result3 if item.strip()]

print(result3)

# Final result:
# - words separated cleanly
# - punctuation treated as separate tokens
# - ready for vocabulary building
