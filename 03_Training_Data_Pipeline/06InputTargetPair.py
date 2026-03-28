import re
import tiktoken  # OpenAI tokenizer (used in GPT models)

# =========================
# STEP 1: LOAD DATA
# =========================

# Read input text file
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

    # Initialize GPT-2 tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Convert raw text → token IDs
    enc_text = tokenizer.encode(raw_text)

    # Print total number of tokens
    print(len(enc_text))


    # =========================
    # STEP 2: CREATE SAMPLE WINDOW
    # =========================

    # Take a subset of tokens (skip first 50 for randomness)
    enc_sample = enc_text[50:]

    # Define context size (number of input tokens)
    context_size = 4

    # Input sequence (x)
    x = enc_sample[:context_size]

    # Target sequence (y) = shifted by 1
    y = enc_sample[1:context_size + 1]

    print(f"x: {x}")
    print(f"y:      {y}")

    # Example:
    # x = [t1, t2, t3, t4]
    # y = [t2, t3, t4, t5]
    # Model learns: next token prediction


    # =========================
    # STEP 3: BUILD CONTEXT → TARGET PAIRS (TOKEN LEVEL)
    # =========================

    # Gradually increase context size
    for i in range(1, context_size + 1):
        context = enc_sample[:i]     # input tokens
        desired = enc_sample[i]      # next token to predict

        print(context, "---->", desired)

    # Example:
    # [t1]        → t2
    # [t1, t2]    → t3
    # [t1, t2, t3]→ t4


    # =========================
    # STEP 4: DECODE FOR HUMAN UNDERSTANDING
    # =========================

    # Convert token IDs back to readable text
    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]

        print(
            tokenizer.decode(context),   # input text
            "---->",
            tokenizer.decode([desired])  # target word
        )

    # Now you can clearly see:
    # "The cat" → "sat"
