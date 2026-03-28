import gensim.downloader as api  # Library to load pretrained NLP models

# =========================
# STEP 1: LOAD PRETRAINED MODEL
# =========================

# Load Google's pretrained Word2Vec model (300-dimensional vectors)
# Trained on huge corpus → captures semantic meaning
model = api.load("word2vec-google-news-300")

# Assign model to variable (contains word → vector mappings)
word_vectors = model


# =========================
# STEP 2: WORD → VECTOR
# =========================

# Get vector representation of word "computer"
print(word_vectors['computer'])

# Check shape of embedding vector (should be 300 dimensions)
print(word_vectors['cat'].shape)


# =========================
# STEP 3: WORD ANALOGY (VERY IMPORTANT)
# =========================

# king - man + women ≈ queen
print(word_vectors.most_similar(
    positive=['king', 'women'],
    negative=['man'],
    topn=5
))

# This shows embeddings capture relationships like:
# gender, roles, hierarchy


# =========================
# STEP 4: SIMILARITY SCORES
# =========================

# Cosine similarity between word vectors
print(word_vectors.similarity('woman', 'man'))
print(word_vectors.similarity('aunt', 'uncle'))
print(word_vectors.similarity('niece', 'nephew'))
print(word_vectors.similarity('tree', 'wood'))

# Higher value → more similar meaning


# =========================
# STEP 5: MOST SIMILAR WORDS
# =========================

# Find words similar to "tower"
print(word_vectors.most_similar("tower", topn=5))

# Shows semantic neighbors in vector space
