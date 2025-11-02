import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from rapidfuzz import fuzz, process
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from tqdm import tqdm


# CONFIGURATION
FUZZ_THRESHOLD = 80      # word-level fuzzy match threshold (0–100)
SIM_THRESHOLD = 0.8      # cosine similarity threshold (0–1)
FUZZY_FALLBACK = 70      # fallback threshold for fuzzy word correction


# Setup
STOPWORDS_PATH = "./stopwords/english"
with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
    STOPWORDS = set(w.strip().lower() for w in f if w.strip())

base_path = "./dataset"
nucc = pd.read_csv(f"{base_path}/nucc_taxonomy_master.csv")
inp = pd.read_csv(f"{base_path}/input_specialties.csv")
syn = pd.read_csv("./synonyms.csv")

# Normalize column names
nucc.columns = [c.strip().lower() for c in nucc.columns]
inp.columns = [c.strip().lower() for c in inp.columns]
syn.columns = [c.strip().lower() for c in syn.columns]

print("NUCC columns:", list(nucc.columns))


# Text Cleaning
def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9&/\-\s]', ' ', s)
    s = s.replace("&", " and ").replace("-", " ").replace("/", " ")
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'\b(dept|department|clinic|center|unit|practice|physician|doctor|doc)\b', ' ', s)
    tokens = [w for w in s.split() if w not in STOPWORDS]
    return " ".join(tokens)

nucc["combined"] = (
    nucc.get("classification", "").fillna('') + " " +
    nucc.get("specialization", "").fillna('') + " " +
    nucc.get("definition", "").fillna('') + " " +
    nucc.get("display_name", "").fillna('')
).apply(clean_text)

nucc["tokens"] = nucc["combined"].apply(lambda x: x.split())
sentences = nucc["tokens"].tolist()


# Train Word2Vec
model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    workers=4,
    epochs=30
)

# Vocabulary dictionary
vocab = {word: model.wv[word] for word in model.wv.key_to_index.keys()}
print(f"Vocabulary size: {len(vocab)} words")


# Synonyms
syn_dict = dict(zip(syn["synonym"].str.lower(), syn["standard"].str.lower()))
def expand_synonyms(word):
    return syn_dict.get(word, word)


# Embedding helper
def sentence_embedding(words):
    vecs = [vocab[w] for w in words if w in vocab]
    if not vecs:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)

nucc["embedding"] = nucc["tokens"].apply(sentence_embedding)
nucc_matrix = np.vstack(nucc["embedding"].values)


# Fuzzy comparison
def fuzzy_word_overlap(raw_words, nucc_words, threshold=FUZZ_THRESHOLD):
    """Return True if any raw word is fuzzily similar to any NUCC word."""
    for rw in raw_words:
        for nw in nucc_words:
            if fuzz.ratio(rw, nw) >= threshold:
                return True
    return False


# Mapping function
def map_specialty_to_nucc(raw_text, sim_threshold=SIM_THRESHOLD, fuzz_threshold=FUZZ_THRESHOLD):
    raw_text = str(raw_text).strip()
    
    # Early check: if NUCC taxonomy code
    if re.match(r"^[0-9A-Z]{10}$", raw_text.upper()) and raw_text.upper().endswith("X"):
        return [{
            "nucc_code": raw_text.upper(),
            "confidence": 1.0,
            "explanation": "Direct NUCC taxonomy code detected (skipped mapping)"
        }]
    
    clean = clean_text(raw_text)
    tokens = [expand_synonyms(w) for w in clean.split()]
    if not tokens:
        return [{"nucc_code": "JUNK", "confidence": 0.0, "explanation": "Empty after cleaning"}]

    # Fuzzy candidates
    candidates = []
    for idx, nucc_words in enumerate(nucc["tokens"]):
        if fuzzy_word_overlap(tokens, nucc_words, threshold=fuzz_threshold):
            candidates.append(idx)

    if not candidates:
        return [{"nucc_code": "JUNK", "confidence": 0.0, "explanation": "No fuzzy word overlap"}]

    # Token embeddings
    token_vecs = []
    for w in tokens:
        if w in vocab:
            token_vecs.append(vocab[w])
        else:
            best_match = process.extractOne(w, list(vocab.keys()), scorer=fuzz.ratio)
            if best_match and best_match[1] >= FUZZY_FALLBACK:
                token_vecs.append(vocab[best_match[0]])

    if not token_vecs:
        return [{"nucc_code": "JUNK", "confidence": 0.0, "explanation": "No valid embeddings"}]

    sent_vec = np.mean(token_vecs, axis=0).reshape(1, -1)

    sims = cosine_similarity(sent_vec, nucc_matrix[candidates])[0]
    valid_idxs_local = np.where(sims >= sim_threshold)[0]

    if len(valid_idxs_local) == 0:
        best_idx = np.argmax(sims)
        idx_global = candidates[best_idx]
        return [{
            "nucc_code": "JUNK",
            "confidence": float(sims[best_idx]),
            "explanation": f"No match above threshold ({sim_threshold}); best sim={sims[best_idx]:.2f}"
        }]

    results = []
    for idx_local in valid_idxs_local:
        idx_global = candidates[idx_local]
        sim_val = sims[idx_local]
        code_col = "code" if "code" in nucc.columns else list(nucc.columns)[0]
        results.append({
            "nucc_code": nucc.iloc[idx_global][code_col],
            "confidence": float(sim_val),
            "explanation": f"Matched with '{nucc.iloc[idx_global]['combined']}' "
                           f"(word-fuzzy + sim={sim_val:.2f})"
        })
    return results


# Apply Mapping
all_results = []

for raw in tqdm(inp["raw_specialty"], desc="Word-level fuzzy mapping (multi)"):
    matches = map_specialty_to_nucc(raw)
    codes = [m["nucc_code"] for m in matches]
    confs = [str(round(m["confidence"], 3)) for m in matches]
    expls = [m["explanation"] for m in matches]

    row = {
        "raw_specialty": raw,
        "nucc_codes": "|".join(codes) if codes else "JUNK",
        "confidence": "|".join(confs) if confs else "",
        "explanation": " || ".join(expls) if expls else ""
    }
    all_results.append(row)


# Save Output
out_df = pd.DataFrame(all_results, columns=["raw_specialty", "nucc_codes", "confidence", "explanation"])
out_path = "./output/output_specialties_multi.csv"
out_df.to_csv(out_path, index=False)
print("Model complete! Total results:", len(all_results))

print("Word-level fuzzy hybrid mapping complete!")
print("Saved file:", out_path)
print(out_df.head(10).to_string(index=False))

FINAL_RESULTS = all_results
