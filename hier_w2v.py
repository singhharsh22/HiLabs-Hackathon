import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# CONFIGURATION (Editable Thresholds)
FUZZ_THRESHOLD = 80     # fuzzy word-level match threshold (0–100)
SIM_THRESHOLD = 0.8     # cosine similarity threshold (0–1)
TFIDF_MIN_WEIGHT = 1e-9 # smoothing for division by zero
STOPWORDS_PATH = "./stopwords/english"
SYNONYMS_PATH = "./synonyms.csv"


# Loading Stopwords & Synonyms
with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
    STOPWORDS = set(w.strip().lower() for w in f if w.strip())

try:
    syn = pd.read_csv(SYNONYMS_PATH)
    syn.columns = [c.strip().lower() for c in syn.columns]
    syn_dict = dict(zip(syn["synonym"].str.lower(), syn["standard"].str.lower()))
    print(f"Loaded {len(syn_dict)} synonym mappings.")
except Exception as e:
    syn_dict = {}
    print("No valid synonyms.csv found or error reading it:", e)

# Load Data
base_path = "./dataset"
nucc = pd.read_csv(f"{base_path}/nucc_taxonomy_master.csv")
inp = pd.read_csv(f"{base_path}/input_specialties.csv")

nucc.columns = [c.strip().lower() for c in nucc.columns]
inp.columns = [c.strip().lower() for c in inp.columns]

# Cleaning (with synonym expansion)
def expand_synonyms(word):
    """Expand abbreviations using synonyms.csv if available"""
    return syn_dict.get(word, word)

def clean_text(s):
    """Lowercase, remove punctuation, expand synonyms, strip stopwords"""
    if pd.isna(s):
        return ""
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(s))  # split camelcase
    s = s.lower()
    s = re.sub(r"[^a-z0-9&/\-\s]", " ", s)
    s = s.replace("&", " and ").replace("-", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s).strip()
    tokens = [expand_synonyms(w) for w in s.split() if w not in STOPWORDS]
    return tokens

# Prepare Hierarchical Corpora
for col in ["grouping", "classification", "specialization", "display_name", "definition"]:
    if col not in nucc.columns:
        nucc[col] = ""

nucc["grouping_tokens"] = nucc["grouping"].apply(clean_text)
nucc["classification_tokens"] = nucc["classification"].apply(clean_text)
nucc["combined_tokens"] = (
    nucc["classification"].fillna('') + " " +
    nucc["specialization"].fillna('') + " " +
    nucc["display_name"].fillna('') + " " +
    nucc["definition"].fillna('')
).apply(clean_text)

# Train Hierarchical Word2Vec
print("Training hierarchical Word2Vec model...")
sentences_grouping = nucc["grouping_tokens"].tolist()
sentences_class = nucc["classification_tokens"].tolist()
sentences_full = nucc["combined_tokens"].tolist()

# Base model on grouping
model = Word2Vec(sentences=sentences_grouping, vector_size=150, window=5, min_count=1, sg=1, epochs=20)

# Update on classification
model.build_vocab(sentences_class, update=True)
model.train(sentences_class, total_examples=len(sentences_class), epochs=15)

# Fine-tune on full combined tokens
model.build_vocab(sentences_full, update=True)
model.train(sentences_full, total_examples=len(sentences_full), epochs=15)

print(f"Vocab size: {len(model.wv)} words")


# TF-IDF weighting for robust sentence embeddings
print("Building TF-IDF weighting...")
dictionary = Dictionary(sentences_full)
corpus = [dictionary.doc2bow(text) for text in sentences_full]
tfidf_model = TfidfModel(corpus)

def sentence_embedding_tfidf(words):
    """Compute weighted embedding using TF-IDF"""
    vecs, weights = [], []
    bow = dictionary.doc2bow(words)
    tfidf_weights = dict(tfidf_model[bow])
    for w in words:
        if w in model.wv and dictionary.token2id.get(w) in tfidf_weights:
            weight = tfidf_weights[dictionary.token2id[w]]
            vecs.append(model.wv[w] * weight)
            weights.append(weight)
    if not vecs:
        return np.zeros(model.vector_size)
    return np.sum(vecs, axis=0) / (np.sum(weights) + TFIDF_MIN_WEIGHT)

# Compute NUCC sentence embeddings
print("Computing NUCC sentence embeddings...")
nucc["embedding"] = nucc["combined_tokens"].apply(sentence_embedding_tfidf)
nucc_matrix = np.vstack(nucc["embedding"].values)

# Prediction via Fuzzy + Embedding Similarity
def fuzzy_word_overlap(raw_words, nucc_words, threshold=FUZZ_THRESHOLD):
    """Return True if any raw word is fuzzily similar to any NUCC word."""
    for rw in raw_words:
        for nw in nucc_words:
            if fuzz.ratio(rw, nw) >= threshold:
                return True
    return False

def map_specialty_to_nucc(raw_text, sim_threshold=SIM_THRESHOLD, fuzz_threshold=FUZZ_THRESHOLD):
    """Map input specialty to NUCC taxonomy"""
    raw_tokens = clean_text(raw_text)
    if not raw_tokens:
        return [{"nucc_code": "JUNK", "confidence": 0.0, "explanation": "Empty input"}]

    # Candidate selection via fuzzy overlap
    candidates = []
    for i, nucc_words in enumerate(nucc["combined_tokens"]):
        if fuzzy_word_overlap(raw_tokens, nucc_words, fuzz_threshold):
            candidates.append(i)

    if not candidates:
        return [{"nucc_code": "JUNK", "confidence": 0.0, "explanation": "No fuzzy overlap"}]

    # Sentence embedding
    query_emb = sentence_embedding_tfidf(raw_tokens).reshape(1, -1)
    sims = cosine_similarity(query_emb, nucc_matrix[candidates])[0]

    valid_idx_local = np.where(sims >= sim_threshold)[0]
    if len(valid_idx_local) == 0:
        best_idx = np.argmax(sims)
        idx_global = candidates[best_idx]
        text_example = " ".join(nucc.iloc[idx_global]["combined_tokens"])
        return [{
            "nucc_code": "JUNK",
            "confidence": float(sims[best_idx]),
            "explanation": f"No match ≥ {sim_threshold}. Best match: '{text_example}' (sim={sims[best_idx]:.2f})"
        }]

    # Sorting valid matches by similarity
    order = np.argsort(sims[valid_idx_local])[::-1]
    valid_idx_local = valid_idx_local[order]

    results = []
    for idx_local in valid_idx_local:
        idx_global = candidates[idx_local]
        sim_val = sims[idx_local]
        text_example = " ".join(nucc.iloc[idx_global]["combined_tokens"])
        results.append({
            "nucc_code": nucc.iloc[idx_global].get("code", "N/A"),
            "grouping": nucc.iloc[idx_global].get("grouping", ""),
            "classification": nucc.iloc[idx_global].get("classification", ""),
            "specialization": nucc.iloc[idx_global].get("specialization", ""),
            "display_name": nucc.iloc[idx_global].get("display_name", ""),
            "confidence": float(sim_val),
            "explanation": f"Matched '{text_example}' (sim={sim_val:.2f})"
        })
    return results

# Applying Mapping and Saving
all_results = []

for raw in tqdm(inp["raw_specialty"], desc="Mapping specialties"):
    if pd.isna(raw) or not str(raw).strip():
        all_results.append({
            "raw_specialty": "",
            "nucc_codes": "JUNK",
            "confidence": "",
            "explanation": ""
        })
        continue

    matches = map_specialty_to_nucc(raw)
    codes = [m["nucc_code"] for m in matches if m.get("nucc_code")]
    confs = [str(round(m["confidence"], 3)) for m in matches if m.get("confidence")]
    expls = [m["explanation"] for m in matches if m.get("explanation")]

    row = {
        "raw_specialty": raw if str(raw).strip() else "",
        "nucc_codes": "|".join(codes) if codes else "JUNK",
        "confidence": "|".join(confs) if confs else "",
        "explanation": " || ".join(expls) if expls else ""
    }
    all_results.append(row)

out_df = pd.DataFrame(all_results, columns=["raw_specialty", "nucc_codes", "confidence", "explanation"])
out_path = "./output/output_specialties_hier_w2v.csv"
out_df.to_csv(out_path, index=False)
print("Model complete! Total results:", len(all_results))

print("Hierarchical Word2Vec + Fuzzy + Synonym hybrid mapping complete!")
print("Saved to:", out_path)
print(out_df.head(10).to_string(index=False))

FINAL_RESULTS = all_results
