# HiLabs-Hackathon
An intelligent system that takes unstandardized specialties and maps them to official NUCC taxonomy codes.

### 🧩 Installation
```bash
# Create a virtual environment
python -m venv venv
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

# HiLabs Hackathon 2025 — Specialty Standardization Challenge
Mapping noisy provider specialties to official NUCC taxonomy codes
1) Problem Introduction

    Health plans store millions of provider records where the specialty is often entered as free text:
    ```
    "Cardio", "ENT Surgeon", "Pediatrics - General", "Addiction Med", ...
    ```
    
    This lack of standardization creates downstream issues (claim routing errors, mismatched directories, and network adequacy gaps). The NUCC Taxonomy provides the federal standard: each specialty/sub-specialty has a unique taxonomy code (e.g., 207L00000X).
    
    Goal: Build a system that reads unstandardized specialties and maps them to official NUCC taxonomy code(s) while handling:
    
    * Abbreviations and synonyms (e.g., OBGYN → Obstetrics & Gynecology)
    
    * Misspellings/typos and partial words
    
    * Multi-specialties and noisy phrases
    
    Junk inputs (return JUNK if confidence is too low)
   
---
2) Datasets Provided

    * NUCC Taxonomy Master: dataset/nucc_taxonomy_master.csv
   
       Columns typically include:
        
        * code — the NUCC taxonomy code (primary output)
        
        * grouping — broad professional domain
        
        * classification — main specialty (e.g., Internal Medicine)
        
        * specialization — sub-specialty (e.g., Cardiovascular Disease)
        
        * display_name — a readable label
        
        * definition — description / notes
        
        * status — active/deprecated
      
      NUCC sample (head):
    ![NUCC Sample](./eda/nucc_dict_head.png)
    
    * Sample Input: dataset/input_specialties.csv
      
      Columns:
        
        * raw_specialty — free-text specialty string(s)
      
      Input sample (head):
   ![Input Sample](./eda/raw_specialty_head.png)

---
3) Synonym Dictionary

    To improve robustness against abbreviations, shorthand, and partial words, the system uses a custom synonym dictionary (dataset/synonyms.csv).
    This file helps map common medical short forms and variants to their standardized forms before embedding or fuzzy comparison.
    
    Each entry consists of two columns:
    
    * synonym → the raw or shorthand form seen in input data
    
    * standard → its canonical expansion used for matching
    
    Only semantic expansions are handled here — spelling corrections and near matches are addressed later via fuzzy matching and vector similarity.
   synonymns.csv sample (head):
![syn](./eda/syn_head.png)

---
4) EDA: Understanding the NUCC Space

Before building the matcher, we visualize label distributions to understand class imbalance and vocabulary:

Grouping distribution:
![Grouping Distribution](./eda/grouping_distribution.png)

Classification distribution:
![Classification Distribution](./eda/classification_distribution.png)

---
5) How to Run Locally

Prereqs: Python 3.10+ recommended. No GPU required.

0. Download this directory, install a virtual environment inside this directory.

1. Create & activate a virtual environment
```
python -m venv venv
venv\Scripts\activate
```
---
2. Install dependencies
In terminal, run :
```
pip install -r requirements.txt
```

3. Ensure repo structure:
```
.
├── dataset/
│   ├── nucc_taxonomy_master.csv
│   ├── input_specialties.csv
│   └── synonyms.csv              # optional
├── stopwords/
│   └── english                   # your stopword list (no NLTK download)
├── output/                       # will be created if absent
├── script.py                     # the end-to-end pipeline
└── README.md
```
5. Run
```
python script.py
```

Result: output/output_specialties_hier_w2v.csv



---
# Provider Specialty Standardization – Word2Vec Ensemble

* This repository implements a multi-stage Word2Vec-based ensemble for mapping raw provider specialty text entries (e.g., "Cardio", "OB/GYN", "Accupunturist") to standardized NUCC Taxonomy codes.
* It was developed as part of the HiLabs Hackathon 2025 challenge: Standardizing Provider Specialties to NUCC Taxonomy.
* Project Structure
```
.
├── dataset/
│   ├── nucc_taxonomy_master.csv      # Reference taxonomy
│   └── input_specialties.csv         # Raw specialties to map
├── synonyms.csv                      # Domain-specific synonym list
├── stopwords/english                 # Local stopword list
├── simple_w2v.py                     # Baseline syntactic Word2Vec model
├── hier_w2v.py                       # Hierarchical semantic Word2Vec model
├── ensem_w2v.py                      # Union ensemble combining both
└── output/                           # Optional output folder
```
* Model Overview
    * simple_w2v.py — Syntactic Word2Vec Mapper
        * A lightweight, token-level embedding model emphasizing string-level similarity.
        * Key features:
        * Trains a Word2Vec model (100-dim skip-gram) on cleaned NUCC text.
        * Expands abbreviations and synonyms using synonyms.csv.
        * Handles spelling variations using rapidfuzz fuzzy matching.
        * Represents each NUCC entry as the mean of token embeddings.
        * For each raw specialty:
              * Cleans text and expands synonyms.
              * Filters NUCC rows with fuzzy word overlap.
              * Computes cosine similarity between input and NUCC embeddings.
              * Returns all codes above SIM_THRESHOLD (default = 0.8).
        
        * Strengths:
              * Robust against typos and abbreviations (e.g., 0b/gyn → OB/GYN).
              * Simple, interpretable results with fuzzy explanations.

Script 1 — Simple Word2Vec + Synonym + Fuzzy Hybrid Mapper

This script performs semantic matching between raw medical specialties (from a user-provided dataset) and the official NUCC Taxonomy Master List using a combination of:

Word2Vec embeddings for semantic understanding

Synonym expansion to handle abbreviations

Word-level fuzzy matching to tolerate spelling variations

Cosine similarity filtering to ensure contextual relevance

📘 Overview

Input files

/dataset/nucc_taxonomy_master.csv   # Official taxonomy reference
/dataset/input_specialties.csv      # Raw specialty names
/synonyms.csv                       # Shorthand → full-form mappings
/stopwords/english                  # Local stopword list


Output

/output/output_specialties_multi.csv


This CSV contains:

raw_specialty	nucc_codes	confidence	explanation
⚙️ Configuration Parameters
Parameter	Description	Default
FUZZ_THRESHOLD	Minimum fuzzy word match ratio between raw and taxonomy words	80
SIM_THRESHOLD	Minimum cosine similarity between sentence embeddings	0.8
FUZZY_FALLBACK	Secondary fuzzy threshold for unrecognized words	70

All thresholds can be tuned at the top of the script:

FUZZ_THRESHOLD = 80
SIM_THRESHOLD = 0.8
FUZZY_FALLBACK = 70

Step-by-Step Methodology
Text Preprocessing

Each entry is cleaned using regex and linguistic normalization:

Converts to lowercase

Removes punctuation and non-alphanumeric characters

Expands symbols (&, -, /)

Strips stopwords and generic clinical words (clinic, doctor, unit, etc.)

Tokenizes and returns clean text

def clean_text(s):
    s = re.sub(r'[^a-z0-9&/\-\s]', ' ', s)
    s = s.replace("&", " and ").replace("-", " ").replace("/", " ")
    s = re.sub(r'\s+', ' ', s).strip()
    ...

Synonym Expansion

A simple CSV-based dictionary (synonyms.csv) maps abbreviations or shorthand terms to canonical full forms.

synonym	standard
obgyn	obstetrics gynecology
cardio	cardiology
ent	otolaryngology
ped	pediatrics

Every token is replaced with its standard equivalent before embedding.

syn_dict = dict(zip(syn["synonym"].str.lower(), syn["standard"].str.lower()))
tokens = [syn_dict.get(w, w) for w in clean.split()]

Word2Vec Model Training

A skip-gram Word2Vec model (sg=1) is trained locally on NUCC’s textual fields (classification, specialization, definition, display_name).

This helps capture contextual meaning, e.g.,
→ “cardiology” ≈ “heart medicine”
→ “ent” ≈ “otolaryngology”

model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    epochs=30
)

Embedding Construction

Each NUCC record is converted into a token list (nucc["tokens"]).

Sentence-level embeddings are computed as the mean of all word vectors.

A precomputed embedding matrix is stored for fast similarity computation.

def sentence_embedding(words):
    vecs = [vocab[w] for w in words if w in vocab]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

Fuzzy + Embedding Matching

Matching is a two-stage hybrid:

🔹 Stage 1: Fuzzy Filtering

Words in each input are compared to NUCC taxonomy tokens.
Candidates are retained if any word pair exceeds FUZZ_THRESHOLD.

if fuzz.ratio(rw, nw) >= threshold:
    candidates.append(i)

🔹 Stage 2: Embedding Similarity

Cosine similarity is computed between the input’s vector and each candidate’s vector.
Only entries exceeding SIM_THRESHOLD are accepted.

sims = cosine_similarity(query_vec, nucc_matrix[candidates])[0]
valid_idxs = np.where(sims >= SIM_THRESHOLD)[0]





   * hier_w2v.py — Hierarchical Semantic Word2Vec Mapper
       * A deeper model incorporating semantic context and hierarchy of the NUCC taxonomy.
       * Key features:
           * Trains progressively on:
               * grouping
               * classification
               * specialization + display_name + definition
               * Uses TF-IDF weighted embeddings for sentence vectors.
               * Combines fuzzy lexical overlap with semantic similarity.
               * Produces ranked matches with similarity-based confidence scores.
        * Strengths:
        * Learns semantic proximity between related specialties (e.g., "acupuncturist" ↔ "reflexologist").
        * More context-aware than the simple model.

* ensem_w2v.py — Union Ensemble
    * A meta-model that executes both base models and merges their predictions.

* Pipeline:
   * Imports and runs both models (simple_w2v and hier_w2v).
   * Aggregates results by raw_specialty.
   * Combines NUCC codes using set union:
   * combined_codes = codes_simple.union(codes_hier)

* Labels the prediction source as:
    * simple_only
    * hier_only
    * simple+hier

* Outputs a unified DataFrame or Excel file.

* Rationale:
    * The simple model handles noisy / misspelled inputs better.
    * The hierarchical model captures conceptual similarity.
    * The ensemble (union) leverages both — increasing recall safely.

Model Overview
simple_w2v.py – Syntactic Word2Vec Mapper

A lightweight Word2Vec model focusing on token-level and string-level similarity.

Key Details

Trains a 100-dimensional skip-gram model using cleaned NUCC text.

Performs fuzzy string matching (rapidfuzz) to handle spelling errors ("0b/gyn" → "ob/gyn").

Expands domain abbreviations using synonyms.csv.

Uses mean embedding per specialty and cosine similarity for mapping.

Outputs

./output/output_specialties_multi.csv
Contains columns: raw_specialty, nucc_codes, explanation.

Strengths

Excels in syntactic robustness — catches spelling and abbreviation variants.

Suitable when input text is noisy or incomplete.

hier_w2v.py – Hierarchical Semantic Word2Vec Mapper

A context-aware model that leverages NUCC taxonomy hierarchy and TF-IDF weighting to learn semantically richer embeddings.

Key Details

Trains sequentially on:

grouping → classification → specialization + definition

Applies TF-IDF weighting during sentence embedding.

Combines fuzzy lexical overlap with cosine similarity over learned vectors.

Provides confidence scores per match.

Outputs

./output/output_specialties_hier_w2v.csv
Columns: raw_specialty, nucc_codes, confidence, explanation.

Strengths

Captures semantic proximity between conceptually related specialties
(e.g., "acupuncturist" ↔ "reflexologist").

More robust when context or domain meaning matters.

ensem_w2v.py – Union Ensemble (Recommended)

A meta-model that runs both models automatically, merges their results, and produces a final unified prediction CSV.

Working Logic

Imports and executes both base scripts (simple_w2v and hier_w2v).

Aggregates and merges predictions by raw_specialty.

Takes the union of NUCC codes from both models:

combined_codes = codes_simple.union(codes_hier)


Labels each prediction source as:

simple_only

hier_only

simple+hier

Produces a consolidated, high-confidence result table.

Outputs

./output/output_union_ensemble.csv (or .xlsx)
Columns:
raw_specialty, nucc_codes, source, explanation_simple, explanation_hier

Why It’s Better

Combines syntactic recall (from the simple model) with semantic precision (from the hierarchical model).

The union ensemble ensures no valid prediction is lost — maximizing accuracy safely.

to be safe, i think i should combine (take union of both the results)
as simple w2v works well on spelling mistakes (like 0b/gyn) where complex one doesn't, and the complex one gives more robust predictions (embeds according to classes also, giving nearby possible answers), like accupunturist and reflexologist, which the simpler model doesn't predict, by learning a bit semantics, while the simple model learns only the syntactics

    7) Confidence Tuning
    
    You control strictness via two knobs:
    
    Fuzzy Token Filter (fuzz_threshold)
    Typical: 70–85.
    Lower → more candidates (recall↑, noise↑).
    Higher → fewer candidates (precision↑, may miss typos).
    
    Cosine Similarity Cutoff (sim_threshold)
    Typical: 0.60–0.80.
    Lower → more matches (recall↑).
    Higher → stricter mapping (precision↑).
    
    Calibration Tip:
    Take a small labeled validation file (10–50 rows), sweep thresholds (grid search), pick the pair that maximizes your preferred metric (e.g., F1 or accuracy), and lock it in the README.


9) Notes, Limitations & Extensions

Typos vs. synonyms:
Typos are primarily handled by fuzzy token overlap; semantic variants are handled by embeddings (plus optional synonyms).

Multi-label vs. single-label:
We allow multiple NUCC codes when similarity ties exceed threshold (useful for composite inputs like “Cardio/Diab”).

Speed:
The fuzzy candidate filter keeps the cosine step fast enough to process ~20k rows well under 15 minutes on a typical laptop.

Extensions:

Cache or pre-persist NUCC embeddings (skip recomputation on each run)

Add domain-specific synonyms (e.g., “PM&R” → “physical medicine rehabilitation”)

Add a lightweight spelling-correction layer before fuzzy (optional)

Swap Word2Vec for BioWordVec (pretrained biomedical word2vec) if you ship the vectors locally
