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
5) Methodology & Pipeline

   This section explains how the Hierarchical Word2Vec + Fuzzy Hybrid Mapper standardizes raw provider specialties to NUCC taxonomy codes.
    The approach combines text cleaning, hierarchical Word2Vec embeddings, TF–IDF weighting, and fuzzy token matching to ensure robust mapping even for misspellings, abbreviations, and domain-specific shorthand.
   
    ---
    Step 0 – Stopword Initialization
    
    The pipeline uses a locally stored stopword list to ensure reproducibility (no external downloads).
    All stopwords are read from ./stopwords/english.
    ```
    STOPWORDS_PATH = "./stopwords/english"
    with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
        STOPWORDS = set(w.strip().lower() for w in f if w.strip())
    ```
    
    This ensures the solution can run offline, meeting the hackathon’s no external API/dependency requirement.
   
    ---
    Step 1 – Data Loading
    
    Both reference (nucc_taxonomy_master.csv) and input (input_specialties.csv) datasets are loaded from the local ./dataset folder.
    ```
    base_path = "./dataset"
    nucc = pd.read_csv(f"{base_path}/nucc_taxonomy_master.csv")
    inp  = pd.read_csv(f"{base_path}/input_specialties.csv")
    
    nucc.columns = [c.strip().lower() for c in nucc.columns]
    inp.columns  = [c.strip().lower()  for c in inp.columns]
    ```
    
    This normalization step ensures consistent downstream column access.
   
    ---
    Step 2 – Text Cleaning & Tokenization
    
    Free-text fields (like classification, specialization, display name, and definition) are cleaned and tokenized.
    The cleaning removes punctuation, normalizes case, and filters stopwords.
    ```
    def clean_text(s):
        if pd.isna(s): return ""
        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(s))
        s = re.sub(r"[^a-z0-9&/\-\s]", " ", s.lower())
        s = s.replace("&"," and ").replace("-", " ").replace("/", " ")
        tokens = [w for w in s.split() if w not in STOPWORDS]
        return tokens
    ```
    
    This ensures that both NUCC and raw specialties share a normalized lexical space.
   
    ---
    Step 3 – Hierarchical Corpus Construction
    
    NUCC data has a natural hierarchy:
    Grouping → Classification → Specialization → Display Name → Definition
    
    Each level is tokenized separately, then combined to build a progressively richer corpus.
    ```
    nucc["grouping_tokens"]       = nucc["grouping"].apply(clean_text)
    nucc["classification_tokens"] = nucc["classification"].apply(clean_text)
    nucc["combined_tokens"] = (
        nucc["classification"].fillna('') + " " +
        nucc["specialization"].fillna('') + " " +
        nucc["display_name"].fillna('') + " " +
        nucc["definition"].fillna('')
    ).apply(clean_text)
    ```
    ---
    Step 4 – Hierarchical Word2Vec Training
    
    A progressive fine-tuning strategy trains Word2Vec at increasing semantic depths:
    
    Grouping level – learns broad medical domains
    
    Classification level – refines main specialty clusters
    
    Combined level – encodes detailed contextual information
    ```
    model = Word2Vec(sentences=nucc["grouping_tokens"], vector_size=150, window=5, sg=1, epochs=20)
    
    # Fine-tune on deeper levels
    model.build_vocab(nucc["classification_tokens"], update=True)
    model.train(nucc["classification_tokens"], total_examples=len(nucc), epochs=15)
    
    model.build_vocab(nucc["combined_tokens"], update=True)
    model.train(nucc["combined_tokens"], total_examples=len(nucc), epochs=15)
    ```
    
    This hierarchical retraining ensures embeddings capture both macro-specialty relationships and micro-subspecialty nuances.
   
    ---
    Step 5 – TF-IDF Weighted Sentence Embeddings
    
    To aggregate token embeddings into a single sentence vector, a TF–IDF weighting scheme is applied.
    This gives higher weight to discriminative medical terms (e.g., cardio, neuro) and downweights generic words.
    ```
    dictionary = Dictionary(nucc["combined_tokens"])
    corpus     = [dictionary.doc2bow(text) for text in nucc["combined_tokens"]]
    tfidf_model = TfidfModel(corpus)
    
    def sentence_embedding_tfidf(words):
        vecs, weights = [], []
        bow = dictionary.doc2bow(words)
        tfidf_weights = dict(tfidf_model[bow])
        for w in words:
            if w in model.wv and dictionary.token2id.get(w) in tfidf_weights:
                weight = tfidf_weights[dictionary.token2id[w]]
                vecs.append(model.wv[w] * weight)
                weights.append(weight)
        return np.sum(vecs, axis=0) / (np.sum(weights) + 1e-9) if vecs else np.zeros(model.vector_size)
    ```
    ---
    Step 6 – NUCC Embedding Precomputation
    
    Each NUCC row’s cleaned text is embedded once and stored for efficient similarity lookup.
    ```
    nucc["embedding"] = nucc["combined_tokens"].apply(sentence_embedding_tfidf)
    nucc_matrix = np.vstack(nucc["embedding"].values)
    ```
    ---
    Step 7 – Hybrid Matching: Fuzzy + Semantic
    
    The system uses a two-stage matching pipeline:
    
    Fuzzy Filtering — Quickly narrow down NUCC candidates that share lexically similar tokens with the input
    
    Semantic Ranking — Among those, compute cosine similarity between TF-IDF-weighted embeddings
    ```
    def fuzzy_word_overlap(raw_words, nucc_words, threshold=80):
        for rw in raw_words:
            for nw in nucc_words:
                if fuzz.ratio(rw, nw) >= threshold:
                    return True
        return False
    ```
    
    For each raw specialty, candidates passing fuzzy overlap are scored by semantic similarity:
    ```
    sims = cosine_similarity(query_emb, nucc_matrix[candidates])[0]
    valid_idx = np.where(sims >= sim_threshold)[0]
    ```
    
    Results are sorted by descending similarity.
   
    ---
    Step 8 – Output Assembly & Cleaning
    
    Each raw specialty is mapped to all matching taxonomy codes above the similarity threshold.
    Empty inputs are skipped gracefully, and unmatched entries are labeled JUNK.
    ```
    row = {
        "raw_specialty": raw,
        "nucc_codes": "|".join(codes) if codes else "JUNK",
        "confidence": "|".join(confs),
        "explanation": " || ".join(expls)
    }
    ```
    
    The final standardized file is saved as:
    ```
    ./output/output_specialties_hier_w2v.csv
    ```
    ---
    6) Output Schema
    
    The tool writes output/output_specialties_hier_w2v.csv with:
    
    Column	Description
    raw_specialty	Original input string (empty if missing)
    nucc_codes	Pipe-separated list of NUCC codes or JUNK
    confidence	Pipe-separated cosine similarities (0–1, rounded) for each returned code
    explanation	Short rationale e.g., “Matched ‘…tokens…’ (sim=0.83)”
    
    Example:
    
    raw_specialty,nucc_codes,confidence,explanation
    Cardio,207RC0000X,0.92,Matched 'cardiology internal medicine cardiovascular disease' (sim=0.92)
    OBGYN,207V00000X,0.89,Matched 'obstetrics gynecology reproductive medicine' (sim=0.89)
    Something random,JUNK,,No match ≥ 0.70. Best match: 'family medicine' (sim=0.41)

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

8) How to Run Locally

Prereqs: Python 3.10+ recommended. No GPU required.

# 1) Create & activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Ensure repo structure:
# .
# ├── dataset/
# │   ├── nucc_taxonomy_master.csv
# │   ├── input_specialties.csv
# │   └── synonyms.csv              # optional
# ├── stopwords/
# │   └── english.txt               # your stopword list (no NLTK download)
# ├── output/                       # will be created if absent
# ├── script.py                     # the end-to-end pipeline
# └── README.md

# 4) Run
python script.py


Result: output/output_specialties_hier_w2v.csv

If you’re using a notebook for EDA, save plots to output/ and embed them in this README via ![title](path).

9) Repository Layout
.
├── dataset/
│   ├── nucc_taxonomy_master.csv
│   ├── input_specialties.csv
│   └── synonyms.csv                # optional
├── stopwords/
│   └── english.txt                 # local stopword file
├── output/
│   ├── grouping_distribution.png
│   ├── classification_distribution.png
│   ├── nucc_head.png               # your preview image
│   └── input_head.png              # your preview image
├── notebooks/
│   └── exploratory_analysis.ipynb  # optional, for EDA/plots
├── script.py                        # hierarchical Word2Vec + fuzzy mapper
├── requirements.txt
└── README.md

10) Notes, Limitations & Extensions

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

11) FAQ

Q: Does this require CUDA/GPU?
A: No. It’s pure CPU (Gensim + RapidFuzz + scikit-learn).

Q: Can I use internet APIs?
A: No. Everything runs locally per challenge rules.

Q: What if my input has blanks?
A: We output an empty raw_specialty, empty confidence/explanation, and nucc_codes=JUNK.

Q: How do I embed the preview tables and histograms in this README?
A: Save images to ./output/ and reference them:
![My Plot](./output/grouping_distribution.png)

Author: Harsh Singh (IIT Kanpur)
Hackathon: HiLabs 2025 — Specialty Standardization Challenge
