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

Abbreviations and synonyms (e.g., OBGYN → Obstetrics & Gynecology)

Misspellings/typos and partial words

Multi-specialties and noisy phrases

Junk inputs (return JUNK if confidence is too low)

***

2) Datasets Provided

NUCC Taxonomy Master: dataset/nucc_taxonomy_master.csv
Columns typically include:

code — the NUCC taxonomy code (primary output)

grouping — broad professional domain

classification — main specialty (e.g., Internal Medicine)

specialization — sub-specialty (e.g., Cardiovascular Disease)

display_name — a readable label

definition — description / notes

status — active/deprecated

Sample Input: dataset/input_specialties.csv
Columns:

raw_specialty — free-text specialty string(s)

You will add two preview images under output/ that show the first 10 rows of each dataset.

2.1 Preview Images (to be added by you)

NUCC sample (head):
![NUCC Sample](./output/nucc_head.png)

Input sample (head):
![Input Sample](./output/input_head.png)

How to create these in a notebook quickly:

import pandas as pd
nucc = pd.read_csv("dataset/nucc_taxonomy_master.csv")
inp  = pd.read_csv("dataset/input_specialties.csv")
nucc.head(10).to_markdown("output/nucc_head.md", index=False)
inp.head(10).to_markdown("output/input_head.md", index=False)
# Optionally render markdown to image (or screenshot in notebook and save as PNG)

3) Synonym Dictionary (Optional but Recommended)

To handle healthcare shorthand and abbreviations, you can provide dataset/synonyms.csv with two columns:

synonym	standard
obgyn	obstetrics gynecology
cardio	cardiology
ent	otolaryngology
ped	pediatrics
fm	family medicine
ortho	orthopedic

The system does not fix spelling mistakes here (that’s left to fuzzy matching); synonyms are for canonical expansions only.

4) EDA: Understanding the NUCC Space

Before building the matcher, we visualize label distributions to understand class imbalance and vocabulary:

Grouping distribution:
![Grouping Distribution](./output/grouping_distribution.png)

Classification distribution:
![Classification Distribution](./output/classification_distribution.png)

How to generate these images (notebook snippet):

import pandas as pd
import matplotlib.pyplot as plt

nucc = pd.read_csv("dataset/nucc_taxonomy_master.csv")

plt.figure()
nucc['grouping'].value_counts().head(20).plot(kind='bar', rot=60)
plt.title('Top Groupings (count)')
plt.tight_layout()
plt.savefig("output/grouping_distribution.png", dpi=150)

plt.figure()
nucc['classification'].value_counts().head(30).plot(kind='bar', rot=60)
plt.title('Top Classifications (count)')
plt.tight_layout()
plt.savefig("output/classification_distribution.png", dpi=150)


In the README, we embed the resulting PNGs with the ![alt](path) syntax.

5) Methodology (Step-by-Step)

This repo implements a hierarchical Word2Vec + TF-IDF + Fuzzy hybrid:

5.1 Text Cleaning

Split CamelCase (PainSpine → Pain Spine)

Lowercase, remove punctuation except / - & (kept as signals), normalize whitespace

Remove stopwords using ./stopwords/english.txt (no external downloads)

5.2 Build Hierarchical Corpora

We prepare three corpora from the NUCC dataset:

Grouping tokens

Classification tokens

Combined tokens = classification + specialization + display_name + definition

This lets us infuse broad → fine semantics progressively.

5.3 Train Word2Vec in Three Stages

Train base Word2Vec on grouping tokens

Update the same model on classification tokens

Fine-tune on the full combined tokens

This mimics the taxonomy hierarchy and nudges embeddings to respect NUCC structure without labeled supervised training.

5.4 TF-IDF Weighted Sentence Embeddings

Build a Gensim Dictionary over combined tokens and a TfidfModel

For each NUCC row and each query, compute a TF-IDF weighted mean of token vectors

This gives sentence embeddings that emphasize discriminative words

5.5 Candidate Narrowing with Fuzzy Token Overlap

Before cosine similarity, filter NUCC rows by token-level fuzzy match (RapidFuzz ratio)

This drops obviously unrelated specialties and speeds up similarity

Why both?

Fuzzy catches typos/variants at token level (e.g., 0B/GYN, obgyn, ob/gyn).

Embeddings capture semantic similarity across synonyms and related phrasing.

5.6 Cosine Similarity & Thresholding

Compute cosine similarity between the query embedding and candidate NUCC embeddings

Return all matches with similarity ≥ SIM_THRESHOLD (e.g., 0.70)

If none qualify, return the best match score for context or JUNK if below threshold policy

5.7 Multi-Match Output

The system can output multiple taxonomy codes (pipe-separated) for genuinely ambiguous inputs (e.g., “Cardio/Diab”).

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
