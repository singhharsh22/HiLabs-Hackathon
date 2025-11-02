# HiLabs-Hackathon
An intelligent system that takes unstandardized specialties and maps them to official NUCC taxonomy codes.

### 🧩 Installation
```bash
# Create a virtual environment
python -m venv venv
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt


Such inconsistencies lead to major issues in:
- Claims routing  
- Provider network adequacy  
- Credentialing and compliance  

To address this, the **NUCC (National Uniform Claim Committee)** provides a standardized taxonomy where each specialty has a **unique code** (e.g., `207L00000X` = Anesthesiology).

Your mission:  
Build an intelligent system that maps **unstructured specialty text** to the correct **NUCC taxonomy codes**, handling:
- Abbreviations  
- Misspellings  
- Synonyms  
- Multi-specialty entries  
- Junk or unrecognized text  

---

## 📊 Data Overview

### NUCC Taxonomy Dataset (`nucc_taxonomy_master.csv`)
| Column | Description |
|---------|--------------|
| `code` | Official NUCC taxonomy code |
| `grouping` | Broad professional domain (e.g., Allopathic & Osteopathic Physicians) |
| `classification` | Main specialty area (e.g., Internal Medicine, Surgery) |
| `specialization` | Sub-specialty (e.g., Cardiovascular Disease) |
| `display_name` | Readable label combining classification and specialization |
| `definition` | Description of the specialty |
| `status` | Active / Deprecated |

**Example (`nucc_taxonomy_master.csv.head(10)`):**

| code | grouping | classification | specialization | display_name |
|------|-----------|----------------|----------------|---------------|
| 207L00000X | Allopathic & Osteopathic Physicians | Anesthesiology |  | Anesthesiology |
| 207LA0401X | Allopathic & Osteopathic Physicians | Anesthesiology | Addiction Medicine | Anesthesiology - Addiction Medicine |
| 207LC0200X | Allopathic & Osteopathic Physicians | Anesthesiology | Critical Care Medicine | Anesthesiology - Critical Care Medicine |
| 207LP3000X | Allopathic & Osteopathic Physicians | Anesthesiology | Pain Medicine | Anesthesiology - Pain Medicine |
| ... | ... | ... | ... | ... |

---

### Input Specialties Dataset (`input_specialties.csv`)
| raw_specialty |
|---------------|
| Anesthesiology |
| Cardio |
| Pain & Spine Doc |
| OBGYN |
| Something random |

---

## 📈 Data Exploration

Below are simple histograms showing the distribution of the most common **groupings** and **classifications** in the NUCC taxonomy:

#### 🩺 Groupings
![Grouping Distribution](./output/grouping_distribution.png)

#### 🧠 Classifications
![Classification Distribution](./output/classification_distribution.png)

These plots give an overview of the variety of provider types and the imbalance in how broad or narrow certain groups are.

---

## ⚙️ Approach Overview

### 🧩 Key Components
| Step | Description |
|------|--------------|
| **Text Cleaning** | Normalize text, split CamelCase, remove punctuation, lowercase, and filter stopwords from `./stopwords/english.txt`. |
| **Synonym Expansion** | Replace short forms or abbreviations using `synonyms.csv` (e.g., `OBGYN → obstetrics gynecology`). |
| **Hierarchical Embedding Learning** | Train a 3-stage `Word2Vec` model: (1) `grouping`, (2) `classification`, (3) `specialization + display_name + definition`. |
| **TF-IDF Weighted Sentence Embeddings** | Generate vector representations emphasizing rare, informative words. |
| **Fuzzy Candidate Filtering** | Use token-level fuzzy ratio to restrict the search to relevant NUCC entries. |
| **Cosine Similarity Scoring** | Compute similarity between the input specialty embedding and NUCC taxonomy embeddings. |
| **Threshold Calibration** | Output all matches ≥ 0.7 similarity; otherwise return top match or mark as `JUNK`. |

---

## 🧠 Example Output (`output_specialties_hier_w2v.csv`)

| raw_specialty | nucc_codes | confidence | explanation |
|----------------|-------------|-------------|-------------|
| Cardio | 207RC0000X | 0.92 | Matched ‘cardiology internal medicine cardiovascular disease’ (sim=0.92) |
| OBGYN | 207V00000X | 0.89 | Matched ‘obstetrics gynecology reproductive medicine’ (sim=0.89) |
| Pain & Spine Doc | 207LP2900X | 0.82 | Matched ‘anesthesiology pain medicine’ (sim=0.82) |
| Something random | JUNK |  | No match ≥ 0.7. Best: ‘family medicine’ (sim=0.41) |

---

## 🧩 How to Run Locally

```bash
# 1️⃣ Create a virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Ensure your folder structure is:
# .
# ├── dataset/
# │   ├── nucc_taxonomy_master.csv
# │   ├── input_specialties.csv
# │   └── synonyms.csv
# ├── stopwords/
# │   └── english.txt
# ├── output/
# └── script.py

# 4️⃣ Run the script
python script.py
