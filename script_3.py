import importlib
import pandas as pd
from tqdm import tqdm

print("Running ensemble model...\n")


# Run Both Models (These Save Their Own CSVs)
print("Running simple Word2Vec model (script_1.py)...")
script1 = importlib.import_module("script_1")

print("Running hierarchical Word2Vec model (script_2.py)...")
script2 = importlib.import_module("script_2")


# Load Model Outputs (Already Saved in Correct Order)
path_simple = "./output/output_specialties_multi.csv"
path_hier = "./output/output_specialties_hier_w2v.csv"

print("\nLoading model outputs...")
df_simple = pd.read_csv(path_simple)
df_hier = pd.read_csv(path_hier)

print(f"Simple model: {len(df_simple)} rows")
print(f"Hierarchical model: {len(df_hier)} rows\n")

# Normalize column names
df_simple.columns = [c.strip().lower() for c in df_simple.columns]
df_hier.columns = [c.strip().lower() for c in df_hier.columns]

# Ensure expected columns exist
for c in ["raw_specialty", "nucc_codes", "confidence", "explanation"]:
    if c not in df_simple.columns:
        df_simple[c] = ""
    if c not in df_hier.columns:
        df_hier[c] = ""


# Merge and Combine Predictions
print("🔗 Merging outputs and taking union...\n")

merged = pd.merge(
    df_simple,
    df_hier,
    on="raw_specialty",
    how="outer",
    suffixes=("_simple", "_hier")
)

final_rows = []
for _, row in tqdm(merged.iterrows(), total=len(merged)):
    codes_simple = set(str(row.get("nucc_codes_simple", "")).split("|"))
    codes_hier = set(str(row.get("nucc_codes_hier", "")).split("|"))

    # Remove blanks and JUNK
    codes_simple = {c for c in codes_simple if c and c != "JUNK"}
    codes_hier = {c for c in codes_hier if c and c != "JUNK"}

    # Union of codes
    combined_codes = sorted(codes_simple.union(codes_hier))
    if not combined_codes:
        combined_codes = ["JUNK"]

    # Combine confidence values
    conf_simple = str(row.get("confidence_simple", "")).strip()
    conf_hier = str(row.get("confidence_hier", "")).strip()
    combined_conf = "|".join(filter(None, [conf_simple, conf_hier]))

    # Combine explanations
    expl_simple = str(row.get("explanation_simple", "")).strip()
    expl_hier = str(row.get("explanation_hier", "")).strip()

    final_rows.append({
        "raw_specialty": row.get("raw_specialty", ""),
        "nucc_codes": "|".join(combined_codes),
        "confidence": combined_conf,
        "explanation_simple": expl_simple,
        "explanation_hier": expl_hier
    })

# Save Final Output
final_df = pd.DataFrame(final_rows)
out_path = "./output/output_union_ensemble.csv"
final_df.to_csv(out_path, index=False)

print("\nnion Ensemble Complete!")
print(f"Saved merged CSV → {out_path}")
print(final_df.head(10).to_string(index=False))
