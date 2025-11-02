import importlib
import pandas as pd
import numpy as np
from tqdm import tqdm


base_path = "./dataset"
inp = pd.read_csv(f"{base_path}/input_specialties.csv")
inp.columns = [c.strip().lower() for c in inp.columns]

# Random 1000 rows for validation
sample_df = inp.sample(n=1000, random_state=42).reset_index(drop=True)
print(f"Selected {len(sample_df)} random specialties for consistency check.\n")

def run_ensemble_once():
    """Run ensemble model (script_3.py) and return its predictions"""
    script3 = importlib.import_module("script_3")
    importlib.reload(script3)   # ensure fresh run each time
    results = getattr(script3, "FINAL_RESULTS", None)
    if results is None:
        raise ValueError("⚠️ script_3.py must define FINAL_RESULTS")
    df = pd.DataFrame(results)
    return df


# Make Two Predictions (Two Separate Runs)
print("🚀 Running ensemble first time...")
df_pred1 = run_ensemble_once()

print("\n🚀 Running ensemble second time...")
df_pred2 = run_ensemble_once()


# Compare Results
print("\n🔍 Comparing predictions for consistency...\n")

merged = pd.merge(
    df_pred1, df_pred2, on="raw_specialty", how="inner",
    suffixes=("_run1", "_run2")
)

def normalize_codes(codes):
    if pd.isna(codes):
        return set()
    return {c.strip() for c in str(codes).split("|") if c.strip() and c != "JUNK"}

merged["codes_run1"] = merged["nucc_codes_run1"].apply(normalize_codes)
merged["codes_run2"] = merged["nucc_codes_run2"].apply(normalize_codes)

# Agreement check
merged["match"] = merged.apply(
    lambda r: r["codes_run1"] == r["codes_run2"], axis=1
)

accuracy = merged["match"].mean()
error_rate = 1 - accuracy

print("")
print("ENSEMBLE CONSISTENCY METRICS")
print("")
print(f"Total Samples Evaluated: {len(merged)}")
print(f"Agreement Accuracy: {accuracy*100:.2f}%")
print(f"Disagreement/Error Rate: {error_rate*100:.2f}%")
print("\n")


# Show Sample Disagreements (No Saving)
disagreements = merged[~merged["match"]]
if len(disagreements) > 0:
    print(f"Total disagreements: {len(disagreements)}\n")
    print("Sample inconsistent predictions (up to 10):\n")
    print(disagreements[["raw_specialty", "nucc_codes_run1", "nucc_codes_run2"]]
          .head(10).to_string(index=False))
else:
    print("Perfect consistency across both runs! No disagreements found.\n")
