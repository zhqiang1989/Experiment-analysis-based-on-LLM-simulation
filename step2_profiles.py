"""
Step 2: Generate Participant Profiles (Random approach)
20 participants total, 10 per condition, balanced demographics.
Random seed for reproducibility.
"""

import numpy as np
import pandas as pd
import os
from config import get_study_dir

STUDY_DIR = get_study_dir()
SEED = 42
N_PER_CONDITION = 10
N_TOTAL = N_PER_CONDITION * 2
CONDITIONS = ["loneliness", "control"]

rng = np.random.default_rng(seed=SEED)

profiles = pd.DataFrame({
    "participant_id": range(1, N_TOTAL + 1),
    "age": rng.normal(35, 12, N_TOTAL).clip(18, 75).astype(int),
    "gender": rng.choice(["male", "female", "non-binary"], N_TOTAL, p=[0.48, 0.48, 0.04]),
    "income": rng.lognormal(10.9, 0.7, N_TOTAL).clip(15000, 300000).astype(int),
    "education": rng.choice(
        ["high_school", "bachelors", "masters", "phd"],
        N_TOTAL, p=[0.25, 0.40, 0.25, 0.10]
    ),
    "employment": rng.choice(
        ["full_time", "part_time", "student", "retired", "unemployed"],
        N_TOTAL, p=[0.55, 0.15, 0.15, 0.08, 0.07]
    ),
    "relationship_status": rng.choice(
        ["single", "in_relationship", "married", "divorced"],
        N_TOTAL, p=[0.35, 0.20, 0.35, 0.10]
    ),
})

# Balanced condition assignment
conditions_balanced = np.array(CONDITIONS * N_PER_CONDITION)
profiles["condition"] = rng.permutation(conditions_balanced)

os.makedirs(f"{STUDY_DIR}/experiment_setting", exist_ok=True)
profiles.to_csv(f"{STUDY_DIR}/experiment_setting/profiles.csv", index=False)

print(f"=== PARTICIPANT PROFILES ({STUDY_DIR}) ===")
print(f"Total: {N_TOTAL} | Seed: {SEED}")
print(f"\nCondition assignment:")
print(profiles["condition"].value_counts().to_string())
print(f"\nAge:    M={profiles['age'].mean():.1f}, SD={profiles['age'].std():.1f}")
print(f"Gender: {profiles['gender'].value_counts().to_dict()}")
print(f"Income: M=${profiles['income'].mean():,.0f}, Median=${profiles['income'].median():,.0f}")
print(f"\nProfiles saved to {STUDY_DIR}/experiment_setting/profiles.csv")
