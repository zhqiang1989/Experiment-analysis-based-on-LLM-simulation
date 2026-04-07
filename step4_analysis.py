"""
Step 4: Analyze the Data
Primary test: Loneliness increases WTP for kid-adulting products.
Key analysis: Integrated OLS regression with cluster-robust SEs.
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

from config import get_study_dir
STUDY_DIR = get_study_dir()

# --- Load data ---
responses = pd.read_csv(f"{STUDY_DIR}/results/responses.csv")
profiles = pd.read_csv(f"{STUDY_DIR}/experiment_setting/profiles.csv")
with open(f"{STUDY_DIR}/experiment_setting/materials.json", "r", encoding="utf-8") as f:
    materials = json.load(f)

df = responses.merge(profiles, on="participant_id", suffixes=("", "_p"))
if "condition_p" in df.columns:
    df.drop(columns=["condition_p"], inplace=True)

product_map = {p["product_id"]: p["product_name"] for p in materials["products"]}
df["product_name"] = df["product_id"].map(product_map)
df["appeal_composite"] = df[["appeal_1", "appeal_2", "appeal_3"]].mean(axis=1)
df["mc_composite"] = df[["mc_lonely", "mc_disconnected", "mc_isolated"]].mean(axis=1)
df["condition_code"] = (df["condition"] == "loneliness").astype(int)
df["age_group"] = pd.cut(df["age"], bins=[17, 25, 35, 50, 80], labels=["18-25", "26-35", "36-50", "51+"])
df["income_group"] = pd.cut(df["income"], bins=[0, 35000, 65000, 100000, 500000],
                             labels=["Low", "Medium", "High", "Very High"])

os.makedirs(f"{STUDY_DIR}/results/figures", exist_ok=True)
results = {}

print(f"=== ANALYSIS: {STUDY_DIR} ===")
print(f"Observations: {len(df)} | Participants: {df['participant_id'].nunique()} | Products: {df['product_id'].nunique()}")

# ============================================================
# 1. ATTENTION CHECK
# ============================================================
attn_pass = (df["attention_check"] == 4).sum()
results["attention_check_pass_rate"] = round(attn_pass / len(df), 3)
print(f"\n--- Attention check pass rate: {attn_pass}/{len(df)} ({results['attention_check_pass_rate']*100:.1f}%)")

# ============================================================
# 2. MANIPULATION CHECK
# ============================================================
print("\n--- Manipulation Check ---")
mc_p = df.groupby(["participant_id", "condition"])["mc_composite"].mean().reset_index()
mc_l = mc_p[mc_p.condition == "loneliness"]["mc_composite"]
mc_c = mc_p[mc_p.condition == "control"]["mc_composite"]
mc_t = stats.ttest_ind(mc_l, mc_c)
mc_d = (mc_l.mean() - mc_c.mean()) / np.sqrt((mc_l.std()**2 + mc_c.std()**2) / 2)
results["manipulation_check"] = {
    "loneliness_mean": round(mc_l.mean(), 3), "loneliness_sd": round(mc_l.std(), 3),
    "control_mean": round(mc_c.mean(), 3), "control_sd": round(mc_c.std(), 3),
    "t": round(mc_t.statistic, 3), "p": round(mc_t.pvalue, 6), "cohens_d": round(mc_d, 3)
}
print(f"Loneliness: M={mc_l.mean():.2f} SD={mc_l.std():.2f} | Control: M={mc_c.mean():.2f} SD={mc_c.std():.2f}")
print(f"t({len(mc_l)+len(mc_c)-2})={mc_t.statistic:.3f}, p={mc_t.pvalue:.4f}, d={mc_d:.3f}")

# ============================================================
# 3. INTEGRATED REGRESSION (KEY PRIMARY TEST)
# ============================================================
print("\n--- Integrated Regression (KEY TEST) ---")
print("WTP ~ condition + age + gender + income + education + employment + relationship_status + product FE")
print("(Cluster-robust SEs grouped by participant)\n")
try:
    int_model = smf.ols(
        "wtp ~ condition_code + age + C(gender) + income + "
        "C(education) + C(employment) + C(relationship_status) + C(product_id)",
        data=df.dropna(subset=["wtp"])
    ).fit(cov_type="cluster", cov_kwds={"groups": df.dropna(subset=["wtp"])["participant_id"]})
    print(int_model.summary())
    b = int_model.params.get("condition_code", np.nan)
    se = int_model.bse.get("condition_code", np.nan)
    t = int_model.tvalues.get("condition_code", np.nan)
    p = int_model.pvalues.get("condition_code", np.nan)
    results["integrated_regression"] = {
        "test_type": "OLS_with_controls_clustered_SE",
        "condition_coef": round(b, 3), "condition_se": round(se, 3),
        "condition_t": round(t, 3), "condition_p": round(p, 6),
        "r_squared": round(int_model.rsquared, 4),
        "adj_r_squared": round(int_model.rsquared_adj, 4),
        "n_obs": int(int_model.nobs),
        "controls": ["age", "gender", "income", "education", "employment", "relationship_status", "product_id"],
        "full_summary": str(int_model.summary()),
    }
    print(f"\n>>> CONDITION EFFECT: beta={b:.3f}, SE={se:.3f}, t={t:.3f}, p={p:.4f}")
    print(f">>> R2={int_model.rsquared:.4f}, Adj R2={int_model.rsquared_adj:.4f}")
except Exception as e:
    print(f"Integrated regression failed: {e}")
    results["integrated_regression"] = {"error": str(e)}

# ============================================================
# 4. SIMPLE T-TEST ON WTP (participant-level means)
# ============================================================
print("\n--- WTP t-test (participant-level means) ---")
wtp_p = df.groupby(["participant_id", "condition"])["wtp"].mean().reset_index()
wtp_l = wtp_p[wtp_p.condition == "loneliness"]["wtp"].dropna()
wtp_c = wtp_p[wtp_p.condition == "control"]["wtp"].dropna()
wtp_t = stats.ttest_ind(wtp_l, wtp_c)
wtp_d = (wtp_l.mean() - wtp_c.mean()) / np.sqrt((wtp_l.std()**2 + wtp_c.std()**2) / 2)
sem_diff = np.sqrt(wtp_l.var()/len(wtp_l) + wtp_c.var()/len(wtp_c))
ci = [wtp_l.mean()-wtp_c.mean() - 1.96*sem_diff, wtp_l.mean()-wtp_c.mean() + 1.96*sem_diff]
results["wtp_ttest"] = {
    "loneliness_mean": round(wtp_l.mean(), 3), "loneliness_sd": round(wtp_l.std(), 3),
    "control_mean": round(wtp_c.mean(), 3), "control_sd": round(wtp_c.std(), 3),
    "t": round(wtp_t.statistic, 3), "p": round(wtp_t.pvalue, 6),
    "cohens_d": round(wtp_d, 3), "ci_95": [round(ci[0], 3), round(ci[1], 3)]
}
print(f"Loneliness: M=${wtp_l.mean():.2f} SD=${wtp_l.std():.2f} | Control: M=${wtp_c.mean():.2f} SD=${wtp_c.std():.2f}")
print(f"t={wtp_t.statistic:.3f}, p={wtp_t.pvalue:.4f}, d={wtp_d:.3f}, 95%CI=[{ci[0]:.2f}, {ci[1]:.2f}]")

# ============================================================
# 5. PURCHASE INTENTION & APPEAL (secondary DVs)
# ============================================================
for dv_name, dv_col in [("purchase_intention", "purchase_intention"), ("appeal", "appeal_composite")]:
    p_val = df.groupby(["participant_id", "condition"])[dv_col].mean().reset_index()
    l_v = p_val[p_val.condition == "loneliness"][dv_col].dropna()
    c_v = p_val[p_val.condition == "control"][dv_col].dropna()
    tt = stats.ttest_ind(l_v, c_v)
    d = (l_v.mean() - c_v.mean()) / np.sqrt((l_v.std()**2 + c_v.std()**2) / 2)
    results[f"{dv_name}_ttest"] = {
        "loneliness_mean": round(l_v.mean(), 3), "control_mean": round(c_v.mean(), 3),
        "t": round(tt.statistic, 3), "p": round(tt.pvalue, 6), "cohens_d": round(d, 3)
    }
    print(f"\n--- {dv_name} --- Loneliness M={l_v.mean():.2f} vs Control M={c_v.mean():.2f}, t={tt.statistic:.3f}, p={tt.pvalue:.4f}, d={d:.3f}")

# ============================================================
# 6. PER-PRODUCT WTP BREAKDOWN
# ============================================================
print("\n--- Per-Product WTP ---")
per_product = []
for prod_id in sorted(df["product_id"].unique()):
    sub = df[df.product_id == prod_id]
    pname = product_map[prod_id]
    l_wtp = sub[sub.condition == "loneliness"]["wtp"].dropna()
    c_wtp = sub[sub.condition == "control"]["wtp"].dropna()
    if len(l_wtp) < 2 or len(c_wtp) < 2:
        continue
    tt = stats.ttest_ind(l_wtp, c_wtp)
    d = (l_wtp.mean() - c_wtp.mean()) / np.sqrt((l_wtp.std()**2 + c_wtp.std()**2) / 2)
    per_product.append({
        "product_id": int(prod_id), "product_name": pname,
        "loneliness_mean": round(l_wtp.mean(), 2), "loneliness_sd": round(l_wtp.std(), 2),
        "control_mean": round(c_wtp.mean(), 2), "control_sd": round(c_wtp.std(), 2),
        "t": round(tt.statistic, 3), "p": round(tt.pvalue, 6), "cohens_d": round(d, 3)
    })
    sig = "*" if tt.pvalue < 0.05 else ""
    print(f"  {pname}: Lonely=${l_wtp.mean():.2f} vs Control=${c_wtp.mean():.2f}, t={tt.statistic:.2f}, p={tt.pvalue:.4f}, d={d:.2f} {sig}")
results["per_product_wtp"] = per_product

# ============================================================
# 7. INTERACTION: PRODUCT x CONDITION
# ============================================================
try:
    int_anova = smf.ols("wtp ~ C(condition) * C(product_id)", data=df).fit()
    anova_t = sm.stats.anova_lm(int_anova, typ=2)
    key = "C(condition):C(product_id)"
    results["product_x_condition"] = {
        "F": round(anova_t.loc[key, "F"], 3),
        "p": round(anova_t.loc[key, "PR(>F)"], 6)
    }
    print(f"\n--- Product x Condition: F={results['product_x_condition']['F']:.3f}, p={results['product_x_condition']['p']:.4f}")
except Exception as e:
    results["product_x_condition"] = {"error": str(e)}

# ============================================================
# 8. DEMOGRAPHIC BREAKDOWNS
# ============================================================
demo_results = {}
for var in ["gender", "age_group", "income_group", "relationship_status"]:
    var_res = {}
    for grp in df[var].dropna().unique():
        sub = df[df[var] == grp]
        l_s = sub[sub.condition == "loneliness"]["wtp"].dropna()
        c_s = sub[sub.condition == "control"]["wtp"].dropna()
        if len(l_s) < 2 or len(c_s) < 2:
            continue
        tt = stats.ttest_ind(l_s, c_s)
        d = (l_s.mean() - c_s.mean()) / np.sqrt((l_s.std()**2 + c_s.std()**2) / 2)
        var_res[str(grp)] = {
            "loneliness_mean": round(l_s.mean(), 2), "control_mean": round(c_s.mean(), 2),
            "t": round(tt.statistic, 3), "p": round(tt.pvalue, 4), "d": round(d, 3),
            "n": len(l_s) + len(c_s)
        }
    demo_results[var] = var_res
results["demographic_breakdowns"] = demo_results

# ============================================================
# 9. MEDIATION (path a and b)
# ============================================================
print("\n--- Mediation Analysis ---")
mediation = {}
for med_name, med_col in [("nostalgia", "med_nostalgia"), ("comfort", "med_comfort"), ("surrogate", "med_surrogate")]:
    med_p = df.groupby(["participant_id", "condition"]).agg(
        mediator=(med_col, "mean"), wtp=("wtp", "mean")
    ).reset_index()
    med_p["condition_code"] = (med_p["condition"] == "loneliness").astype(int)
    pa = smf.ols("mediator ~ condition_code", data=med_p).fit()
    pbc = smf.ols("wtp ~ condition_code + mediator", data=med_p).fit()
    a = pa.params["condition_code"]
    b = pbc.params["mediator"]
    mediation[med_name] = {
        "path_a": round(a, 3), "path_a_p": round(pa.pvalues["condition_code"], 4),
        "path_b": round(b, 3), "path_b_p": round(pbc.pvalues["mediator"], 4),
        "direct_effect": round(pbc.params["condition_code"], 3),
        "direct_p": round(pbc.pvalues["condition_code"], 4),
        "indirect": round(a * b, 3)
    }
    print(f"  {med_name}: a={a:.3f}(p={pa.pvalues['condition_code']:.3f}), b={b:.3f}(p={pbc.pvalues['mediator']:.3f}), indirect={a*b:.3f}")
results["mediation"] = mediation

# ============================================================
# DESCRIPTIVES TABLE
# ============================================================
desc = df.groupby("condition").agg(
    wtp_mean=("wtp", "mean"), wtp_sd=("wtp", "std"),
    pi_mean=("purchase_intention", "mean"),
    appeal_mean=("appeal_composite", "mean"),
    n_obs=("participant_id", "count")
).round(3)
results["descriptives"] = desc.to_dict(orient="index")
print(f"\n--- Descriptives ---\n{desc.to_string()}")

# ============================================================
# VISUALIZATIONS
# ============================================================
# Figure 1: Main effect (WTP by condition)
fig, ax = plt.subplots(figsize=(5, 4))
sns.barplot(data=wtp_p, x="condition", y="wtp", order=["control", "loneliness"],
            errorbar="ci", palette=["#4ECDC4", "#FF6B6B"], ax=ax)
ax.set_xlabel("Condition"); ax.set_ylabel("Mean WTP ($)")
ax.set_title("WTP by Condition")
plt.tight_layout()
fig.savefig(f"{STUDY_DIR}/results/figures/main_effect_wtp.png", dpi=300, bbox_inches="tight")
plt.close()

# Figure 2: Per-product breakdown
fig, ax = plt.subplots(figsize=(11, 4))
sns.barplot(data=df, x="product_name", y="wtp", hue="condition",
            hue_order=["control", "loneliness"], errorbar="ci",
            palette=["#4ECDC4", "#FF6B6B"], ax=ax)
ax.set_xlabel("Product"); ax.set_ylabel("WTP ($)")
ax.set_title("WTP by Product and Condition")
plt.xticks(rotation=15, ha="right"); ax.legend(title="Condition")
plt.tight_layout()
fig.savefig(f"{STUDY_DIR}/results/figures/wtp_by_product.png", dpi=300, bbox_inches="tight")
plt.close()

# Figure 3: Manipulation check
fig, ax = plt.subplots(figsize=(5, 4))
sns.barplot(data=mc_p, x="condition", y="mc_composite", order=["control", "loneliness"],
            errorbar="ci", palette=["#4ECDC4", "#FF6B6B"], ax=ax)
ax.set_xlabel("Condition"); ax.set_ylabel("Felt Loneliness (1-7)")
ax.set_title("Manipulation Check")
plt.tight_layout()
fig.savefig(f"{STUDY_DIR}/results/figures/manipulation_check.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nFigures saved to {STUDY_DIR}/results/figures/")

# ============================================================
# SAVE RESULTS
# ============================================================
with open(f"{STUDY_DIR}/results/analysis_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)

print(f"Analysis saved to {STUDY_DIR}/results/analysis_results.json")
print("\n=== ANALYSIS COMPLETE ===")
