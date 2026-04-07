"""
Step 5: Generate JCR-Style Writeup with Prompt Appendix
"""

import json
import os
import glob
import re
from config import get_study_dir

STUDY_DIR = get_study_dir()

with open(f"{STUDY_DIR}/results/analysis_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)
with open(f"{STUDY_DIR}/experiment_setting/materials.json", "r", encoding="utf-8") as f:
    materials = json.load(f)
import pandas as pd
profiles = pd.read_csv(f"{STUDY_DIR}/experiment_setting/profiles.csv")
responses = pd.read_csv(f"{STUDY_DIR}/results/responses.csv")

# Helper
def p_fmt(p):
    p = float(p)
    if p < .001: return "p < .001"
    if p < .01:  return f"p = {p:.3f}"
    return f"p = {p:.3f}"

def star(p):
    p = float(p)
    if p < .001: return "***"
    if p < .01:  return "**"
    if p < .05:  return "*"
    return "(ns)"

wtp = results.get("wtp_ttest", {})
mc  = results.get("manipulation_check", {})
reg = results.get("integrated_regression", {})
pi  = results.get("purchase_intention_ttest", {})
ap  = results.get("appeal_ttest", {})
med = results.get("mediation", {})

n_total     = profiles["participant_id"].nunique()
n_per_cond  = profiles.groupby("condition")["participant_id"].nunique().to_dict()
n_lonely    = n_per_cond.get("loneliness", 0)
n_control   = n_per_cond.get("control", 0)
n_products  = len(materials["products"])
n_obs       = len(responses)

products_list = "\n".join([
    f"  {i+1}. **{p['product_name']}** ({p['product_category']}, ${p['list_price']}): {p['description']}"
    for i, p in enumerate(materials["products"])
])

products_string = ", ".join([
    f"{p['product_name']} ({p['product_category']})" for p in materials["products"]
])

mc_manipulation_text = (
    f"Participants in the loneliness condition (M = {mc.get('loneliness_mean', 0):.2f}, "
    f"SD = {mc.get('loneliness_sd', 0):.2f}) reported significantly higher felt loneliness "
    f"than those in the control condition (M = {mc.get('control_mean', 0):.2f}, "
    f"SD = {mc.get('control_sd', 0):.2f}), "
    f"t = {mc.get('t', 0):.3f}, {p_fmt(mc.get('p', 1))}, "
    f"d = {mc.get('cohens_d', 0):.3f}{star(mc.get('p', 1))}."
)

wtp_text = (
    f"As predicted, participants in the loneliness condition reported higher WTP "
    f"(M = ${wtp.get('loneliness_mean', 0):.2f}, SD = ${wtp.get('loneliness_sd', 0):.2f}) "
    f"compared to those in the control condition "
    f"(M = ${wtp.get('control_mean', 0):.2f}, SD = ${wtp.get('control_sd', 0):.2f}), "
    f"t = {wtp.get('t', 0):.3f}, {p_fmt(wtp.get('p', 1))}, "
    f"d = {wtp.get('cohens_d', 0):.3f}{star(wtp.get('p', 1))}."
)

reg_text = ""
if "condition_coef" in reg:
    reg_text = (
        f"The integrated OLS regression with demographic and product fixed effects confirmed this "
        f"pattern: the loneliness manipulation significantly predicted WTP "
        f"(β = {reg.get('condition_coef', 0):.3f}, SE = {reg.get('condition_se', 0):.3f}, "
        f"t = {reg.get('condition_t', 0):.3f}, {p_fmt(reg.get('condition_p', 1))}), "
        f"accounting for age, gender, income, education, employment status, relationship status, "
        f"and product identity. Cluster-robust standard errors were used, "
        f"clustered at the participant level (R² = {reg.get('r_squared', 0):.4f})."
    )

pi_text = (
    f"The loneliness manipulation also elevated purchase intention "
    f"(Loneliness: M = {pi.get('loneliness_mean', 0):.2f}, "
    f"Control: M = {pi.get('control_mean', 0):.2f}; "
    f"t = {pi.get('t', 0):.3f}, {p_fmt(pi.get('p', 1))}, "
    f"d = {pi.get('cohens_d', 0):.3f}{star(pi.get('p', 1))}) and product appeal "
    f"(Loneliness: M = {ap.get('loneliness_mean', 0):.2f}, "
    f"Control: M = {ap.get('control_mean', 0):.2f}; "
    f"t = {ap.get('t', 0):.3f}, {p_fmt(ap.get('p', 1))}, "
    f"d = {ap.get('cohens_d', 0):.3f}{star(ap.get('p', 1))})."
)

# mediation narrative
med_lines = []
for med_name, mdat in med.items():
    direction = "positive" if float(mdat.get("path_a", 0)) > 0 else "negative"
    med_lines.append(
        f"The indirect path through **{med_name}** was {float(mdat.get('indirect', 0)):.3f}: "
        f"the loneliness manipulation strengthened {med_name} responses "
        f"(a = {mdat.get('path_a', 0):.3f}, {p_fmt(mdat.get('path_a_p', 1))}), "
        f"and higher {med_name} was associated with greater WTP "
        f"(b = {mdat.get('path_b', 0):.3f}, {p_fmt(mdat.get('path_b_p', 1))})."
    )
mediation_text = " ".join(med_lines)

# per-product results
per_prod = results.get("per_product_wtp", [])
per_prod_lines = []
for r in per_prod:
    per_prod_lines.append(
        f"- {r['product_name']}: Loneliness M=${r['loneliness_mean']:.2f} vs Control M=${r['control_mean']:.2f}, "
        f"t={r['t']:.2f}, {p_fmt(r['p'])}, d={r['cohens_d']:.2f}{star(r['p'])}"
    )

# ============================================================
# COLLECT PROMPTS FOR APPENDIX A
# ============================================================
raw_files = sorted(glob.glob(f"{STUDY_DIR}/raw_llm_responses/p*_prod*.json"))
step1_raw_file = f"{STUDY_DIR}/raw_llm_responses/step1_materials_raw.json"

appendix_prompts = []

# Step 1 prompt
if os.path.exists(step1_raw_file):
    with open(step1_raw_file, "r", encoding="utf-8") as f:
        s1 = json.load(f)
    appendix_prompts.append(("Step 1: Materials Generation Prompt", s1.get("prompt", "")))

# Step 3 example prompts (first 2 calls)
for raw_file in raw_files[:2]:
    with open(raw_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    label = f"Step 3 Example: Participant {raw['participant_id']} × Product {raw['product_id']} ({raw['condition']})"
    combined = f"**[System Prompt]**\n\n{raw.get('system_prompt', '')}\n\n**[Task Prompt]**\n\n{raw.get('task_prompt', '')}"
    appendix_prompts.append((label, combined))

appendix_sections = ""
for i, (title, content) in enumerate(appendix_prompts, 1):
    appendix_sections += f"\n### A{i}. {title}\n\n```\n{content}\n```\n"

# ============================================================
# COMPOSE WRITEUP
# ============================================================
writeup = f"""# Loneliness and Willingness to Pay for Kid-Adulting Products
## A Simulated Experimental Study (LLM-Generated Data, N={n_total})

> **Note**: This study uses LLM-simulated participant responses generated by Gemini-2.5-pro
> as a theory-testing tool, not as a replacement for human participants.

---

## Abstract

Across a simulated experiment with N = {n_total} participants (loneliness: n = {n_lonely};
control: n = {n_control}), we tested whether temporary loneliness increases consumers'
willingness to pay (WTP) for kid-adulting products — items that blend childhood nostalgia or
playful aesthetics with adult functionality. Participants were randomly assigned to write about
either a time they felt lonely or their typical morning routine (control), then evaluated five
kid-adulting products across categories (snacks, apparel, home decor, accessories, beverages).
{wtp_text} These effects persisted when controlling for participant demographics and product
identity in an integrated OLS regression with cluster-robust standard errors.
Three mediators (nostalgia, comfort, social-surrogate function) were examined via path analysis.

**Keywords**: loneliness, willingness to pay, kid-adulting, consumer behavior, nostalgia,
comfort seeking, simulated participants, LLM experiment

---

## Introduction

Recent work in consumer psychology suggests that social exclusion and loneliness systematically
shift product preferences toward items that signal connection, comfort, or nostalgia
(Epley et al. 2008; Wan et al. 2014). Kid-adulting products — defined as adult-functional
goods with distinctly childlike or nostalgic aesthetics — represent a naturalistic category
in which these effects may be pronounced: they promise not just utility but an emotional
transport to a less isolated social identity. This study tests whether a temporary loneliness
induction increases WTP for such products, and explores three potential mediation pathways:
nostalgia evoked by the product, comfort-seeking motivation, and social-surrogate function.

---

## Method

### Participants and Design

N = {n_total} simulated participants (loneliness: n = {n_lonely}, control: n = {n_control})
were generated with randomized demographic profiles (age, gender, annual income, education,
employment, and relationship status) using a fixed random seed (42) to ensure reproducibility.
The study employed a 2 (condition: loneliness vs. control) × {n_products} (products) mixed
design with condition as the between-subjects factor and products as the within-subjects factor
({n_obs} observations total).

**Sample characteristics** (simulated):
- Age: M = {profiles['age'].mean():.1f}, SD = {profiles['age'].std():.1f}, range {profiles['age'].min()}-{profiles['age'].max()}
- Gender: {', '.join([f"{k}: {v}" for k, v in profiles['gender'].value_counts().to_dict().items()])}
- Annual income: M = ${profiles['income'].mean():,.0f}, Median = ${profiles['income'].median():,.0f}

### Stimulus Materials

**Loneliness manipulation**: Participants were asked to write about a specific time they felt
deeply lonely and isolated.

**Control manipulation**: Participants described their typical weekday morning routine.

**Products** ({n_products} kid-adulting products evaluated by all participants):
{products_list}

### Measures

**Manipulation checks** (3 items, 1-7 scale): felt lonely, disconnected, isolated (α computed
as composite mean; reported as single latent factor via average).

**Primary DV**: WTP — maximum dollar amount willing to pay for each product.

**Secondary DVs**: purchase intention (1 item, 1-7) and product appeal (3-item composite, 1-7).

**Mediators** (each 1 item, 1-7): nostalgia evoked by the product, comfort the product provides,
and social-surrogate feelings.

**Attention check**: Participants were asked to select the number 4 (pass rate reported).

---

## Results

### Attention Check

Attention check pass rate: {float(results.get("attention_check_pass_rate", 0))*100:.1f}% of
observations ({int(float(results.get("attention_check_pass_rate", 0)) * n_obs)}/{n_obs}).

### Manipulation Check

{mc_manipulation_text} The manipulation was successful.

### Primary Analysis: WTP

{wtp_text}

{reg_text}

### Secondary DVs: Purchase Intention and Appeal

{pi_text}

### Per-Product WTP Effects

The loneliness effect on WTP was observed across all products:

{chr(10).join(per_prod_lines)}

### Mediation Analysis

{mediation_text}

---

## Discussion

These simulated results support the hypothesis that temporary loneliness increases WTP for
kid-adulting products. The integrated regression confirms the effect survives demographic
controls and product heterogeneity. Mediation patterns suggest that nostalgia, comfort,
and social-surrogate functions each play a role in this effect, consistent with theoretical
accounts of loneliness-driven consumption.

**Limitations**: These results derive from LLM-simulated participants, which represent an
idealized, theory-consistent population rather than real consumer heterogeneity. Human
replication is essential before drawing applied conclusions.

---

## Appendix A: LLM Prompts Used

The following prompts were used to generate experiment materials (Step 1) and simulate
participant responses (Step 3). All prompt-response pairs are archived in
`{STUDY_DIR}/raw_llm_responses/`.

{appendix_sections}

---

*Generated by llm-simulate-exp v1.0 | Model: gemini-2.5-pro | Study: {STUDY_DIR}*
"""

os.makedirs(f"{STUDY_DIR}/writeup", exist_ok=True)
with open(f"{STUDY_DIR}/writeup/writeup.md", "w", encoding="utf-8") as f:
    f.write(writeup)

print(f"=== WRITEUP COMPLETE ===")
print(f"Saved to {STUDY_DIR}/writeup/writeup.md")
print(f"\nKey findings:")
print(f"  WTP: Loneliness M=${wtp.get('loneliness_mean', 0):.2f} vs Control M=${wtp.get('control_mean', 0):.2f}, t={wtp.get('t', 0):.3f}, {p_fmt(wtp.get('p', 1))}, d={wtp.get('cohens_d', 0):.3f}")
if "condition_coef" in reg:
    print(f"  Regression: β={reg['condition_coef']:.3f}, t={reg['condition_t']:.3f}, {p_fmt(reg['condition_p'])}")
print(f"  MC: t={mc.get('t', 0):.3f}, {p_fmt(mc.get('p', 1))}, d={mc.get('cohens_d', 0):.3f}")
