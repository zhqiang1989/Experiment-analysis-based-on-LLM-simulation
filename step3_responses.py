"""
Step 3: Simulate Participant Responses
20 participants x 5 products = 100 LLM calls.
Each participant does writing task (loneliness or control), then rates all 5 products.
"""

import json
import os
import time
import pandas as pd
from vertexai.generative_models import GenerationConfig
from config import model, get_study_dir

STUDY_DIR = get_study_dir()

# --- Load data ---
with open(f"{STUDY_DIR}/experiment_setting/materials.json", "r", encoding="utf-8") as f:
    materials = json.load(f)
profiles = pd.read_csv(f"{STUDY_DIR}/experiment_setting/profiles.csv")

RAW_DIR = f"{STUDY_DIR}/raw_llm_responses"
RESPONSES_FILE = f"{STUDY_DIR}/results/responses.csv"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(f"{STUDY_DIR}/results", exist_ok=True)

DV_KEYS = [
    "mc_lonely", "mc_disconnected", "mc_isolated",
    "wtp", "purchase_intention",
    "appeal_1", "appeal_2", "appeal_3",
    "attention_check",
    "med_nostalgia", "med_comfort", "med_surrogate",
]


# -----------------------------------------------------------------------
# PROMPT BUILDERS
# -----------------------------------------------------------------------
def build_system_prompt(profile):
    edu_map = {
        "high_school": "a high school diploma",
        "bachelors": "a bachelor's degree",
        "masters": "a master's degree",
        "phd": "a PhD",
    }
    edu_label = edu_map.get(profile["education"], profile["education"])
    return (
        f"You are participating in a research study as a simulated participant. "
        f"Respond as if you are a real person with these characteristics: "
        f"Age {profile['age']}, {profile['gender']}, annual income ${int(profile['income']):,}, "
        f"education: {edu_label}, employment: {profile['employment'].replace('_', ' ')}, "
        f"relationship status: {profile['relationship_status'].replace('_', ' ')}. "
        f"Respond naturally and authentically. Do not break character or mention you are an AI."
    )


def build_task_prompt(condition, product, materials):
    writing_task = materials["manipulation"][condition]["prompt_text"]
    mc = materials["manipulation_checks"]
    dv = materials["dv_measures"]
    med = materials["mediators"]
    ac = materials["attention_check"]

    prompt = f"""Earlier in this study, you were asked to complete a short writing task. The instructions were:

"{writing_task}"

You completed that writing task. Now please evaluate the following product:

---
Product: {product['product_name']} ({product['product_category']})
List price: ${product['list_price']}
Description: {product['description']}
---

Please answer each question honestly based on how you feel right now.

Manipulation check (how you feel RIGHT NOW, after the writing task):
- mc_lonely: "{mc[0]['item_text']}" ({mc[0]['scale']})
- mc_disconnected: "{mc[1]['item_text']}" ({mc[1]['scale']})
- mc_isolated: "{mc[2]['item_text']}" ({mc[2]['scale']})

About this product:
- wtp: {dv['wtp']['description']} (give a number in USD, e.g. 12.50)
- purchase_intention: "{dv['purchase_intention']['item_text']}" ({dv['purchase_intention']['scale']})
- appeal_1: "{dv['appeal'][0]['item_text']}" ({dv['appeal'][0]['scale']})
- appeal_2: "{dv['appeal'][1]['item_text']}" ({dv['appeal'][1]['scale']})
- appeal_3: "{dv['appeal'][2]['item_text']}" ({dv['appeal'][2]['scale']})
- med_nostalgia: "{med['nostalgia']['item_text']}" ({med['nostalgia']['scale']})
- med_comfort: "{med['comfort']['item_text']}" ({med['comfort']['scale']})
- med_surrogate: "{med['surrogate']['item_text']}" ({med['surrogate']['scale']})
- attention_check: "{ac['item_text']}" (1-7, correct answer is {ac['correct_answer']})

Respond ONLY with valid JSON:
{{
  "mc_lonely": <1-7>,
  "mc_disconnected": <1-7>,
  "mc_isolated": <1-7>,
  "wtp": <number>,
  "purchase_intention": <1-7>,
  "appeal_1": <1-7>,
  "appeal_2": <1-7>,
  "appeal_3": <1-7>,
  "attention_check": <1-7>,
  "med_nostalgia": <1-7>,
  "med_comfort": <1-7>,
  "med_surrogate": <1-7>
}}"""
    return prompt


# -----------------------------------------------------------------------
# SIMULATE ONE CALL (with retries)
# -----------------------------------------------------------------------
def simulate_one(profile, condition, product, max_retries=3):
    system_prompt = build_system_prompt(profile)
    task_prompt = build_task_prompt(condition, product, materials)
    pid = int(profile["participant_id"])
    prod_id = int(product["product_id"])

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                [system_prompt, task_prompt],
                generation_config=GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.8,
                )
            )
            raw_text = response.text
            data = json.loads(raw_text)

            # Save raw prompt + response
            raw_file = os.path.join(RAW_DIR, f"p{pid}_prod{prod_id}.json")
            with open(raw_file, "w", encoding="utf-8") as f:
                json.dump({
                    "participant_id": pid,
                    "product_id": prod_id,
                    "condition": condition,
                    "system_prompt": system_prompt,
                    "task_prompt": task_prompt,
                    "raw_response": raw_text,
                    "parsed_response": data,
                }, f, indent=2, ensure_ascii=False)

            # Clamp Likert ranges
            for key in DV_KEYS:
                if key != "wtp" and key in data and data[key] is not None:
                    data[key] = max(1, min(7, int(data[key])))
            if "wtp" in data and data["wtp"] is not None:
                data["wtp"] = max(0.0, float(data["wtp"]))

            return data
        except Exception as e:
            print(f"    Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))

    return None


# -----------------------------------------------------------------------
# RESUME LOGIC
# -----------------------------------------------------------------------
completed_pairs = set()
all_responses = []

if os.path.exists(RESPONSES_FILE):
    existing_df = pd.read_csv(RESPONSES_FILE)
    all_responses = existing_df.to_dict("records")
    for row in all_responses:
        completed_pairs.add((int(row["participant_id"]), int(row["product_id"])))
    print(f"Resuming: {len(completed_pairs)} responses already collected.")
else:
    print("Starting fresh.")

# -----------------------------------------------------------------------
# MAIN SIMULATION LOOP
# -----------------------------------------------------------------------
total_calls = len(profiles) * len(materials["products"])
call_num = len(completed_pairs)

print(f"\nSimulating {total_calls} participant x product responses...")
print(f"({len(completed_pairs)} done, {total_calls - len(completed_pairs)} remaining)\n")

for _, profile in profiles.iterrows():
    for product in materials["products"]:
        pair = (int(profile["participant_id"]), int(product["product_id"]))
        if pair in completed_pairs:
            continue

        call_num += 1
        pid = profile["participant_id"]
        prod_id = product["product_id"]
        cond = profile["condition"]

        print(f"[{call_num}/{total_calls}] P{pid} x Prod{prod_id} ({cond})...", end=" ", flush=True)

        try:
            result = simulate_one(profile, cond, product)
        except Exception as e:
            print(f"OUTER ERROR: {e}")
            result = None

        if result is None:
            print("FAILED — filling with None")
            result = {k: None for k in DV_KEYS}

        result["participant_id"] = int(pid)
        result["product_id"] = int(prod_id)
        result["condition"] = cond
        all_responses.append(result)

        # Incremental save
        pd.DataFrame(all_responses).to_csv(RESPONSES_FILE, index=False)

        wtp_str = f"${result['wtp']:.2f}" if result.get("wtp") is not None else "N/A"
        print(f"WTP={wtp_str}, PI={result.get('purchase_intention', 'N/A')}")

        time.sleep(1)

# -----------------------------------------------------------------------
# FINAL SUMMARY
# -----------------------------------------------------------------------
df = pd.DataFrame(all_responses)
print(f"\n=== RESPONSE SUMMARY ({STUDY_DIR}) ===")
print(f"Total: {len(df)} | Failed: {df['wtp'].isna().sum()}")
print(f"\nWTP by condition:")
print(df.groupby("condition")["wtp"].mean().round(2).to_string())
print(f"\nPurchase intention by condition:")
print(df.groupby("condition")["purchase_intention"].mean().round(2).to_string())
print(f"\nAttention check pass rate (correct=4): {(df['attention_check'] == 4).sum()}/{len(df)}")
print(f"\nSaved to {RESPONSES_FILE}")
print(f"Raw responses in {RAW_DIR}/")
