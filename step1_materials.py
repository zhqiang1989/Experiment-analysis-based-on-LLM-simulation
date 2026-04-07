"""
Step 1: Generate Experiment Materials
Hypothesis: Temporary loneliness (vs. neutral control) increases WTP for kid-adulting products.
Design: 2 (loneliness vs. control) between-subjects x 5 kid-adulting products within-subjects.
"""

import json
import os
import glob
import sys
from vertexai.generative_models import GenerationConfig
from config import model

# --- Determine study directory ---
# Accept explicit name via CLI or auto-increment
if len(sys.argv) > 1:
    STUDY_DIR = sys.argv[1]
    print(f"Using specified study directory: {STUDY_DIR}/")
else:
    existing = [d for d in glob.glob("study_*") if os.path.isdir(d)]
    nums = []
    for d in existing:
        try:
            nums.append(int(d.replace("study_", "").split("_v")[0]))
        except ValueError:
            pass
    STUDY_DIR = f"study_{max(nums) + 1}" if nums else "study_1"
    print(f"Creating new study directory: {STUDY_DIR}/")

for subdir in ["experiment_setting", "raw_llm_responses", "results/figures", "writeup"]:
    os.makedirs(f"{STUDY_DIR}/{subdir}", exist_ok=True)

with open("_current_study.txt", "w") as f:
    f.write(STUDY_DIR)

# -----------------------------------------------------------------------
# MATERIALS GENERATION PROMPT
# -----------------------------------------------------------------------
MATERIALS_PROMPT = """You are an expert experimental psychologist designing a consumer behavior experiment.

**Hypothesis**: When people are temporarily induced into a state of loneliness (vs. a neutral control state), they will show higher willingness to pay (WTP) for kid-adulting products — products that blend childhood nostalgia or playful aesthetics with adult functionality.

**Background on Kid-Adulting Products**: These are products marketed to adults that incorporate childhood elements — cartoon characters on coffee mugs, cereal-themed snacks for adults, backpacks with playful designs, novelty stationery, etc. They satisfy adult needs but carry a nostalgic, childlike appeal.

**Task**: Generate complete experiment materials for this study. The design is:
- **Manipulation**: 2 conditions between-subjects (loneliness vs. neutral control writing task)
- **Products**: 5 distinct kid-adulting products across different categories (snacks, apparel, home decor, accessories, beverages) — each participant evaluates ALL 5 products
- **Primary DV**: Willingness to Pay (WTP) in USD (open-ended)
- **Secondary DVs**: Purchase intention (1-7 Likert), product appeal (3 items, 1-7 Likert)
- **Manipulation checks**: 3 items assessing felt loneliness (1-7 Likert)
- **Mediators**: nostalgia evoked by product, comfort-seeking, social surrogate feelings (each 1-7 Likert)
- **Attention check**: 1 item instructing participants to select a specific value

Generate the output as a single valid JSON object with exactly this structure:
{
  "study_name": "Loneliness and WTP for Kid-Adulting Products",
  "hypothesis": "...",
  "design": "2-condition (loneliness vs. control) between-subjects; 5 products within-subjects",
  "manipulation": {
    "loneliness": {
      "label": "loneliness",
      "prompt_text": "[3-5 sentence writing task instruction asking participants to recall and write about a specific time they felt deeply lonely, isolated, and disconnected from others. Should be vivid and emotionally engaging enough to temporarily induce loneliness affect.]"
    },
    "control": {
      "label": "control",
      "prompt_text": "[3-5 sentence writing task instruction asking participants to describe their typical weekday morning routine in neutral, descriptive detail. Should be emotionally flat and not evoke any social emotions.]"
    }
  },
  "products": [
    {
      "product_id": 1,
      "product_name": "...",
      "product_category": "Snacks",
      "list_price": ...,
      "description": "[2-3 sentences describing the product in appealing, marketable terms. Must clearly blend childhood nostalgia or playful aesthetics with adult-appropriate function.]"
    },
    { "product_id": 2, "product_category": "Apparel", ... },
    { "product_id": 3, "product_category": "Home Decor", ... },
    { "product_id": 4, "product_category": "Accessories", ... },
    { "product_id": 5, "product_category": "Beverages", ... }
  ],
  "manipulation_checks": [
    {"item_id": 1, "item_text": "Right now, I feel lonely.", "scale": "1=Strongly Disagree to 7=Strongly Agree", "construct": "loneliness"},
    {"item_id": 2, "item_text": "Right now, I feel disconnected from others.", "scale": "1=Strongly Disagree to 7=Strongly Agree", "construct": "loneliness"},
    {"item_id": 3, "item_text": "Right now, I feel isolated.", "scale": "1=Strongly Disagree to 7=Strongly Agree", "construct": "loneliness"}
  ],
  "dv_measures": {
    "wtp": {"description": "Maximum willingness to pay for the product, in USD (open-ended number)"},
    "purchase_intention": {"item_text": "I would purchase this product.", "scale": "1=Strongly Disagree to 7=Strongly Agree"},
    "appeal": [
      {"item_id": 1, "item_text": "This product appeals to me.", "scale": "1=Strongly Disagree to 7=Strongly Agree"},
      {"item_id": 2, "item_text": "I find this product attractive.", "scale": "1=Strongly Disagree to 7=Strongly Agree"},
      {"item_id": 3, "item_text": "This product is desirable.", "scale": "1=Strongly Disagree to 7=Strongly Agree"}
    ]
  },
  "mediators": {
    "nostalgia": {"item_text": "This product reminds me of my childhood.", "scale": "1=Strongly Disagree to 7=Strongly Agree"},
    "comfort": {"item_text": "This product would make me feel comforted.", "scale": "1=Strongly Disagree to 7=Strongly Agree"},
    "surrogate": {"item_text": "This product would make me feel more connected.", "scale": "1=Strongly Disagree to 7=Strongly Agree"}
  },
  "attention_check": {
    "item_text": "For quality control, please select the number 4 for this item.",
    "correct_answer": 4,
    "scale": "1-7"
  },
  "control_variables": ["age", "gender", "annual_income", "education_level", "employment_status", "relationship_status", "trait_loneliness", "nostalgia_proneness"]
}

Make each product description vivid and specific (real brand-like name, realistic price for the category). Output ONLY valid JSON — no explanation, no markdown fences."""

# -----------------------------------------------------------------------
# CALL THE LLM
# -----------------------------------------------------------------------
print("Generating experiment materials via Gemini...")
response = model.generate_content(
    MATERIALS_PROMPT,
    generation_config=GenerationConfig(
        response_mime_type="application/json",
        temperature=0.7,
    )
)

materials = json.loads(response.text)

# Save raw LLM response
with open(f"{STUDY_DIR}/raw_llm_responses/step1_materials_raw.json", "w", encoding="utf-8") as f:
    json.dump({
        "prompt": MATERIALS_PROMPT,
        "raw_response": response.text,
    }, f, indent=2, ensure_ascii=False)

# Save parsed materials
with open(f"{STUDY_DIR}/experiment_setting/materials.json", "w", encoding="utf-8") as f:
    json.dump(materials, f, indent=2, ensure_ascii=False)

# -----------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------
print(f"\n=== MATERIALS SUMMARY ===")
print(f"Loneliness prompt: {materials['manipulation']['loneliness']['prompt_text'][:100]}...")
print(f"Control prompt:    {materials['manipulation']['control']['prompt_text'][:100]}...")
print(f"\nProducts ({len(materials['products'])}):")
for p in materials["products"]:
    print(f"  {p['product_id']}. {p['product_name']} ({p['product_category']}) - ${p['list_price']}")
print(f"\nManipulation checks: {len(materials['manipulation_checks'])} items")
print(f"Mediators: {list(materials['mediators'].keys())}")
print(f"\nMaterials saved to {STUDY_DIR}/experiment_setting/materials.json")
print(f"Raw response saved to {STUDY_DIR}/raw_llm_responses/step1_materials_raw.json")
