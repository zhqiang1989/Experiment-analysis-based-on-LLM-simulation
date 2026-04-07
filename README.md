# Experiment Analysis Based on LLM Simulation

A Python pipeline for simulating consumer psychology experiments using LLM-generated participant responses (Google Gemini 2.5 Pro via Vertex AI). The pipeline covers the full research workflow: stimulus generation → participant profiling → response collection → statistical analysis → writeup.

> **Note**: This pipeline uses LLM-simulated participant responses as a theory-testing and prototyping tool, not as a replacement for human participants.

---

## Study 1: Loneliness and WTP for Kid-Adulting Products

**Hypothesis**: Temporary loneliness (vs. neutral control) increases consumers' willingness to pay (WTP) for kid-adulting products — adult-functional goods with childlike or nostalgic aesthetics.

**Design**: 2 (condition: loneliness vs. control) × 5 products mixed design  
**N**: 20 simulated participants (n=10 per condition), 100 observations total  
**Model**: Gemini 2.5 Pro

### Key Results

| Measure | Loneliness | Control | t | p | d |
|---|---|---|---|---|---|
| **Manipulation check** (loneliness felt) | M=5.71 | M=1.71 | 17.88 | <.001 | 7.99*** |
| **WTP** | M=$15.45 | M=$15.30 | 0.14 | .893 | 0.06 (ns) |
| **Product appeal** | M=4.55 | M=3.90 | 2.13 | .048 | 0.95* |
| **Purchase intention** | M=3.82 | M=3.22 | 1.52 | .145 | 0.68 (ns) |

OLS regression with demographic and product controls: β=0.527, SE=0.287, t=1.840, p=.066 (cluster-robust SEs, R²=0.96).

Full writeup: [study_1/writeup/writeup.md](study_1/writeup/writeup.md)

---

## Pipeline

```
step1_materials.py   → Generate experiment stimuli (manipulations + products) via LLM
step2_profiles.py    → Generate randomized participant demographic profiles
step3_responses.py   → Collect LLM responses for each participant × product
step4_analysis.py    → Statistical analysis (t-tests, OLS regression, mediation, figures)
step5_writeup.py     → Generate formatted writeup from results
config.py            → Shared Vertex AI authentication and study directory logic
```

Each step auto-detects the current study directory from `_current_study.txt` (written by step 1) or accepts a CLI argument:

```bash
python step3_responses.py study_1   # operate on a specific study
```

Running step 1 again auto-increments the study directory (`study_2/`, `study_3/`, ...).

---

## Repository Structure

```
├── step1_materials.py
├── step2_profiles.py
├── step3_responses.py
├── step4_analysis.py
├── step5_writeup.py
├── config.py
├── study_1/
│   ├── experiment_setting/
│   │   ├── materials.json       # LLM-generated stimuli
│   │   └── profiles.csv         # Simulated participant profiles
│   ├── raw_llm_responses/       # Per-participant × per-product JSON responses
│   ├── results/
│   │   ├── responses.csv        # Parsed responses (long format)
│   │   ├── analysis_results.json
│   │   └── figures/
│   └── writeup/
│       └── writeup.md
```

---

## Setup

### Requirements

```bash
pip install google-cloud-aiplatform google-auth scipy numpy pandas matplotlib
```

### Authentication

This pipeline uses a Google Cloud service account with Vertex AI access.

1. Create a service account in [Google Cloud Console](https://console.cloud.google.com/) and download the JSON key.
2. Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the key path:

```powershell
# PowerShell — add to your $PROFILE for persistence
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\your-service-account-key.json"
```

```bash
# Bash/zsh
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-service-account-key.json"
```

3. Enable the **Vertex AI API** in your GCP project and ensure the service account has the `roles/aiplatform.user` role.

> **Security**: Never commit your service account JSON key. It is excluded by `.gitignore`.

### Running

```bash
python step1_materials.py   # creates study_1/
python step2_profiles.py
python step3_responses.py   # ~100 API calls; takes several minutes
python step4_analysis.py
python step5_writeup.py
```

---

## Model

- **Model**: `gemini-2.5-pro` (via Google Vertex AI, region: `us-central1`)
- **SDK**: `google-cloud-aiplatform` (`vertexai`) — do NOT use `google-generativeai` (region restrictions apply)
