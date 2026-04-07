"""
Shared configuration for LLM experiment simulation pipeline.
Steps 2-5 import auth and study directory from here.
"""

import json
import sys
import os
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel

# --- Service Account ---
# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to your
# service account JSON key path before running. Example (PowerShell):
#   $env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\your-service-account-key.json"
# Or add it to your PowerShell profile for persistence.
SERVICE_ACCOUNT_FILE = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "key/your-service-account-key.json"  # fallback if env var not set
)
with open(SERVICE_ACCOUNT_FILE) as f:
    sa_info = json.load(f)
PROJECT_ID = sa_info["project_id"]
LOCATION = "us-central1"

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
model = GenerativeModel("gemini-2.5-pro")


def get_study_dir():
    """Return study directory from CLI arg or _current_study.txt.

    Usage: python step3_responses.py [study_dir]
    If no arg, reads from _current_study.txt (written by step1).
    """
    if len(sys.argv) > 1:
        d = sys.argv[1]
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Study directory '{d}' does not exist.")
        return d
    if os.path.exists("_current_study.txt"):
        with open("_current_study.txt") as f:
            return f.read().strip()
    raise FileNotFoundError(
        "No study directory found. Pass as CLI arg (e.g. 'python step3_responses.py study_1') "
        "or run step1_materials.py first."
    )
