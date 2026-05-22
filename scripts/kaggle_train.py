"""
Kaggle Notebook training script for orbit_war.

Paste each section as a separate notebook cell, or run top-to-bottom.
Requires: GPU accelerator enabled + WANDB_API_KEY added to Kaggle Secrets.

Steps to set up on Kaggle:
  1. New notebook → Settings → Accelerator: GPU P100
  2. Add-ons → Secrets → Add secret: WANDB_API_KEY = <your key>
  3. Add-ons → Secrets → enable "WANDB_API_KEY" for this notebook
  4. Set REPO_URL below to your GitHub repo
  5. Run all cells
"""

# ── Cell 1: Clone repo ────────────────────────────────────────────────────────
REPO_URL = "https://github.com/SCIERke/Orbit_War.git"
REPO_DIR = "orbit_war"

import subprocess, os

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)

os.chdir(REPO_DIR)
print("Working dir:", os.getcwd())

# ── Cell 2: Install dependencies ──────────────────────────────────────────────
# kaggle-environments>=1.29.1 required — orbit_wars env bundled from that version
subprocess.run([
    "pip", "install", "-q",
    "kaggle-environments==1.29.1",
    "sb3-contrib", "wandb", "python-dotenv", "gymnasium",
], check=True)

# ── Cell 3: Secrets → env vars ────────────────────────────────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
    print("WANDB_API_KEY loaded from Kaggle Secrets")
except Exception as e:
    print(f"Could not load secrets ({e}). Set WANDB_API_KEY manually if needed.")

# ── Cell 4: Verify GPU ────────────────────────────────────────────────────────
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# ── Cell 5: Train ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.getcwd())

from agent.ppo import PPOAgent

ppo_agent = PPOAgent(seed=42)
run_path = ppo_agent.learn(total_timesteps=3_000_000, run_dir="/kaggle/working/runs")

# ── Cell 6: Performance test ──────────────────────────────────────────────────
from tests.performance_test import run_performance_test

win_rate = run_performance_test(
    model_path=os.path.join(run_path, "model"),
    n_games=20,
)
print(f"\nFinal win rate: {win_rate:.0%}")

# ── Cell 7: Save model to /kaggle/working (auto-saved as output dataset) ──────
import shutil

model_zip = os.path.join(run_path, "model.zip")
dest = "/kaggle/working/trained_model.zip"
shutil.copy(model_zip, dest)
print(f"Model saved to {dest}")
print("Download via: Kaggle notebook → Data → Output → trained_model.zip")
