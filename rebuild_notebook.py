import nbformat as nbf
import json
import os

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("""# 🌩️ CloudSRE v2 — Multi-Tier GRPO Training on Colab
**OpenEnv India Hackathon 2026 Finale | HF Space | GitHub**

Train our 1.5B parameter SRE agent using GRPO with HF TRL. The agent learns to diagnose and fix cascading microservice outages, outsmarting a 72B Adversarial Designer.

| Component | Detail |
| :--- | :--- |
| **Environment** | HF Space (16-service asynchronous simulated environment) |
| **Training** | This Colab notebook (T4 / A100 GPU) |
| **Model** | `Qwen/Qwen2.5-1.5B` + LoRA |
| **Framework** | HF TRL v0.29+ GRPO with vLLM backend |

---

## 1. Install Dependencies
Install the required packages for TRL, vLLM, and visualization."""),
    nbf.v4.new_code_cell("""!pip install -q \\
    "trl[vllm]>=0.29.0" \\
    "vllm>=0.11.0" \\
    "peft" \\
    "transformers" \\
    "datasets" \\
    "huggingface_hub>=0.20.0" \\
    "matplotlib" \\
    "numpy" \\
    "pandas" \\
    "seaborn"
"""),
    nbf.v4.new_markdown_cell("""## 2. Configuration & Secrets
Set the HF Space URL, model, and hyperparameters. Add `HF_TOKEN` to Colab Secrets (the 🔑 key icon in the left sidebar)."""),
    nbf.v4.new_code_cell("""import os

# --- HuggingFace token ---
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except (ImportError, KeyError, ModuleNotFoundError):
    if "HF_TOKEN" not in os.environ:
        print("WARNING: HF_TOKEN not found. Set it in Colab Secrets or as env var.")

# --- Environment server (HF Space) ---
# Format: https://{username}-{space-name}.hf.space
ENV_URL = "https://dardrax-cloudsre-environment.hf.space"

# --- Model ---
MODEL_ID = "Qwen/Qwen2.5-1.5B"
HUB_REPO = "DarDrax/cloudsre-1.5B-FINAL"

# --- Training hyperparameters ---
NUM_EPISODES = 50
NUM_GENERATIONS = 4
MAX_TURNS = 15

print(f"Environment : {ENV_URL}")
print(f"Model       : {MODEL_ID}")
print(f"Episodes    : {NUM_EPISODES}")
"""),
    nbf.v4.new_markdown_cell("""## 3. Smoke Test — Verify Environment Connectivity
Connect to the HuggingFace Space, reset the 16-service environment (which injects faults), and run a healthcheck command to confirm the round-trip works."""),
    nbf.v4.new_code_cell("""import httpx
import json

print(f"Connecting to {ENV_URL} ...")

client = httpx.Client(base_url=ENV_URL, timeout=30.0)

try:
    # 1. Reset Environment
    reset_res = client.post(f"{ENV_URL}/reset").json()
    print("✅ Connected and Reset successfully!")
    print(f"Alert Context:\\n{reset_res['observation']['context'][:300]}...\\n")
    
    # 2. Run Smoke Test Command
    step_payload = {"action": {"command": "healthcheck api-gateway"}}
    step_res = client.post(f"{ENV_URL}/step", json=step_payload).json()
    
    print(f"--- Smoke Test Step (Command: healthcheck api-gateway) ---")
    print(f"Reward: {step_res['reward']:.2f}")
    print(f"Output:\\n{step_res['observation']['command_output']}")
    print("\\n✅ Smoke test passed. Environment is ready for training.")
except Exception as e:
    print(f"❌ Connection failed: {e}")
"""),
    nbf.v4.new_markdown_cell("""## 4. GRPO Training Definition
We define the system prompt, reward functions, and the GRPO rollout mechanism. The agent earns rewards for proper syntax, successful diagnosis, and fully resolving the cascading outages."""),
]

# Read the existing script content
with open('training_scripts/CloudSRE_Training_Colab.py', 'r', encoding='utf-8') as f:
    script_content = f.read()

nb['cells'].append(nbf.v4.new_code_cell(script_content))

nb['cells'].extend([
    nbf.v4.new_markdown_cell("""## 5. Visualize Reward Curve
Use matplotlib and seaborn to visualize the training breakthrough."""),
    nbf.v4.new_code_cell("""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming the training script outputs a reward CSV in the output_dir
# (Update path based on your specific run folder)
try:
    import glob
    log_files = glob.glob("outputs/*/reward_log.csv")
    if log_files:
        latest_log = max(log_files, key=os.path.getctime)
        df = pd.read_csv(latest_log)
        
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df, x='episode', y='total_reward', marker='o')
        plt.title('GRPO Training Reward Curve')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.show()
except Exception as e:
    print(f"Could not plot: {e}")
"""),
    nbf.v4.new_markdown_cell("""## 6. Push to HuggingFace Hub
Upload the trained LoRA adapter."""),
    nbf.v4.new_code_cell("""# trainer.push_to_hub(repo_id=HUB_REPO)
# print(f"Model successfully pushed to https://huggingface.co/{HUB_REPO}")
""")
])

with open('CloudSRE_Training_Colab.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generated successfully!")
