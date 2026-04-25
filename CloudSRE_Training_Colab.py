# =============================================================================
# 🔥 CloudSRE v2 — ULTIMATE RELAY TRAINING NOTEBOOK
# =============================================================================
# 
# WHO:    For Hari's friends who are new to Colab + HF.
# WHAT:   Train a 1.5B model to fix cloud infrastructure incidents.
# WHERE:  Google Colab (free T4 GPU) OR HuggingFace Space (T4 GPU).
# HOW:    Copy-paste each cell. Don't change ANYTHING unless told to.
#
# ╔═══════════════════════════════════════════════════════════════════╗
# ║  FRIEND ASSIGNMENT — READ THIS FIRST                            ║
# ║                                                                   ║
# ║  🟦 COLAB FRIEND (starts FIRST):                                 ║
# ║    → Platform: Google Colab (free T4)                             ║
# ║    → Trains: warmup + single_fault (the easy tiers)              ║
# ║    → Uses base model (no previous training needed)               ║
# ║    → Pushes result as "leg1" when done                           ║
# ║    → In Cells 6, 7, 9: use the defaults (LEG 1)                 ║
# ║                                                                   ║
# ║  🟧 HF FRIEND (starts AFTER Colab Friend says "pushed"):        ║
# ║    → Platform: HuggingFace Space (T4 GPU — cheapest option)      ║
# ║    → Trains: cascade + multi_cascade (the hard tiers)            ║
# ║    → Downloads Colab Friend's model automatically                ║
# ║    → Pushes result as "leg2" when done                           ║
# ║    → In Cells 6, 7, 9: UNCOMMENT the LEG 2 lines               ║
# ║                                                                   ║
# ║  🟩 HARI (starts AFTER HF Friend says "pushed"):                 ║
# ║    → Trains: adversarial (the death spirals)                     ║
# ║    → In Cells 6, 7, 9: UNCOMMENT the LEG 3 lines               ║
# ╚═══════════════════════════════════════════════════════════════════╝
#
# TIME ESTIMATE: ~45 min per leg on T4
# =============================================================================

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1: GPU CHECK
# 👥 BOTH FRIENDS RUN THIS — Colab Friend AND HF Friend
# If it says "No GPU", stop and fix before continuing.
# ═══════════════════════════════════════════════════════════════════════════════
import torch
print("=" * 60)
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU FOUND: {gpu_name} ({gpu_mem:.1f} GB)")
    print("   You're good to go!")
else:
    print("❌ NO GPU DETECTED!")
    print("   Colab: Runtime → Change runtime type → T4 GPU")
    print("   HF:    Make sure you selected a T4 GPU Space")
    print("   Then re-run this cell.")
print("=" * 60)

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2: INSTALL EVERYTHING (takes ~3-4 minutes, be patient)
# 👥 BOTH FRIENDS RUN THIS — Colab Friend AND HF Friend
# DO NOT MODIFY THIS CELL.
# ═══════════════════════════════════════════════════════════════════════════════
print("Installing dependencies... this takes 3-4 minutes...")

# Install PyTorch first (Colab has it, HF Spaces may not)
# Pin versions to avoid conflicts with unsloth (needs torch<2.11)
!pip install "torch>=2.4.0,<2.11.0" "torchvision>=0.19.0,<0.26.0" 2>/dev/null

# Install unsloth + all its dependencies
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" 2>/dev/null
!pip install unsloth_zoo 2>/dev/null
!pip install trl peft accelerate bitsandbytes 2>/dev/null
!pip install xformers triton 2>/dev/null

# Install training dependencies
!pip install httpx wandb matplotlib 2>/dev/null

print("\n✅ All dependencies installed!")

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3: LOGIN TO HUGGINGFACE + WANDB
# 👥 BOTH FRIENDS RUN THIS — Colab Friend AND HF Friend
# This will ask for your tokens. Paste them when prompted.
# 
# HuggingFace token: https://huggingface.co/settings/tokens (create a WRITE token)
# WandB token: https://wandb.ai/authorize
# ═══════════════════════════════════════════════════════════════════════════════

from huggingface_hub import login as hf_login
import wandb

print("=" * 60)
print("STEP 1: HuggingFace Login")
print("  Go to: https://huggingface.co/settings/tokens")
print("  Create a token with WRITE permission")
print("  Paste it below:")
print("=" * 60)
hf_login()

print("\n" + "=" * 60)
print("STEP 2: Weights & Biases Login")
print("  Go to: https://wandb.ai/authorize")
print("  Copy the API key and paste it below:")
print("=" * 60)
wandb.login()

print("\n✅ Both logins successful!")

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4: CLONE THE REPO
# 👥 BOTH FRIENDS RUN THIS — Colab Friend AND HF Friend
# Auto-detects platform: works on Colab, HF Space, or local machine.
# ═══════════════════════════════════════════════════════════════════════════════
import os

# Auto-detect platform
if os.path.exists("/content"):       # Google Colab
    BASE_DIR = "/content"
    PLATFORM = "Colab"
elif os.path.exists("/home/user"):   # HuggingFace Space
    BASE_DIR = "/home/user"
    PLATFORM = "HF Space"
else:                                 # Local / other
    BASE_DIR = os.path.expanduser("~")
    PLATFORM = "Local"

REPO_DIR = os.path.join(BASE_DIR, "CloudSRE")
print(f"Platform detected: {PLATFORM}")
print(f"Working in: {BASE_DIR}")

os.chdir(BASE_DIR)

# Remove old copy if it exists
!rm -rf CloudSRE

# Clone fresh
!git clone https://github.com/Harikishanth/CloudSRE.git
os.chdir(REPO_DIR)
print(f"\n✅ Repo cloned. Working directory: {os.getcwd()}")
!ls -la train_grpo.py  # Verify the training script exists

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5: VERIFY ENVIRONMENT IS ALIVE
# 👥 BOTH FRIENDS RUN THIS — Colab Friend AND HF Friend
# If this fails, the Space might be sleeping — go to the URL and wake it up.
# ═══════════════════════════════════════════════════════════════════════════════

import httpx

ENV_URL = "https://dardrax-cloudsre-environment.hf.space"

print(f"Checking environment at: {ENV_URL}")
try:
    r = httpx.get(f"{ENV_URL}/health", timeout=30)
    print(f"  Status: {r.status_code}")
    print(f"  Response: {r.text[:200]}")
    print(f"\n✅ Environment is ALIVE and responding!")
except Exception as e:
    print(f"\n❌ Environment is DOWN: {e}")
    print(f"   Go to {ENV_URL} in your browser and wait for it to start")
    print(f"   Then re-run this cell")

# Quick smoke test — reset an episode
try:
    r = httpx.post(f"{ENV_URL}/reset", json={"task_id": "warmup"}, timeout=60)
    data = r.json()
    obs = data.get("observation", data)
    n_services = len(obs.get("service_health", {}))
    print(f"  Services running: {n_services}")
    print(f"  Alert: {obs.get('alert', 'none')[:100]}")
    print(f"\n✅ Smoke test passed — {n_services} services online!")
except Exception as e:
    print(f"\n❌ Smoke test FAILED: {e}")

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6: LOAD THE MODEL
# ⚠️ READ CAREFULLY — different for each friend!
#
# 🟦 COLAB FRIEND: Use the default (LEG 1 — base model). Don't change anything.
# 🟧 HF FRIEND:    COMMENT OUT the LEG 1 line, UNCOMMENT the LEG 2 line.
# 🟩 HARI:          COMMENT OUT the LEG 1 line, UNCOMMENT the LEG 3 line.
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_ID = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"  # ← 🟦 COLAB FRIEND uses this (LEG 1)
# MODEL_ID = "DarDrax/cloudsre-1.5B-leg1"             # ← 🟧 HF FRIEND uses this (LEG 2)
# MODEL_ID = "DarDrax/cloudsre-1.5B-leg2"             # ← 🟩 HARI uses this (LEG 3)

print(f"Loading model: {MODEL_ID}")
print("This takes 2-3 minutes on a T4...")

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters for training
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"\n✅ Model loaded!")
print(f"   Total parameters:     {total:,}")
print(f"   Trainable (LoRA):     {trainable:,}")
print(f"   Memory used:          {torch.cuda.memory_allocated()/1e9:.1f} GB")

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7: 🏃 START TRAINING 🏃
# ⚠️ READ CAREFULLY — different for each friend!
#
# 🟦 COLAB FRIEND: Use the default (warmup,single_fault). Don't change anything.
# 🟧 HF FRIEND:    COMMENT OUT LEG 1 line, UNCOMMENT LEG 2 line.
# 🟩 HARI:          COMMENT OUT LEG 1 line, UNCOMMENT LEG 3 line.
# ═══════════════════════════════════════════════════════════════════════════════

CURRICULUM = "warmup,single_fault"      # ← 🟦 COLAB FRIEND uses this (LEG 1)
# CURRICULUM = "cascade,multi_cascade"  # ← 🟧 HF FRIEND uses this (LEG 2)
# CURRICULUM = "adversarial"            # ← 🟩 HARI uses this (LEG 3)

# ──── DO NOT CHANGE ANYTHING BELOW THIS LINE ────

ENV_URL = "https://dardrax-cloudsre-environment.hf.space"
WANDB_PROJECT = "CloudSRE-Hackathon-Run"
EPISODES = 25  # per tier
GROUP_SIZE = 4  # parallel rollouts

print("=" * 70)
print(f"  🔥 GRPO TRAINING — CloudSRE v2")
print(f"  Curriculum:    {CURRICULUM}")
print(f"  Episodes/tier: {EPISODES}")
print(f"  Group size:    {GROUP_SIZE} rollouts")
print(f"  Environment:   {ENV_URL}")
print(f"  WandB project: {WANDB_PROJECT}")
print("=" * 70)

!cd {REPO_DIR} && python train_grpo.py \
    --env-url "{ENV_URL}" \
    --model-id "{MODEL_ID}" \
    --curriculum "{CURRICULUM}" \
    --episodes-per-tier {EPISODES} \
    --group-size {GROUP_SIZE} \
    --max-turns 10 \
    --lr 5e-6 \
    --save-every 10 \
    --wandb-project "{WANDB_PROJECT}" \
    --output-dir "./cloudsre-grpo"

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8: 📊 PLOT TRAINING RESULTS (run after training finishes)
# 👥 BOTH FRIENDS RUN THIS — Colab Friend AND HF Friend
# ═══════════════════════════════════════════════════════════════════════════════
import json
import matplotlib.pyplot as plt

log_path = os.path.join(REPO_DIR, "grpo_training_log.json")
graph_path = os.path.join(REPO_DIR, "grpo_reward_curve.png")

try:
    with open(log_path) as f:
        log = json.load(f)

    rewards = [ep["reward"] for ep in log["per_episode"]]
    episodes = list(range(1, len(rewards) + 1))

    # Rolling average
    window = min(5, len(rewards))
    rolling = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        rolling.append(sum(rewards[start:i+1]) / (i - start + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Reward curve
    ax1.plot(episodes, rewards, alpha=0.4, color='#3498db', linewidth=1, label='Per-episode')
    ax1.plot(episodes, rolling, color='#e74c3c', linewidth=2.5, label=f'{window}-ep avg')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('CloudSRE v2 — GRPO Training Reward', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Resolution rate per tier
    tiers = log.get("tier_results", {})
    if tiers:
        names = list(tiers.keys())
        rates = [tiers[t]["resolution_rate"] for t in names]
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
        bars = ax2.bar(names, rates, color=colors[:len(names)], alpha=0.85, edgecolor='white')
        ax2.set_ylabel('Resolution Rate (%)', fontsize=12)
        ax2.set_title('Resolution Rate by Tier', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 100)
        for bar, rate in zip(bars, rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Graph saved to {graph_path}")
except FileNotFoundError:
    print("❌ Training log not found — did training complete?")

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9: 🏁 PUSH MODEL TO HUGGINGFACE
# ⚠️ READ CAREFULLY — different for each friend!
#
# 🟦 COLAB FRIEND: Use the default (leg1). Don't change anything.
#                   After this finishes, TEXT THE GROUP: "Pushed! 🟧 HF Friend go!"
#
# 🟧 HF FRIEND:    COMMENT OUT LEG 1 line, UNCOMMENT LEG 2 line.
#                   After this finishes, TEXT THE GROUP: "Pushed! 🟩 Hari go!"
#
# 🟩 HARI:          COMMENT OUT LEG 1 line, UNCOMMENT LEG 3 line.
# ═══════════════════════════════════════════════════════════════════════════════

PUSH_REPO = "DarDrax/cloudsre-1.5B-leg1"    # ← 🟦 COLAB FRIEND pushes this
# PUSH_REPO = "DarDrax/cloudsre-1.5B-leg2"  # ← 🟧 HF FRIEND pushes this
# PUSH_REPO = "DarDrax/cloudsre-1.5B-FINAL" # ← 🟩 HARI pushes this

from unsloth import FastLanguageModel

model_path = os.path.join(REPO_DIR, "cloudsre-grpo")
print(f"Loading trained model from {model_path} ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=2048,
    load_in_4bit=True,
)

print(f"Pushing to HuggingFace: {PUSH_REPO} ...")
model.push_to_hub(PUSH_REPO)
tokenizer.push_to_hub(PUSH_REPO)

print(f"\n✅ MODEL PUSHED TO: https://huggingface.co/{PUSH_REPO}")
print(f"\n📣 Tell your friend: 'Pushed! Start your leg!'")

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 10: 📸 SAVE WANDB SCREENSHOT (for README)
# 👥 BOTH FRIENDS RUN THIS — Colab Friend AND HF Friend
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("📊 Your WandB graphs are at:")
print(f"   https://wandb.ai/YOUR_USERNAME/{WANDB_PROJECT}")
print()
print("TO PUT THE GRAPH IN README:")
print("  1. Go to your WandB run page")
print("  2. Screenshot the reward curve")
print("  3. Save it as 'grpo_reward_curve.png'")
print("  4. Or use the local one we just saved!")
print("=" * 60)

# Download the graph (Colab only — on HF Space, just find it in the file browser)
try:
    from google.colab import files
    files.download(os.path.join(REPO_DIR, 'grpo_reward_curve.png'))
    print("✅ Downloaded graph to your computer!")
except ImportError:
    print(f"Not on Colab — find the graph at: {os.path.join(REPO_DIR, 'grpo_reward_curve.png')}")
except:
    print("Graph file not found — run Cell 8 first")
