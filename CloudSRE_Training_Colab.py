# =============================================================================
# 🔥 CloudSRE v2 — ULTIMATE RELAY TRAINING NOTEBOOK
# =============================================================================
# 
# WHO:    For Hari's friends who are new to Colab + HF.
# WHAT:   Train a 1.5B model to fix cloud infrastructure incidents.
# WHERE:  Google Colab (free T4 GPU) → HuggingFace Space (environment).
# HOW:    Copy-paste each cell. Don't change ANYTHING unless told to.
#
# ╔═══════════════════════════════════════════════════════════════════╗
# ║  FRIEND ASSIGNMENT — READ THIS FIRST                            ║
# ║                                                                   ║
# ║  FRIEND A (starts FIRST):                                        ║
# ║    → Trains: warmup + single_fault (the easy tiers)              ║
# ║    → Uses base model (no previous training needed)               ║
# ║    → Pushes result as "leg1" when done                           ║
# ║    → In Cells 6, 7, 9: use the defaults (LEG 1)                 ║
# ║                                                                   ║
# ║  FRIEND B (starts AFTER Friend A says "pushed"):                 ║
# ║    → Trains: cascade + multi_cascade (the hard tiers)            ║
# ║    → Downloads Friend A's model automatically                    ║
# ║    → Pushes result as "leg2" when done                           ║
# ║    → In Cells 6, 7, 9: UNCOMMENT the LEG 2 lines               ║
# ║                                                                   ║
# ║  HARI (starts AFTER Friend B says "pushed"):                     ║
# ║    → Trains: adversarial (the death spirals)                     ║
# ║    → In Cells 6, 7, 9: UNCOMMENT the LEG 3 lines               ║
# ╚═══════════════════════════════════════════════════════════════════╝
#
# TIME ESTIMATE: ~45 min per leg on Colab T4
# =============================================================================

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1: GPU CHECK (Run this FIRST — if it says "No GPU", stop and fix)
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
    print("   Go to: Runtime → Change runtime type → Hardware accelerator → T4 GPU")
    print("   Then re-run this cell.")
print("=" * 60)

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2: INSTALL EVERYTHING (takes ~3-4 minutes, be patient)
# ═══════════════════════════════════════════════════════════════════════════════
# DO NOT MODIFY THIS CELL.
print("Installing dependencies... this takes 3-4 minutes...")

# Install unsloth FIRST (it pins specific versions)
!pip install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" 2>/dev/null
!pip install --no-deps trl peft accelerate bitsandbytes 2>/dev/null
!pip install xformers triton 2>/dev/null

# Install training dependencies
!pip install httpx wandb matplotlib 2>/dev/null

print("\n✅ All dependencies installed!")

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3: LOGIN TO HUGGINGFACE + WANDB
# ═══════════════════════════════════════════════════════════════════════════════
# This will ask for your tokens. Paste them when prompted.
# 
# HuggingFace token: https://huggingface.co/settings/tokens (create a WRITE token)
# WandB token: https://wandb.ai/authorize

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
# ═══════════════════════════════════════════════════════════════════════════════
import os
os.chdir("/content")

# Remove old copy if it exists
!rm -rf CloudSRE

# Clone fresh
!git clone https://github.com/Harikishanth/CloudSRE.git
os.chdir("/content/CloudSRE")
print(f"\n✅ Repo cloned. Working directory: {os.getcwd()}")
!ls -la train_grpo.py  # Verify the training script exists

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5: VERIFY ENVIRONMENT IS ALIVE
# ═══════════════════════════════════════════════════════════════════════════════
# This checks that the HF Space environment is running and responding.
# If this fails, the Space might be sleeping — go to the HF Space URL and wake it up.

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
# ═══════════════════════════════════════════════════════════════════════════════
# 
# ⚠️ READ THIS ⚠️
# 
# IF YOU ARE LEG 1 (first person training):
#   Use the base model: "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
#
# IF YOU ARE LEG 2 (your friend already pushed "leg1"):
#   Use: "DarDrax/cloudsre-1.5B-leg1"
#
# IF YOU ARE LEG 3 (Hari, after friend pushed "leg2"):
#   Use: "DarDrax/cloudsre-1.5B-leg2"
#
# CHANGE THE LINE BELOW TO MATCH YOUR LEG:

MODEL_ID = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"  # ← LEG 1 (base model)
# MODEL_ID = "DarDrax/cloudsre-1.5B-leg1"             # ← LEG 2 (uncomment this)
# MODEL_ID = "DarDrax/cloudsre-1.5B-leg2"             # ← LEG 3 (uncomment this)

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
# ═══════════════════════════════════════════════════════════════════════════════
#
# ⚠️ READ THIS ⚠️
#
# IF YOU ARE LEG 1: curriculum = "warmup,single_fault"
# IF YOU ARE LEG 2: curriculum = "cascade,multi_cascade"
# IF YOU ARE LEG 3: curriculum = "adversarial"
#
# CHANGE THE LINE BELOW TO MATCH YOUR LEG:

CURRICULUM = "warmup,single_fault"  # ← LEG 1
# CURRICULUM = "cascade,multi_cascade"  # ← LEG 2 (uncomment this)
# CURRICULUM = "adversarial"            # ← LEG 3 (uncomment this)

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

!cd /content/CloudSRE && python train_grpo.py \
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
# ═══════════════════════════════════════════════════════════════════════════════
import json
import matplotlib.pyplot as plt

try:
    with open("/content/CloudSRE/grpo_training_log.json") as f:
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
    plt.savefig('/content/CloudSRE/grpo_reward_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Graph saved to grpo_reward_curve.png")
except FileNotFoundError:
    print("❌ Training log not found — did training complete?")

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9: 🏁 PUSH MODEL TO HUGGINGFACE (MANDATORY — your friend needs this!)
# ═══════════════════════════════════════════════════════════════════════════════
#
# ⚠️ CHANGE THE REPO NAME BELOW TO MATCH YOUR LEG ⚠️
#
# IF YOU ARE LEG 1: push to "DarDrax/cloudsre-1.5B-leg1"
# IF YOU ARE LEG 2: push to "DarDrax/cloudsre-1.5B-leg2"
# IF YOU ARE LEG 3: push to "DarDrax/cloudsre-1.5B-FINAL"

PUSH_REPO = "DarDrax/cloudsre-1.5B-leg1"  # ← CHANGE THIS FOR YOUR LEG
# PUSH_REPO = "DarDrax/cloudsre-1.5B-leg2"   # ← LEG 2
# PUSH_REPO = "DarDrax/cloudsre-1.5B-FINAL"  # ← LEG 3

from unsloth import FastLanguageModel

print(f"Loading trained model from ./cloudsre-grpo ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/content/CloudSRE/cloudsre-grpo",
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

# You can also download the locally generated graph:
from google.colab import files
try:
    files.download('/content/CloudSRE/grpo_reward_curve.png')
    print("✅ Downloaded graph to your computer!")
except:
    print("Graph file not found — run Cell 8 first")
