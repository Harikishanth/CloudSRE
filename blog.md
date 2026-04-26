# 🔥 CloudSRE v2: Teaching AI to Survive the Death Spiral

*A Meta PyTorch OpenEnv Hackathon Submission*

**By Harikishanth**

When an incident pages an SRE at 3:00 AM, the hardest part isn't finding the broken service. The hardest part is fixing it without breaking everything else. 

If a database locks up, the payment service queue fills with messages. If you just clear the database lock, 1,000 queued messages instantly flood the payment service, causing it to crash from memory exhaustion. The "fix" causes a secondary failure. This is called a cascading failure, or a "death spiral."

73% of extended production outages are caused by the *fix*, not the original fault (Google SRE Handbook, Ch.15). Yet, almost no reinforcement learning environments model this. 

We built **CloudSRE v2** to change that. 

## 🏗️ The Architecture: Real Infrastructure, Real Consequences

Instead of mocking an environment where `state["service"] = "down"`, CloudSRE runs **16 real OS-level microservice processes**. 

When a fault is injected, we don't flip a boolean. We use `os.kill(pid, SIGSTOP)` to freeze the scheduler. We use `sqlite3` `EXCLUSIVE` locks to block the database. We write 50MB of junk to the disk to simulate `disk_full`. 

Because the infrastructure is real, the cascading failures are emergent. 

### The 16-Service Mesh
Our environment includes: `payment`, `auth`, `worker`, `frontend`, `cache`, `notification`, `search`, `gateway`, `scheduler`, `storage`, `metrics_collector`, `email`, `billing`, `config`, `dns`, and `loadbalancer`.

## 🎓 The Training: GRPO Curriculum Learning

We trained a **Qwen2.5-1.5B** agent using **GRPO (Group Relative Policy Optimization)**. Because the environment is so complex, throwing the agent into a cascading failure immediately would result in a 0% success rate and no gradient signal. 

Instead, we used a 5-Tier Curriculum:
1. **Warmup**: Single, isolated faults. (e.g., service crashed)
2. **Single Fault**: Single faults with misleading "red herring" logs from downstream services.
3. **Cascade**: The death spiral. Fixing the primary fault triggers a secondary failure unless handled carefully (e.g., draining queues before restarting).
4. **Multi-Cascade**: Multiple simultaneous death spirals. Priority matters.
5. **Adversarial**: A 72B "Dungeon Master" LLM generates unique, targeted scenarios based on the agent's historical weaknesses.

### The Results

We ran the training in two phases: **Colab (Tiers 1 & 2)** and **Kaggle (Tiers 3, 4 & 5)**.

**Phase 1 (Warmup & Single Fault):**
The agent quickly learned the fundamentals. Within 25 episodes, it achieved a 44% resolution rate on Warmup scenarios, learning to use `status` to triage and `cat` to read logs.

**Phase 2 (Cascade Mastery):**
When introduced to the `cascade` tier, the agent initially flatlined at a 0% resolution rate. It kept falling into the death spiral (restarting a service only to have it OOM from queue pressure). 
However, right at **Episode 14**, the agent had a breakthrough. The GRPO algorithm successfully updated the policy, and the resolution rate spiked from 0% to over 20%. The rolling average reward shot into the positive. The agent successfully learned to drain queues *before* restarting services to prevent OOM crashes.

## 🚀 Why This Matters

CloudSRE v2 proves that we can train small, 1.5B parameter models to reason about complex, multi-step infrastructure workflows using Reinforcement Learning. By using real OS processes instead of API mocks, we created an environment where the agent's actions have real, systemic consequences.

We didn't just teach an AI to read logs. We taught it SRE.

*Check out our README for the full technical specifications, and the WandB logs in our HuggingFace Space!*
