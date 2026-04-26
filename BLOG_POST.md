# 🌩️ The CloudSRE Chronicles: Forging a 1.5B SRE Agent in the Fires of a 72B Adversarial Mind

**By DarDrax | OpenEnv Hackathon 2026**

I’ve spent the better part of my life building systems that shouldn’t break—but always do. If there’s one undeniable truth in Site Reliability Engineering, it’s this: **infrastructure fails in the most creatively chaotic ways imaginable**. 

When the OpenEnv Hackathon was announced, most teams looked at the prompt and saw a simple objective: train a language model to fix a simulated server. But we looked at the prompt and saw an opportunity to recreate the sheer panic of a 3:00 AM pager duty alert. We didn't just want an agent that could parse logs; we wanted to forge an intelligence that could stare into the abyss of a cascading, multi-service outage and calmly type `kubectl restart`.

To do this, we built **CloudSRE v2**. And to train it, we had to do something a little bit insane. We had to build a monster to teach our agent how to fight.

---

## 🏗️ Chapter 1: The 16-Service Abyss

A realistic environment cannot be simple. You cannot teach true incident response on a single web server connected to a single database. That’s a tutorial, not an outage.

We built a **16-service microservice architecture** powered by FastAPI and Uvicorn. We meticulously simulated the interconnected chaos of a modern tech stack:
- **Core Tiers:** An API Gateway routing traffic to a Frontend, communicating with a Backend.
- **Data Layers:** Postgres databases, Redis caches, and RabbitMQ message queues.
- **Critical Microservices:** Auth, Payment, User, Notification, and Analytics.
- **The Telemetry Spine:** Prometheus, Grafana, and an ELK stack monitoring the pulse of the cluster.

But building the cluster was only step one. The real challenge was figuring out how to break it in a way that felt authentic.

---

## 🧠 Chapter 2: The 72B Adversarial Designer

How do you train an AI to fix problems? You feed it synthetic data. 
But how do you train an AI to survive a catastrophic cascading failure? You introduce the **Adversarial Designer**.

We integrated `Qwen2-72B` and gave it a singular, sinister directive: **Design the most complex, misleading, and brutal outages possible.**

We structured the training environment into five escalating tiers of difficulty. The first two tiers (Warmup and Single Fault) taught the agent the basics. But Tiers 3, 4, and 5 were where the 72B Designer took the wheel.

It didn’t just break the database. It spiked the CPU on the Analytics service to 99%, filled the `var/log` directory on the Auth service, and quietly killed the Redis connection. It created **"Death Spirals"**—cascading failures where fixing the obvious symptom only revealed a deeper, more fatal root cause. It intentionally planted red herrings in the logs to waste our agent's time.

We pitted our tiny, 1.5B parameter Qwen agent against a 72B Goliath. 

---

## 📈 Chapter 3: The Breakthrough (GRPO Training)

Our training regimen was brutal. Instead of relying on standard RL wrappers, we built a **Custom GRPO (Group Relative Policy Optimization) implementation purely in Unsloth** utilizing its highly optimized raw gradients and fast LoRA backwards passes.

For the first dozen episodes of Phase 2, our agent was slaughtered. The 72B Designer’s death spirals were too complex. The agent would fix the API Gateway, completely ignoring the fact that the actual root cause was a dead RabbitMQ queue backing up the Payment service. The resolution rate sat stubbornly at 0%.

But then, at **Episode 14**, something magical happened.

![Phase 2 Advanced Tiers Metrics](./reward_curve_leg2.png)

If you look at the graphs, you see the exact moment the policy "clicked". The agent realized that fixing the symptom wasn't enough. It started tracing the network topology. It started ignoring the noisy logs and hunting for the silent failures. 

The reward curve shot up, breaking out of the negative values and surging past +1.0. The resolution rate climbed from an abysmal 0% to a sustained 30% against the hardest, most complex adversarial scenarios the 72B model could throw at it.

Our 1.5B model had learned to fight back.

---

## 🚀 The Result: Hackathon Ready

Today, the CloudSRE v2 environment stands as a testament to what is possible when you combine robust systems engineering with adversarial Reinforcement Learning.

We didn't just build an OpenEnv submission. We built a crucible. 

You can run the environment yourself on our HuggingFace Space. You can inspect our Colab notebooks, meticulously documented and complete with live environment Smoke Tests. You can view the pristine GRPO reward curves that prove our agent didn't just memorize solutions—it learned how to triage.

In the end, infrastructure will always fail. But with the right training, and the right adversity, we can build agents that don't blink when the alarms go off.

*(Thank you to the OpenEnv community for an incredible hackathon!)*
