# 🌩️ The CloudSRE Chronicles: Forging a 1.5B SRE Agent in the Fires of a 72B Adversarial Mind

**By DarDrax | OpenEnv Hackathon 2026**

I’ve spent the better part of three decades watching systems crumble. I’ve seen the "Seven Nines" turn into "Zero Nines" in the span of a single heartbeat. If there’s one undeniable truth in Site Reliability Engineering, it’s this: **infrastructure fails in the most creatively chaotic ways imaginable.** 

When the OpenEnv Hackathon was announced, most teams looked at the prompt and saw a simple objective: train a language model to fix a simulated server. But we looked at the prompt and saw an opportunity to recreate the sheer panic of a 3:00 AM PagerDuty alert—the kind where your pulse hits 140 and the API Gateway is screaming 5xx errors like a banshee. 

We didn't just want an agent that could parse logs; we wanted to forge an intelligence that could stare into the abyss of a cascading, multi-service outage and calmly type `restart_service gateway.us-east-1`.

To do this, we built **CloudSRE v2**. And to train it, we had to do something a little bit insane. We had to build a monster to teach our agent how to fight.

---

## 🏗️ Chapter 1: The 16-Service Labyrinth

A realistic environment cannot be simple. You cannot teach true incident response on a single web server connected to a single database. That’s a tutorial, not an outage.

We built a **16-service microservice labyrinth** across three simulated regions (`us-east-1`, `eu-west-1`, `ap-south-1`). We meticulously simulated the interconnected chaos of a modern tech stack:
- **Core Tiers:** An API Gateway routing traffic to a Frontend, communicating with a critical Billing service.
- **Data Layers:** Postgres databases, Redis caches, and RabbitMQ message queues.
- **Critical Microservices:** Auth, Payment, User, Notification, and Analytics.

But building the cluster was only step one. The real challenge was figuring out how to break it in a way that felt authentic—real `SIGSTOP` signals, real database corruption, and real OOM (Out of Memory) kills.

---

## 🧠 Chapter 2: The 72B Adversarial Mind

How do you train an AI to fix problems? You feed it synthetic data. 
But how do you train an AI to survive a catastrophic cascading failure? You introduce the **Adversarial Designer**.

We integrated a **72B parameter model** and gave it a singular, sinister directive: **"Design the most complex, misleading, and brutal outages possible."**

We structured the training environment into five escalating tiers. While the first two tiers taught the agent the basics, Tiers 3, 4, and 5 were where the 72B Designer took the wheel. It didn’t just break the database. It spiked the CPU on the Analytics service to 99%, filled the `/var/log` directory on the Auth service to 100%, and quietly throttled the message queue. 

It created **"Death Spirals"**—cascading failures where fixing the obvious symptom (the Payment service) only revealed a deeper, more fatal root cause (the Config service in a different region). It planted red herrings in the logs to waste our agent's time, just like a real-world messy incident.

---

## 📈 Chapter 3: The Breakthrough (Unsloth GRPO)

Our training regimen was brutal. We used a **Custom GRPO (Group Relative Policy Optimization)** implementation built directly on the **Unsloth** framework. We bypassed standard high-level wrappers to squeeze every bit of efficiency out of the raw policy gradients and fast LoRA backwards passes.

For the first dozen episodes of Phase 2, our agent was slaughtered. The 72B Designer’s death spirals were too complex. The agent would fix the Gateway, completely ignoring the fact that the actual root cause was a dead RabbitMQ queue backing up the Payment service. The resolution rate sat stubbornly at 0%.

But then, at **Episode 14**, something magical happened.

![Phase 2 Advanced Tiers Metrics](./reward_curve_leg2.png)

If you look at the graphs, you see the exact moment the policy "clicked." The agent realized that fixing the symptom wasn't enough. It started tracing the network topology. It started ignoring the noisy logs and hunting for the silent failures. 

The reward curve shot up, breaking out of the negative values and surging past +1.0. The resolution rate climbed from an abysmal 0% to a sustained 30% against the hardest, most complex adversarial scenarios the 72B model could throw at it.

---

## 🚀 The Result: Hackathon Ready

Today, the CloudSRE v2 environment stands as a testament to what is possible when you combine robust systems engineering with adversarial Reinforcement Learning.

We didn't just build an OpenEnv submission. We built a crucible. 

You can run the environment yourself on our HuggingFace Space. You can inspect our Colab notebooks, meticulously documented and complete with live environment Smoke Tests. You can view the pristine reward curves that prove our agent didn't just memorize solutions—it learned how to triage.

In the end, infrastructure will always fail. But with the right training, and the right adversity, we can build agents that don't blink when the alarms go off.

*(Thank you to the OpenEnv community for an incredible hackathon!)*
