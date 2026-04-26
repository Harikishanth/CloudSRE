# 🌩️ The CloudSRE Chronicles: Bridging the Gap Between Theory and 3 AM Chaos

**By DarDrax | OpenEnv Hackathon 2026**

As students, we spend most of our time in the clean, sanitized world of textbooks and simple "Hello World" tutorials. But in the real world of Site Reliability Engineering (SRE), systems aren't clean. They are messy, interconnected, and prone to the most creative failures imaginable.

For the OpenEnv Hackathon, our team decided to stop playing it safe. We didn't just want to train a language model to fix a server—we wanted to build a crucible that could replicate the sheer, unadulterated chaos of a production cascading failure. We wanted an agent that didn't just memorize logs, but one that could navigate the labyrinth of a 16-service microservice outage.

This is the story of how we built **CloudSRE v2**.

---

## 🏗️ Chapter 1: Building the 16-Service Labyrinth

A realistic environment cannot be a toy. To teach true incident response, we built a **16-service microservice architecture** across three simulated regions (`us-east-1`, `eu-west-1`, `ap-south-1`). 

We meticulously designed the dependencies to mirror a real enterprise tech stack:
- **Core Tiers:** An API Gateway routing traffic to a Frontend, communicating with a critical Billing service.
- **Infrastructure Layers:** Postgres databases, Redis caches, and RabbitMQ message queues.
- **Service Mesh:** Auth, Payment, User, Notification, and Analytics.

Every service in our environment isn't just a mock—it has its own simulated PID namespace, filesystem, and real networking. We implemented actual `SIGSTOP` signals, database corruption, and OOM (Out of Memory) simulations to ensure the agent was dealing with real-world physics, not just text flags.

---

## 🧠 Chapter 2: The 72B Adversarial Mind

To train a world-class agent, you need a world-class opponent. We integrated a **72B parameter model** as our **Adversarial Designer**.

Its singular directive was to outsmart our agent by designing the most brutal, cascading outages possible. It didn’t just break a service; it created **"Death Spirals."** For example, it would fill the `/var/log` directory on the Auth service, which would cause the Gateway to timeout, which would then overflow the RabbitMQ queues in the Payment region. 

Fixing the "obvious" problem wouldn't save the system—the agent had to use causal reasoning to find the silent root cause before the whole cluster collapsed.

---

## 📈 Chapter 3: The Breakthrough (Custom Unsloth GRPO)

Our training regimen pushed our hardware—and our code—to the limit. We implemented a **Custom GRPO (Group Relative Policy Optimization)** pipeline directly on top of the **Unsloth** framework. By writing the policy gradient updates and advantage computations manually, we were able to achieve 10x the training speed of standard wrappers.

The results were stark. For the first dozen episodes, the resolution rate was 0%. The agent was lost in the "Death Spirals." But around **Episode 14**, we witnessed a genuine breakthrough.

![Phase 2 Advanced Tiers Metrics](./reward_curve_leg2.png)

The policy "clicked." The agent stopped chasing red herrings and started tracing the topological flow of the services. The reward curve surged, and the resolution rate climbed from 0% to a sustained 30% against the hardest adversarial scenarios. Our 1.5B model had learned to think like an SRE.

---

## 🚀 Final Thoughts

CloudSRE v2 is more than just a hackathon submission to us. It was an exploration of how we can use large models to build safer, more resilient infrastructure. 

We’ve provided everything for you to dive in: a live HuggingFace Space, a bulletproof Colab Training Notebook, and the final 1.5B parameter model weights. We hope this environment serves as a benchmark for what's possible in the next generation of AI-driven reliability engineering.

*(Thank you to the OpenEnv community for an incredible hackathon!)*
