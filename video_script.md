# CloudSRE v2: Hackathon Demo Video Script

**Target Length:** 1 Minute 45 Seconds (STRICTLY UNDER 2 MINS)
**Format:** Screen recording with voiceover (use OBS, Loom, or Windows Game Bar `Win + Alt + R`).
**Setup before recording:** 
1. Have the Gradio dashboard open in your browser (`python dashboard.py` or your HF Space).
2. Have your `README.md` architecture diagram visible.
3. Keep the WandB `cascade_resolution_rate.png` graph handy.

---

### Part 1: The Hook & Architecture (0:00 - 0:30)
**[Visual: Show the GitHub README Architecture Diagram]**

**Speaker:** 
"Hi judges. This is CloudSRE v2. 
Most environments simulate a single service using fake HTTP responses. We built a 16-node microservice architecture where our agent executes real Linux commands like `curl` and `ps` against actual subprocesses. 
To train it, we used GRPO on a 1.5B model, while a 72B Adversarial LLM dynamically generated novel failures to exploit its weaknesses."

---

### Part 2: The Live Demo (0:30 - 1:15)
**[Visual: Switch to the Gradio Dashboard. Click 'Inject Single Fault']**

**Speaker:** 
"Here's the live environment. I've injected a failure into the Payment service. 
Watch the agent's terminal. It isn't guessing. It uses `systemctl status` to triage, hits the endpoints with `curl`, and reads the actual error logs. 
It correctly identifies the issue, issues the targeted restart, and verifies it. The system is restored to green."

---

### Part 3: The Cascade Breakthrough (1:15 - 1:45)
**[Visual: Switch to the WandB `cascade_resolution_rate.png` Graph]**

**Speaker:** 
"But SRE is about cascading failures. When we moved to the Cascade tier, simply restarting services caused queue flooding. 
We trained this entirely on a single free Kaggle T4 GPU. In our WandB logs, you can see the exact moment the agent had a breakthrough, learning to drain queues before restarting services.
Our entire environment runs locally with zero configuration. Thank you."
