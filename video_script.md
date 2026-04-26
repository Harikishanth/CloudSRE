# CloudSRE v2: Hackathon Demo Video Script

**Target Length:** 3 Minutes
**Format:** Screen recording with voiceover (use OBS, Loom, or Windows Game Bar `Win + Alt + R`).
**Setup before recording:** 
1. Have the Gradio dashboard open in your browser (`python dashboard.py`).
2. Have your `README.md` architecture diagram visible.
3. Keep the WandB `cascade_resolution_rate.png` graph handy for the end.

---

### Part 1: The Hook & Architecture (0:00 - 0:45)
**[Visual: Show the GitHub README Architecture Diagram]**

**Speaker:** 
"Hi judges, this is CloudSRE v2. 
Most RL environments for SREs are just text-based simulators. We wanted to build something real. We built a fully functional, 16-node microservice architecture running locally, where the agent has to execute *real* Linux commands—like `ps`, `kill`, `curl`, and `sqlite3`—to fix actual infrastructure outages."

"To train it, we used a Dual-LLM architecture. We trained a Qwen 1.5B agent using GRPO across a 5-tier curriculum, while a massive 72B LLM acts as an 'Adversarial Dungeon Master', dynamically grading the agent and exploiting its weaknesses."

---

### Part 2: The Live Demo (0:45 - 2:00)
**[Visual: Switch to the Gradio Dashboard. Click 'Inject Single Fault']**

**Speaker:** 
"Let's look at the live environment in action. 
I just injected a failure into the Payment service. In the background, real Linux processes are running."

**[Visual: Point to the dashboard as the Payment box turns RED and the Agent Terminal starts scrolling]**

**Speaker:**
"Watch the agent's terminal. It isn't just guessing. It runs `systemctl status` to triage, uses `curl` to check the endpoints, and reads the logs to find the exact error. 
It correctly identifies the issue and issues the exact commands to restart and verify the payment service."

**[Visual: Wait for the dashboard boxes to turn back to GREEN]**

**Speaker:**
"And there it goes. The system is completely restored to green."

---

### Part 3: The Cascade Breakthrough (2:00 - 3:00)
**[Visual: Switch to the WandB `cascade_resolution_rate.png` Graph]**

**Speaker:** 
"Once the agent mastered single faults, we moved it to the 'Cascade' difficulty. In a cascade, simply restarting a service causes a thundering herd that crashes the system again. It requires complex mitigation, like draining queues first."

"We trained this entirely on Kaggle's free T4 GPUs. If you look at our WandB logs, right at Episode 14 of the Cascade tier, you can see the exact moment the agent has a breakthrough. It discovered how to mitigate the queue flooding before fixing the root cause, jumping from a 20% resolution rate to nearly 90%."

**[Visual: Show the GitHub repo / HF Space]**

**Speaker:**
"Everything—from the environment to the training pipeline—is open source and available in our HuggingFace space. Thank you for your time."
