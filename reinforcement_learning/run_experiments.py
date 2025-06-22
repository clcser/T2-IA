import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os

NUM_RUNS = 100
NUM_EPISODES = 2500
EXECUTABLE = "./main"

def run_experiments(label, algorithm_value):
    all_rewards = np.zeros((NUM_RUNS, NUM_EPISODES))

    for run in range(NUM_RUNS):
        print(f"[{label}] Run {run + 1}/{NUM_RUNS}")

        with open("config.h", "w") as f:
            f.write(f"#define ALGORITHM {algorithm_value}\n")

        subprocess.run(["g++", "main.cpp", "-o", "main", "-include", "config.h"])
        subprocess.run([EXECUTABLE], stdout=subprocess.DEVNULL)

        rewards = []
        with open("Rewards.txt", "r") as f:
            for line in f:
                if "Total reward obtained" in line:
                    reward = float(line.split(":")[-1].strip())
                    rewards.append(reward)

        if len(rewards) != NUM_EPISODES:
            raise ValueError(f"{label} Run {run + 1}: Expected {NUM_EPISODES}, got {len(rewards)}")

        all_rewards[run] = rewards

    median = np.percentile(all_rewards, 50, axis=0)
    q25 = np.percentile(all_rewards, 25, axis=0)
    q75 = np.percentile(all_rewards, 75, axis=0)

    return median, q25, q75

# correr algoritmos
q_median, q_q25, q_q75 = run_experiments("Q-learning", algorithm_value=1)
s_median, s_q25, s_q75 = run_experiments("SARSA", algorithm_value=2)

# dibujar
plt.figure(figsize=(10, 5))

step = 25
episodes = np.arange(0, NUM_EPISODES, step)

q_med_down = q_median[::step]
q_q25_down = q_q25[::step]
q_q75_down = q_q75[::step]

s_med_down = s_median[::step]
s_q25_down = s_q25[::step]
s_q75_down = s_q75[::step]

# qlearning
plt.plot(episodes, q_med_down, label="Q-learning (mediana)", color='#3485FF')
plt.fill_between(episodes, q_q25_down, q_q75_down, color='#3485FF', alpha=0.2, label="Q-learning IQR")

# sarsa
plt.plot(episodes, s_med_down, label="SARSA (mediana)", color='#FF447F')
plt.fill_between(episodes, s_q25_down, s_q75_down, color='#FF447F', alpha=0.2, label="SARSA IQR")

plt.xlabel("Episode")
plt.ylabel("Recompensa acumulada")
plt.title(f"SARSA vs Q-learning Cliff Walking(Mdn + IQR, {NUM_RUNS} runs)")
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("sarsa_vs_qlearning_iqr_1.png")
plt.show()