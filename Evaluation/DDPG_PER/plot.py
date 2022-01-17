import json
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_rewards(triplet, clfd, filename, x=None, window=5):
    N = min(len(triplet), len(clfd))
    running_avg_triplet = np.empty(N)
    running_avg_clfd = np.empty(N)
    for t in range(N):
        running_avg_triplet[t] = np.mean(triplet[max(0, t - window):(t + 1)])
    for t in range(N):
        running_avg_clfd[t] = np.mean(clfd[max(0, t - window):(t + 1)])
    if x is None:
        x = [i for i in range(N)]
    plt.clf()
    plt.ylabel('Rewards')
    plt.xlabel('Episode')
    plt.plot(x, running_avg_triplet, color='red', label='triplet')
    plt.plot(x, running_avg_clfd, color='blue', label='clfd')
    plt.legend()
    plt.savefig(filename)

triplet_rewards =  json.load(open("data_triplet/rewards.json", "r"))[:1000]
clfd_rewards =  json.load(open("data_clfd/rewards.json", "r"))

plot_rewards(triplet = triplet_rewards, clfd_ = clfd_rewards, filename= "combined.pdf")
