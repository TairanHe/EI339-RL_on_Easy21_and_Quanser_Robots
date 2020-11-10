import numpy as np
import matplotlib.pyplot as plt
import os

from policy_eval import *

log_path = "./log/"
fig_path = "./fig/"
if not os.path.exists(log_path):
    os.mkdir(log_path)
if not os.path.exists(fig_path):
    os.mkdir(fig_path)


def plot_reward_curve(ave_rewards, names, save_path=None, show=False):
    """ Plotting the reward curve. 

    :param ave_rewards: <list> the list of average reward trace to plot.
                               All elements should be of the same length. Below 2 parameters are the same.
    :param names: <list> the list of names to plot.
    :param save_path: <str> path to save the figure. None for not saving figure.
    :param show: <bool> whether to show the figure or not.
    """
    if len(ave_rewards) == 0 or len(ave_rewards[0]) == 0:
        raise Exception("Length of reward trace is 0! Please check the data.")
    if len(ave_rewards) != len(names):
        raise Exception("Length of reward is inconsistent! Please check the data.")

    episode_num = len(ave_rewards[0])

    fig = plt.figure(figsize=(8, 6), dpi=160)
    ax1 = fig.add_subplot(111)

    ax1.set(title="Average Cumulative Reward",
            xlim=[-episode_num * 0.05, episode_num * 1.05],
            ylim=[-0.25, 0.05],
            xlabel="Episode",
            ylabel="Average Reward")
    ax1.set_xticks(range(0, (episode_num // 10) * 10 + 1, episode_num // 10))
    ax1.set_xticklabels(range(0, (episode_num // 10) * 10 + 1, episode_num // 10), rotation=45)
    ax1.set_yticks(np.arange(-0.20, 0.05, 0.05))
    for i in np.arange(-0.20, 0.05, 0.05):
        ax1.axhline(i, color="black", alpha=0.4, ls="--")

    for idx, ave_reward_trace in enumerate(ave_rewards):
        ax1.plot(range(episode_num), ave_reward_trace, label=names[idx])

    ax1.legend(ncol=1)
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def compare():
    env = Easy21()
    show_episode_num = 100_000
    ave_rewards, names = [], []
    files = [("Q_learning_LR=0.010_e=0.001_100000_ave_reward_100000.txt",
              "Q-learning, Epsilon, LearningRate=0.01, Epsilon=0.001"), 
             ("Q_learning_LR=0.005_DE_ave_reward_100000.txt",
              "Q-learning, DynamicEpsilon, LearningRate=0.005"), 
             ("policy_iter_exp_ave_reward_1000000.txt",
              "Policy Iteration"), 
             ("MCMC_ave_reward_1000000.txt",
              "MCMC (Markov Chain Monte-Carlo)"), 
            ]
    
    for file_name, name in files:
        try:
            R = load_reward(log_path + file_name)
            if len(R) > show_episode_num:
                R = R[:show_episode_num]
            ave_rewards.append(R)
            names.append(name)
        except:
            print("No file named", file_name, "!\nPlease check the file list, or try running 'main.py' again.")
    
    plot_reward_curve(ave_rewards, names, save_path=fig_path + "CMP.png", show=True)


if __name__ == '__main__':
    compare()
