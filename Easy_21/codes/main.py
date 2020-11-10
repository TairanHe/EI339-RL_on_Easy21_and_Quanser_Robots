import numpy as np
import os

from policy_eval import *
from env_easy21 import Easy21


def value_iter_run(test_episode_num=1_000_000):
    """ Run value iteration algorithm. """
    model = ValueIteration(env)
    model.value_iteration()
    model.test_policy(episode_num=test_episode_num)
    model.save_fig(fig_path)
    model.save_result(log_path)


def Q_learning_run(episode_num=100_000):
    """ Run Q-learning algorithm. 

    Parameter introduction please refer to 'Q_learning.py'
    """
    ave_rewards, rmses, names = [], [], []

    PARAM_DICT = {"DynamicEpsilon, LearningRate=0.005": {"alpha": 0.005},
                  "DynamicEpsilon, LearningRate=0.01": {"alpha": 0.01},
                  "DynamicEpsilon, LearningRate=0.05": {"alpha": 0.05},
                  "DynamicEpsilon, LearningRate=0.1": {"alpha": 0.1},
                  "DynamicEpsilon, LearningRate=0.5": {"alpha": 0.5},

                  "Epsilon, LearningRate=0.01, Epsilon=0.001": {"de": False, "e": 0.001},
                  "Epsilon, LearningRate=0.01, Epsilon=0.01": {"de": False, "e": 0.01},
                  "Epsilon, LearningRate=0.01, Epsilon=0.1": {"de": False, "e": 0.1},
                  "Epsilon, LearningRate=0.01, Epsilon=1.0": {"de": False, "e": 1.},
                  }

    for name, param in PARAM_DICT.items():
        print("#" * 5, name, "#" * 5)
        model = QLearning(env, episode_num=episode_num,
                          learning_rate=param.get("alpha", 0.01), 
                          dynamic_epsilon=param.get("de", True),
                          epsilon=param.get("e", 0.9),
                          damping_factor=param.get("r", None),
                          final_epsilon=param.get("fe", None))
        model.run(print_info=False)
        ave_rewards.append(model.average_reward_trace)
        rmses.append(model.rmse_trace)
        names.append(name)
        model.save_fig(fig_path)
        model.save_result(log_path)

    plot_learning_curve(ave_rewards[:5], rmses[:5], names[:5],
                        save_path=fig_path + "different_alpha_CMP.png", show=False)
    plot_learning_curve(ave_rewards[5:], rmses[5:], names[5:],
                        save_path=fig_path + "different_epsilon_CMP.png", show=False)

    result = []
    for i in range(len(ave_rewards)):
        result.append((names[i], ave_rewards[i][-1]))
    result.sort(key=lambda x: x[1])
    for name, reward in result:
        print(name)
        print("\tAverage Reward: {:.4f}".format(reward))


def policy_iter_run(test_episode_num=1_000_000):
    """ Run value iteration algorithm.

    Parameter introduction please refer to 'policy_iter.py'
    """
    model = PolicyIteration(env)
    model.policy_iteration()
    model.test_policy(episode_num=test_episode_num)
    model.save_fig(fig_path)
    model.save_result(log_path)


def MCMC_run(episode_num=1_000_000):
    """ Run MCMC algorithm.

    Parameter introduction please refer to 'monte_carlo.py'
    """
    model = MCMC(env)
    model.run(episode_num=episode_num)
    model.run(episode_num=episode_num, update=False)
    model.save_fig(fig_path)
    model.save_result(log_path)

    ave_rewards, rmses, names = [model.average_reward_trace[episode_num:]], [model.rmse_trace[:episode_num]], ["MCMC"]
    plot_learning_curve(ave_rewards, rmses, names, save_path=fig_path + "MCMC_reward.png", show=False)


if __name__ == "__main__":
    env = Easy21()
    log_path = "./log/"
    fig_path = "./fig/"
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)

    from value_iter import ValueIteration
    value_iter_run()

    from Q_learning import QLearning
    Q_learning_run()

    from policy_iter import PolicyIteration
    policy_iter_run()

    from monte_carlo import MCMC
    MCMC_run()
    
    from compare import compare
    compare()
