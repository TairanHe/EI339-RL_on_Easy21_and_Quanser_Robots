import numpy as np
import os

from policy_eval import *
from env_easy21 import Easy21


class ValueIteration:
    def __init__(self, env: Easy21, reward_decay=1., theta=0.00001,):
        self.action_space = Easy21.action_space
        self.action_num = Easy21.action_num
        self.state_space = Easy21.state_space
        self.non_terminal_state_space = Easy21.non_terminal_state_space
        self.terminal_state_space = Easy21.terminal_state_space
        self.state_num = Easy21.state_num

        self.gamma = reward_decay
        self.theta = theta

        self.env = env
        self.state = self.env.reset()
        self.reward_trace = []
        self.average_reward_trace = []

        self.value = {state: 0 for state in self.non_terminal_state_space}
        self.pi = {state: 0 for state in self.non_terminal_state_space}

    def update_value(self):
        """ Update the value function by one iteration. """
        delta = 0
        new_value = {state: 0. for state in self.non_terminal_state_space}
        for s in self.non_terminal_state_space:
            qsa_list = []
            for a in range(self.action_num):
                qsa = 0
                for prob, next_state, reward in self.env.trans_mat[s][a]:
                    next_state, game_end = (next_state[0], next_state[1]), next_state[2]
                    if game_end == 0:  # game not ends
                        qsa += prob * (reward + self.gamma * self.value[next_state])
                    else:  # game ends
                        qsa += prob * reward
                qsa_list.append(qsa)
            new_value[s] = np.max(qsa_list)
            delta = max(delta, abs(new_value[s] - self.value[s]))
            self.pi[s] = np.argmax(qsa_list)
        self.value = new_value
        return delta

    def value_iteration(self):
        """ Update the value function until difference is less than theta. """
        cnt = 1
        delta = self.theta + 1
        while delta > self.theta:
            delta = self.update_value()
            print("Value iteration: {:d}, delta = {:.4f}".format(cnt, delta))
            cnt += 1
        return cnt

    def test_policy(self, episode_num=1_000_000):
        """ Test current policy on real environment by certain times. 

        :param episode_num: <int> the number of episodes to run.
        """
        average_reward = 0
        for i in range(episode_num):
            if (i + 1) % (episode_num // 10) == 0:
                print("Episode: {} / {}".format(i + 1, episode_num))
                print("\tAverage Reward: {}".format(average_reward))
            state, reward = self.env.reset(), 0
            while state[2] == 0:
                action = self.pi[(state[0], state[1])]
                state, reward = self.env.step(state, action)
            average_reward += (reward - average_reward) / (i + 1)
            self.average_reward_trace.append(average_reward)

    def save_fig(self, fig_path):
        """ Save the value function figure in the given path. """
        plot_V(self.value, save_path=(fig_path + "optimal_value.pdf"))

    def save_result(self, log_path):
        """ Save the logs in the given path. """
        with open(log_path + "optimal_value.txt", "w") as f:
            f.write(str(self.value))
        with open(log_path + "optimal_policy.txt", "w") as f:
            f.write(str(self.pi))
        try:
            ART = np.array(self.average_reward_trace, dtype=np.float16)
            ART_name = log_path + "optimal_exp_ave_reward_{}.txt".format(len(self.average_reward_trace))
            np.savetxt(ART_name, ART, fmt="%.4f", delimiter=",")
        except Exception as e:
            print("No experiment value! Saving average reward trace failed.")


def main():
    env = Easy21()
    log_path = "./log/"
    fig_path = "./fig/"
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    
    model = ValueIteration(env)
    model.value_iteration()
    model.test_policy()

    model.save_fig(fig_path)
    model.save_result(log_path)


if __name__ == '__main__':
    main()

