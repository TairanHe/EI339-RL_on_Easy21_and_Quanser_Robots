import numpy as np
import os

from env_easy21 import Easy21
from policy_eval import *

log_path = "./log/"
fig_path = "./fig/"
if not os.path.exists(log_path):
    os.mkdir(log_path)
if not os.path.exists(fig_path):
    os.mkdir(fig_path)

global OPTIMAL_POLICY, OPTIMAL_VALUE
try:
    OPTIMAL_POLICY = load_file_dangerous(log_path + "optimal_policy.txt")
except IOError:
    raise Exception("No file named {}!\nPlease run 'value_iter.py' first.".format(log_path + "optimal_policy.txt"))
try:
    OPTIMAL_VALUE = load_file_dangerous(log_path + "optimal_value.txt")
except IOError:
    raise Exception("No file named {}!\nPlease run 'value_iter.py' first.".format(log_path + "optimal_value.txt"))


class QLearning:
    def __init__(self, env: Easy21, learning_rate=0.05, reward_decay=1., episode_num=1_000_000, epsilon=0.1,
                 dynamic_epsilon=False, damping_factor=None, final_epsilon=None, optimistic=False):
        """
        :param learning_rate: <float> learning rate alpha.
        :param reward_decay: <float> reward decay factor gamma.
        :param episode_num: <int> default number of episodes to run.
        :param epsilon: <float> the exploration factor epsilon.
        :param dynamic_epsilon: <bool> whether to turn on Dynamic Epsilon mode or not.
        :param damping_factor: <float> damping factor r for exponential dynamic epsilon, suggest None.
        :param final_epsilon: <float> minimal epsilon for exponential dynamic epsilon, suggest None.
        """
        self.env = env
        self.action_space = Easy21.action_space
        self.action_num = Easy21.action_num
        self.state_space = Easy21.state_space
        self.non_terminal_state_space = Easy21.non_terminal_state_space
        self.terminal_state_space = Easy21.terminal_state_space
        self.state_num = Easy21.state_num

        self.q_table = {}
        self.alpha = learning_rate
        self.gamma = reward_decay
        self.episode_num = episode_num
        self.epsilon = epsilon
        self.dynamic_epsilon = dynamic_epsilon
        self.r = damping_factor if damping_factor is not None else np.power(0.05, 10 / self.episode_num)
        self.final_epsilon = final_epsilon if final_epsilon is not None else 10 / self.episode_num

        self.reward_trace = []
        self.average_reward_trace = []
        self.rmse_trace = []
        self.optimistic = optimistic
        if self.dynamic_epsilon:
            self.name = "LR={:.3f}_DE".format(self.alpha, self.epsilon, self.r, self.episode_num) +  "_optimistic_" + str(self.optimistic)
        else:
            self.name = "LR={:.3f}_e={:.3f}_{:d}".format(self.alpha, self.epsilon, self.episode_num)  + "_optimistic_" + str(self.optimistic)

    def _update_q_table(self, state_, action_, reward_, next_state_):
        """ Update Q table. """
        state_ = (state_[0], state_[1])
        next_state_, game_end = (next_state_[0], next_state_[1]), next_state_[2]
        if self.optimistic:
            if self.q_table.get(state_) is None: self.q_table[state_] = [1] * self.action_num
            if self.q_table.get(next_state_) is None: self.q_table[next_state_] = [1] * self.action_num
        else:
            if self.q_table.get(state_) is None: self.q_table[state_] = [0] * self.action_num
            if self.q_table.get(next_state_) is None: self.q_table[next_state_] = [0] * self.action_num

        q_target = reward_ if game_end == 1 else reward_ + self.gamma * max(self.q_table[next_state_])
        q_predict = self.q_table[state_][action_]

        self.q_table[state_][action_] += self.alpha * (q_target - q_predict)

    def choose_action(self, state_):
        """ Choose action of the given state np.argmax(self.q_table[state_]) according to Q-table. """
        state_ = (state_[0], state_[1])
        if self.q_table.get(state_) is None: self.q_table[state_] = [0] * self.action_num
        if np.random.rand() < self.epsilon:  # choose random action
            action = np.random.randint(self.action_num)
        else:  # choose the index of best action according to Q table
            action = np.argmax(self.q_table[state_])
        return action

    def choose_action_deterministic(self, state_):
        """ Choose action of the given state according to Q-table. """
        if self.q_table.get(state_) is None: self.q_table[state_] = [0] * self.action_num
        action = np.argmax(self.q_table[state_])
        return action

    def run_one_step(self, state):
        """ Run one step on the environment and return the reward. """
        action = self.choose_action(state)
        next_state, reward = self.env.step(state, action)
        self._update_q_table(state, action, reward, next_state)
        return next_state, reward

    def run_one_episode(self):
        """ Run one step on the environment and return the final reward. """
        state, reward = self.env.reset(), 0
        while state[2] == 0:  # game not ends
            state, reward = self.run_one_step(state)
        return reward

    def run(self, episode_num=None, section_num=10, print_info=True):
        """ Run Q-learning algorithm. 

        :param episode_num: <int> the number of episodes to run.
                                  If left None, then the number will be decided by the default number self.episode_num
        :param section_num: <int> the number of sections in the running process.
                                  Used for showing process in each section. E.g. set to 10 will show info for 10 times.
        :param print_info: <bool> whether to show information or not
        """
        episode_num = self.episode_num if episode_num is None else episode_num
        average_reward = 0
        average_reward_section = 0
        section_size = (episode_num // section_num)
        for i in range(episode_num):
            reward = self.run_one_episode()
            # self.reward_trace.append(reward)
            average_reward += (reward - average_reward) / (i + 1)
            self.average_reward_trace.append(average_reward)
            self.rmse_trace.append(rms_error_q(OPTIMAL_VALUE, self.q_table))

            if (i + 1) % section_size == 0:
                average_reward_section += (reward - average_reward_section) / section_size
                policy = {k: 0 if self.q_table.get(k, [0, 0])[0] > self.q_table.get(k, [0, 0])[1] else 1 
                          for k in self.non_terminal_state_space}
                print("Episode: {} / {}".format(i + 1, episode_num))
                if print_info:
                    print("\tSection: {} / {}".format(int((i + 1) / section_size), section_num))
                    print("\tAverage Reward: {:.6f}".format(self.average_reward_trace[-1]))
                    print("\tAverage Reward of Last Section: {:.6f}".format(average_reward_section))
                    print("\tQ-table RMSE: {:.6f}".format(self.rmse_trace[-1]))
                    print("\tDifferent policy num: {}".format(policy_dif(OPTIMAL_POLICY, policy)))
                average_reward_section = 0
            else:
                average_reward_section += (reward - average_reward_section) / ((i + 1) % section_size)

            if self.dynamic_epsilon:
                self.epsilon = max(self.epsilon * self.r, self.final_epsilon)

    def save_fig(self, fig_path):
        """ Save the value function figure in the given path. """
        plot_Q(self.q_table, save_path=(fig_path + "Q_learning_" + self.name + "_value.pdf"))

    def save_result(self, log_path):
        """ Save the logs in the given path. """
        with open(log_path + "Q_learning_" + self.name + "_value.txt", "w") as f:
            f.write(str(self.q_table))
        try:
            ART = np.array(self.average_reward_trace, dtype=np.float16)
            ART_name = log_path + "Q_learning_" + self.name + "_ave_reward_{}.txt".format(len(ART))
            np.savetxt(ART_name, ART, fmt="%.4f", delimiter=",")
            RMSE = np.array(self.rmse_trace, dtype=np.float16)
            RMSE_name = log_path + "Q_learning_" + self.name + "_rmse_{}.txt".format(len(RMSE))
            np.savetxt(RMSE_name, RMSE, fmt="%.4f", delimiter=",")
        except Exception as e:
            print("No experiment value! Saving average reward trace failed.")


def main():
    env = Easy21()
    ave_rewards, rmses, names = [], [], []

    EPISODE_NUM = 100_000  # episode number
    PARAM_DICT = {"DynamicEpsilon, LearningRate=0.005, optimistic=False": {"alpha": 0.005, "optimistic":False},
                  "DynamicEpsilon, LearningRate=0.010, optimistic=False": {"alpha": 0.010, "optimistic":False},
                  "DynamicEpsilon, LearningRate=0.015, optimistic=False": {"alpha": 0.015, "optimistic":False},
                  "DynamicEpsilon, LearningRate=0.020, optimistic=False": {"alpha": 0.020, "optimistic":False},
                  "DynamicEpsilon, LearningRate=0.025, optimistic=False": {"alpha": 0.025, "optimistic":False},
                  "DynamicEpsilon, LearningRate=0.030, optimistic=False": {"alpha": 0.030, "optimistic":False},
                  "DynamicEpsilon, LearningRate=0.050, optimistic=False": {"alpha": 0.050, "optimistic":False},
                  "DynamicEpsilon, LearningRate=0.100, optimistic=False": {"alpha": 0.100, "optimistic":False},

                  "DynamicEpsilon, LearningRate=0.005, optimistic=True": {"alpha": 0.005, "optimistic":True},
                  "DynamicEpsilon, LearningRate=0.010, optimistic=True": {"alpha": 0.010, "optimistic":True},
                  "DynamicEpsilon, LearningRate=0.015, optimistic=True": {"alpha": 0.015, "optimistic":True},
                  "DynamicEpsilon, LearningRate=0.020, optimistic=True": {"alpha": 0.020, "optimistic":True},
                  "DynamicEpsilon, LearningRate=0.025, optimistic=True": {"alpha": 0.025, "optimistic":True},
                  "DynamicEpsilon, LearningRate=0.030, optimistic=True": {"alpha": 0.030, "optimistic":True},
                  "DynamicEpsilon, LearningRate=0.050, optimistic=True": {"alpha": 0.050, "optimistic":True},
                  "DynamicEpsilon, LearningRate=0.100, optimistic=True": {"alpha": 0.100, "optimistic":True},


                  "Epsilon, LearningRate=0.020, Epsilon=0.001, optimistic=False": {"de": False, "e": 0.001, "optimistic":False},
                  "Epsilon, LearningRate=0.020, Epsilon=0.005, optimistic=False": {"de": False, "e": 0.001, "optimistic":False},
                  "Epsilon, LearningRate=0.020, Epsilon=0.01, optimistic=False": {"de": False, "e": 0.01, "optimistic":False},
                  "Epsilon, LearningRate=0.020, Epsilon=0.05, optimistic=False": {"de": False, "e": 0.01, "optimistic":False},
                  "Epsilon, LearningRate=0.020, Epsilon=0.1, optimistic=False": {"de": False, "e": 0.1, "optimistic":False},
                  "Epsilon, LearningRate=0.020, Epsilon=0.5, optimistic=False": {"de": False, "e": 0.1, "optimistic":False},
                  "Epsilon, LearningRate=0.020, Epsilon=1.0, optimistic=False": {"de": False, "e": 1., "optimistic":False},

                  "Epsilon, LearningRate=0.020, Epsilon=0.001, optimistic=True": {"de": False, "e": 0.001, "optimistic":True},
                  "Epsilon, LearningRate=0.020, Epsilon=0.005, optimistic=True": {"de": False, "e": 0.001, "optimistic":True},
                  "Epsilon, LearningRate=0.020, Epsilon=0.01, optimistic=True": {"de": False, "e": 0.01, "optimistic":True},
                  "Epsilon, LearningRate=0.020, Epsilon=0.05, optimistic=True": {"de": False, "e": 0.01, "optimistic":True},
                  "Epsilon, LearningRate=0.020, Epsilon=0.1, optimistic=True": {"de": False, "e": 0.1, "optimistic":True},
                  "Epsilon, LearningRate=0.020, Epsilon=0.5, optimistic=True": {"de": False, "e": 0.1, "optimistic":True},
                  "Epsilon, LearningRate=0.020, Epsilon=1.0, optimistic=True": {"de": False, "e": 1., "optimistic":True},
                  }  # dict of 'name: param'

    for name, param in PARAM_DICT.items():
        print("#" * 5, name, "#" * 5)
        model = QLearning(env, episode_num=EPISODE_NUM,
                          learning_rate=param.get("alpha", 0.01), 
                          dynamic_epsilon=param.get("de", True),
                          epsilon=param.get("e", 0.9), 
                          optimistic=param.get("optimistic"))
        model.run(print_info=False)
        ave_rewards.append(model.average_reward_trace)
        rmses.append(model.rmse_trace)
        names.append(name)
        model.save_fig(fig_path)
        model.save_result(log_path)

    plot_learning_curve(ave_rewards[:5], rmses[:5], names[:5],
                        save_path=fig_path + "Q_learning_different_alpha_CMP.pdf", show=False)
    plot_learning_curve(ave_rewards[5:], rmses[5:], names[5:],
                        save_path=fig_path + "Q_learning_different_epsilon_CMP.pdf", show=False)

    result = []
    for i in range(len(ave_rewards)):
        result.append((names[i], ave_rewards[i][-1]))
    result.sort(key=lambda x: x[1])
    for name, reward in result:
        print(name)
        print("\tAverage Reward: {:.4f}".format(reward))


if __name__ == '__main__':
    main()
