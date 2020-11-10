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


class MCMC:
    def __init__(self, env: Easy21, learning_rate=0.05, reward_decay=1.):
        """
        :param learning_rate: <float> learning rate alpha.
        :param reward_decay: <float> reward decay factor gamma.
        """
        self.env = env
        self.action_space = Easy21.action_space
        self.action_num = Easy21.action_num
        self.state_space = Easy21.state_space
        self.non_terminal_state_space = Easy21.non_terminal_state_space
        self.terminal_state_space = Easy21.terminal_state_space
        self.state_num = len(self.state_space)

        self.q_table = {state: [1, 1] for state in self.non_terminal_state_space}
        self.count = {state: [1, 1] for state in self.non_terminal_state_space}
        self.gamma = reward_decay

        self.reward_trace = []
        self.average_reward_trace = []
        self.rmse_trace = []

    def _update_q_table(self, state_, action_, reward_, next_state_):
        """ Update Q table. """
        state_ = (state_[0], state_[1])
        next_state_, game_end = (next_state_[0], next_state_[1]), next_state_[2]
        self.count[state_][action_] += 1

        q_target = reward_ if game_end == 1 else reward_ + self.gamma * max(self.q_table[next_state_])
        q_predict = self.q_table[state_][action_]

        # self.q_table[state_][action_] += self.lr * (q_target - q_predict)
        self.q_table[state_][action_] += 1 / self.count[state_][action_] * (q_target - q_predict)

    def choose_action_random(self, state_):
        """ Choose random action. """
        return np.random.randint(self.action_num)

    def choose_action_deterministic(self, state_):
        """ Choose action of the given state according to Q-table. """
        state_ = (state_[0], state_[1])
        return np.argmax(self.q_table[state_])

    def run_one_step(self, state, update):
        """ Run one step on the environment and return the reward. """
        if update:
            action = self.choose_action_random(state)
            next_state, reward = self.env.step(state, action)
            self._update_q_table(state, action, reward, next_state)
        else:
            action = self.choose_action_deterministic(state)
            next_state, reward = self.env.step(state, action)
        return next_state, reward

    def run_one_episode(self, update):
        """ Run one step on the environment and return the final reward. """
        state, reward = self.env.reset(), 0
        while state[2] == 0:  # game not ends
            state, reward = self.run_one_step(state, update=update)
        return reward

    def run(self, episode_num=1_000_000, update=True):
        """ Run Q-learning algorithm. 

        :param episode_num: <int> the number of episodes to run.
                                  If left None, then the number will be decided by the default number self.episode_num
        :param update: <bool> whether to update Q value in the running process.
        """
        average_reward = 0
        for i in range(episode_num):
            if (i + 1) % (episode_num // 10) == 0:
                print("Episode: {}".format(i + 1))
                print("\tAverage Reward: {}".format(self.average_reward_trace[-1]))
                print("\tQ-table RMSE: {:.6f}".format(self.rmse_trace[-1]))
                time.sleep(0.5)
            reward = self.run_one_episode(update=update)
            self.reward_trace.append(reward)
            average_reward += (reward - average_reward) / (i + 1)
            self.average_reward_trace.append(average_reward)
            self.rmse_trace.append(rms_error_q(OPTIMAL_VALUE, self.q_table))
        return average_reward

    def save_fig(self, fig_path):
        """ Save the value function figure in the given path. """
        plot_Q(self.q_table, save_path=(fig_path + "MCMC_optimistice_True_value.pdf"))

    def save_result(self, log_path):
        """ Save the logs in the given path. """
        with open(log_path + "MCMC_optimistice_True_value.txt", "w") as f:
            f.write(str(self.q_table))
        try:
            ART = np.array(self.average_reward_trace, dtype=np.float16)
            ART = ART[len(ART) // 2:]
            ART_name = log_path + "MCMC_optimistice_True_ave_reward_{}.txt".format(len(ART))
            np.savetxt(ART_name, ART, fmt="%.4f", delimiter=",")
            RMSE = np.array(self.rmse_trace, dtype=np.float16)
            RMSE = RMSE[:len(ART)]
            RMSE_name = log_path + "MCMC_optimistice_True_rmse_{}.txt".format(len(RMSE))
            np.savetxt(RMSE_name, RMSE, fmt="%.4f", delimiter=",")
        except Exception as e:
            print("No experiment value! Saving average reward trace failed.")


def main():
    env = Easy21()
    model = MCMC(env)
    model.run(episode_num=1_000_000)
    model.run(episode_num=1_000_000, update=False)

    model.save_fig(fig_path)
    model.save_result(log_path)

    ave_rewards, rmses, names = [model.average_reward_trace[1_000_000:]], [model.rmse_trace[:1_000_000]], ["MCMC"]
    plot_learning_curve(ave_rewards, rmses, names, save_path=fig_path + "MCMC_optimistice_True_reward.pdf", show=False)


if __name__ == "__main__":
    main()
