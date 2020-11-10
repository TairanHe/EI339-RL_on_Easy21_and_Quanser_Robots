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


class PolicyIteration:
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
        self.pi = {state: [1 / self.action_num] * self.action_num for state in self.non_terminal_state_space}

    def update_value(self):
        """ Update the value function by one iteration. """
        delta = 0
        new_value = {state: 0. for state in self.non_terminal_state_space}
        for s in self.non_terminal_state_space:
            for a in range(self.action_num):
                # q(s,a) = r(s,a) + gamma * sum{ p(s'|s,a) * v(s') }
                #        = sum{ p(s'|s,a) * [ r(s'|s,a) + gamma * v(s') ] }
                qsa = 0
                for prob, next_state, reward in self.env.trans_mat[s][a]:
                    next_state, game_end = (next_state[0], next_state[1]), next_state[2]
                    if game_end == 0:  # game not ends
                        qsa += prob * (reward + self.gamma * self.value[next_state])
                    else:  # game ends
                        qsa += prob * reward
                # v(s) = sum{ pi(a|s) * q(s,a) }
                new_value[s] += self.pi[s][a] * qsa
            delta = max(delta, abs(new_value[s] - self.value[s]))
        self.value = new_value
        return delta

    def policy_evaluation(self):
        """ Update the value function until difference is less than theta. """
        cnt = 1
        while self.update_value() > self.theta:
            print("Eval: {:d}, RMSE: {:.4f}".format(cnt, rms_error(OPTIMAL_VALUE, self.value)))
            cnt += 1
        print("Eval: {:d}, RMSE: {:.4f}".format(cnt, rms_error(OPTIMAL_VALUE, self.value)))
        return cnt

    def policy_improvement(self):
        """ Update the policy by one iteration. """
        stable = True
        for s in self.non_terminal_state_space:
            qsa_list = []
            for a in range(self.action_num):
                qsa = 0
                # q(s,a)=sum(p*(r+gamma*v(s'))
                for prob, next_state, reward in self.env.trans_mat[s][a]:
                    next_state, game_end = (next_state[0], next_state[1]), next_state[2]
                    if game_end == 0:  # game not ends
                        qsa += prob * (reward + self.gamma * self.value[next_state])
                    else:  # game ends
                        qsa += prob * reward
                qsa_list.append(qsa)
            # pi(s) = argmax{ q(s,a) }
            max_q = max(qsa_list)
            cnt_q = qsa_list.count(max_q)
            # Uniformly divide probs between actions with highest Q(s,a)
            new_pi = [1 / cnt_q if q == max_q else 0 for q in qsa_list]
            if self.pi[s] != new_pi:
                stable = False
                self.pi[s] = new_pi
        return stable

    def policy_iteration(self):
        """ Update the policy until stable. """
        cnt = 0
        stable = False
        while not stable:
            print("---- %d iterations----" % cnt)
            self.policy_evaluation()
            stable = self.policy_improvement()
            policy = {k: 0 if self.pi[k][0] > self.pi[k][1] else 1 for k in self.non_terminal_state_space}
            dif_num = policy_dif(OPTIMAL_POLICY, policy)
            print("Different policy num: {}".format(dif_num))
            # if dif_num == 0:  break
            cnt += 1

    def test_policy(self, episode_num=1_000_000):
        """ Test current policy on real environment by certain times. 

        :param episode_num: <int> the number of episodes to run.
        """
        average_reward = 0
        for i in range(episode_num):
            if (i + 1) % (episode_num // 10) == 0:
                print("Episode: {}".format(i + 1))
                print("Average Reward: {}".format(average_reward))
            state, reward = self.env.reset(), 0
            while state[2] == 0:
                action = 1 if self.pi[(state[0], state[1])][1] > 0.01 else 0
                state, reward = self.env.step(state, action)
            # #### Statistic average, incremental update #### #
            average_reward += (reward - average_reward) / (i + 1)
            self.average_reward_trace.append(average_reward)
        return average_reward

    def save_fig(self, fig_path):
        """ Save the value function figure in the given path. """
        plot_V(self.value, save_path=(fig_path + "policy_iter_value.pdf"))

    def save_result(self, log_path):
        """ Save the logs in the given path. """
        with open(log_path + "policy_iter_value.txt", "w") as f:
            f.write(str(self.value))
        with open(log_path + "policy_iter_policy.txt", "w") as f:
            f.write(str(self.pi))
        try:
            ART = np.array(self.average_reward_trace, dtype=np.float16)
            ART_name = log_path + "policy_iter_exp_ave_reward_{}.txt".format(len(self.average_reward_trace))
            np.savetxt(ART_name, ART, fmt="%.4f", delimiter=",")
        except Exception as e:
            print("No experiment value! Saving average reward trace failed.")


def main():
    env = Easy21()
    model = PolicyIteration(env)
    model.policy_iteration()
    model.test_policy()

    model.save_fig(fig_path)
    model.save_result(log_path)


if __name__ == '__main__':
    main()
