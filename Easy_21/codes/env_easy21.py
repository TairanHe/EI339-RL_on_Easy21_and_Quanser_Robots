import numpy as np
from tools import *


class Easy21:
    action_space = ["stick", "hit"]  # 0 for stick (do not draw card) and 1 for hit (draw card)
    action_num = len(action_space)
    non_terminal_state_space = [(i, j) for i in range(1, 22) for j in range(1, 11)]
    terminal_state_space = [(i, j) for i in range(1, 22) for j in range(16, 22)] + \
                           [(-1, j) for j in range(1, 11)] + [(i, -1) for i in range(1, 22)]  # -1 for bust
    state_space = non_terminal_state_space + terminal_state_space
    state_num = len(state_space)

    def __init__(self):
        # This experiment requires explicit input and output of states
        pass

    def step(self, state_, action_):
        """ Take one step with the given (player's) action.

        :param state_: <tup> the state:
                           state_[0] for player's sum,
                           state_[1] for dealer's sum,
                           state_[2] for game end or not, 0 for game not ends and 1 for game ends
        :param action_: <int> 0 for stick (do not draw card) and 1 for hit (draw card)
        :return: tuple:
                     state: <tup> the next state after the action,
                     reward: <int> the reward,
                     finished : <bool> the flag for whether the game is finished.
        """
        if action_ == 0:  # player sticks
            while 1 <= state_[1] < 16:
                draw = self._draw()
                state_ = (state_[0], state_[1] + draw, 1)
        else:  # player hits
            draw = self._draw()
            game_end = 1 if (state_[0] + draw < 1 or state_[0] + draw > 21) else 0
            state_ = (state_[0] + draw, state_[1], game_end)
        return state_, self._reward(state_)

    def _draw(self) -> int:
        """ Draw one card.

        The num of card is uniformly distributed between [1, 10], and the color is red (p = 1/3) or black (p = 2/3).
        If the color is red, then the sum is subtracted, else the sum is added, by the number of card.

        :return: <int> point of card
        """
        sign = 1 if np.random.randint(0, 3) <= 1 else -1  # sign is '+' with p = 2/3, and sign is '-' with p = 1/3
        num = np.random.randint(1, 11)
        return sign * num

    def _reward(self, state_):
        """ Return the final reward.

        :param: state_: the state
        :return: <int> the reward
        """
        if state_[2] != 1:  # game not ended, reward is 0
            return 0
        elif state_[0] > 21 or state_[0] < 1:  # player busts
            return -1
        elif state_[1] > 21 or state_[1] < 1:  # dealer busts
            return 1
        elif state_[0] == state_[1]:  # no busts, sum is equal
            return 0
        elif state_[0] > state_[1]:  # no busts, player's sum is larger
            return 1
        else:  # no busts, dealer's sum is larger
            return -1

    def reset(self):
        """ Reset the environment.

        :return: the (randomly) reset state
        """
        return np.random.randint(1, 11), np.random.randint(1, 11), 0

    """
    Below is to calculate the transition matrix.
    This is useful in value iteration and policy iteration
    """
    @lazy
    def trans_mat_player(self):
        mat = np.zeros([22, 22], dtype=np.float64)
        for i in range(1, 12):
            mat[i - 1, :] = np.array([1 / 30] * (i - 1) + [0.] + [2 / 30] * 10 + [0.] * (11 - i) + [(11 - i) / 30])
        for i in range(12, 22):
            mat[i - 1, :] = np.array([0.] * (i - 11) + [1 / 30] * 10 + [0.] + [2 / 30] * (21 - i) + [(i * 2 - 22) / 30])
        mat[21, 21] = 1.
        return mat

    @lazy
    def trans_mat_dealer(self):
        mat = np.zeros([22, 22], dtype=np.float64)
        for i in range(1, 12):
            mat[i - 1, :] = np.array([1 / 30] * (i - 1) + [0.] + [2 / 30] * 10 + [0.] * (11 - i) + [(11 - i) / 30])
        for i in range(12, 16):
            mat[i - 1, :] = np.array([0.] * (i - 11) + [1 / 30] * 10 + [0.] + [2 / 30] * (21 - i) + [(i * 2 - 22) / 30])
        for i in range(16, 23):
            mat[i - 1, i - 1] = 1.
        delta = 1
        while delta > 1e-6:
            mat_ = np.matmul(mat, mat)
            delta = np.sum(np.abs(mat - mat_))
            mat = mat_
        return mat

    @lazy
    def trans_mat(self):
        trans_mat = dict()
        for state in self.non_terminal_state_space:
            player_sum, dealer_sum = state
            if player_sum == -1 or dealer_sum == -1:
                continue
            res_player, res_dealer = list(), list()
            for next_player_sum, prob in enumerate(self.trans_mat_player[player_sum - 1]):
                if prob > 1e-7:
                    if next_player_sum != 21:
                        next_state = (next_player_sum + 1, dealer_sum, 0)
                        res_player.append((prob, str(next_state), 0))
                    else:
                        next_state = (-1, dealer_sum, 1)
                        res_player.append((prob, str(next_state), -1))

            for next_dealer_sum, prob in enumerate(self.trans_mat_dealer[dealer_sum - 1]):
                if prob > 1e-7:
                    if next_dealer_sum != 21:
                        next_state = (player_sum, next_dealer_sum + 1, 1)
                        res_dealer.append((prob, str(next_state), self._reward(next_state)))
                    else:
                        next_state = (player_sum, -1, 1)
                        res_dealer.append((prob, str(next_state), 1))
            trans_mat[state] = [res_dealer, res_player]
        return trans_mat
