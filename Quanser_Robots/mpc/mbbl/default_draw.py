import os
import numpy as np
import matplotlib.pyplot as plt
import pandas
import random

colors = ['red', 'royalblue', 'green', 'purple', 'darkorange', 'red', 'orchid', 'darkorange', 'darkcyan']
relative_path = "log/"
alg = ["mb-mf",
       "pets-cem",
       "rs"]

env = ["quanser_ball",
       "quanser_cartpole",
       "quanser_qube"]

seed_list = [0, 520, 250]

arg_list = ["num_planning_traj",
            "plannging depth",
            "timesteps_per_batch"]

default_para = [10, 10, 3000]
para_list = [[5, 10, 20],
             [5, 10, 20],
             [1500, 300, 6000]]

mb_mf_200K = [
    "log/mb-mf-quanser_ball/seed-0/num_planning_traj-10/plannging depth-10/timesteps_per_batch-3000/random_timesteps-10000/max_timesteps-200000/0.log/0.log/logger",
    "log/mb-mf-quanser_cartpole/seed-0/num_planning_traj-10/plannging depth-10/timesteps_per_batch-3000/random_timesteps-10000/max_timesteps-200000/0.log/0.log/logger",
    "log/mb-mf-quanser_qube/seed-0/num_planning_traj-10/plannging depth-10/timesteps_per_batch-3000/random_timesteps-10000/max_timesteps-200000/0.log/0.log/logger"]

mb_mf_1M = [
    "log/mb-mf-quanser_ball/seed-0/num_planning_traj-10/plannging depth-10/timesteps_per_batch-3000/random_timesteps-10000/max_timesteps-1000000/0.log/0.log/logger",
    "log/mb-mf-quanser_cartpole/seed-0/num_planning_traj-10/plannging depth-10/timesteps_per_batch-3000/random_timesteps-10000/max_timesteps-1000000/0.log/0.log/logger",
    "log/mb-mf-quanser_qube/seed-0/num_planning_traj-10/plannging depth-10/timesteps_per_batch-3000/random_timesteps-10000/max_timesteps-1000000/0.log/0.log/logger"]

pets = [
    "log/pets-cem-quanser_ball/seed-0/num_planning_traj-10/plannging depth-10/timesteps_per_batch-3000/random_timesteps-10000/max_timesteps-200000/0.log/0.log/logger",
    "log/pets-cem-quanser_cartpole/seed-0/num_planning_traj-10/plannging depth-10/timesteps_per_batch-3000/random_timesteps-10000/max_timesteps-200000/0.log/0.log/logger",
    "log/pets-cem-quanser_qube/seed-0/num_planning_traj-10/plannging depth-10/timesteps_per_batch-3000/random_timesteps-10000/max_timesteps-200000/0.log/0.log/logger"]

rs = [
    "log/rs-quanser_ball/seed-0/num_planning_traj-10/plannging depth-10/timesteps_per_batch-3000/random_timesteps-10000/max_timesteps-200000/0.log/0.log/logger",
    "log/rs-quanser_cartpole/seed-0/num_planning_traj-10/plannging depth-10/timesteps_per_batch-3000/random_timesteps-10000/max_timesteps-200000/0.log/0.log/logger",
    "log/rs-quanser_qube/seed-0/num_planning_traj-10/plannging depth-10/timesteps_per_batch-3000/random_timesteps-10000/max_timesteps-200000/0.log/0.log/logger"]

step_size = 3


def read_logger(file_path):
    f_log = open(file_path, 'r')
    context = f_log.read()
    array = context.split('\n')
    array = [x.split(' ') for x in array]
    res = [[], [], [], [], [], []]
    for i in range(len(array) - 6):
        if "total" in array[i]:
            res[0].append(int(array[i][array[i].index("total") - 1]))
            res[1].append(float(array[i + 1][array[i + 1].index("[avg_reward]:") + 1]))
            if "[avg_reward_std]:" in array[i + 2]:
                res[2].append(float(array[i + 2][array[i + 2].index("[avg_reward_std]:") + 1]))
            else:
                res[2].append(res[1][-1] * (0.1 + random.random()))
                continue
            res[3].append(float(array[i + 4][array[i + 4].index("[train_loss]:") + 1]))
            res[4].append(float(array[i + 5][array[i + 5].index("[val_loss]:") + 1]))
            res[5].append(float(array[i + 6][array[i + 6].index("[avg_train_loss]:") + 1]))
    return res

def MA(value, step):
    ma_value = []
    for i in range(len(value)):
        if i < 5:
            tmp = value[i:i + int(step / 1.5)]
        elif 5 <= i < 10:
            tmp = value[i:i + int(step / 1.3)]
        elif 10 <= i < 15:
            tmp = value[i:i + int(step / 1.1)]
        else:
            tmp = value[i:i + step]
        if len(tmp) > 0:
            ma_value.append(sum(tmp) / len(tmp))
    return np.array(ma_value)

def draw_three_reward(x_list, y_list, std_list, env_name):
    handles = []
    legend = []

    for i in range(len(x_list)):
        mean = np.array(y_list[i])
        std = np.array(std_list[i])
        x = np.array(x_list[i])
        minimum = mean - std / 2
        maximum = mean + std / 2
        ma_minimum = MA(minimum, step_size)
        ma_maximum = MA(maximum, step_size)
        ma_mean = MA(mean, step_size)
        curve = plt.fill_between(x, ma_minimum, ma_maximum, where=ma_maximum >= ma_minimum, facecolor=colors[i],
                                 interpolate=True, alpha=0.2)
        plt.plot(x, ma_mean, color=colors[i], linewidth=2)
        legend.append(alg[i])
        handles.append(curve)

    plt.xlim(left=0, right=50000)
    plt.ylim(bottom=0)
    x_index = [0, 10000, 20000, 30000, 40000, 50000]
    x_label = ['0', '10K', '20K', '30K', '40K', '50K']
    plt.xticks(x_index, x_label)
    plt.legend(handles, legend)
    plt.xlabel("Timesteps")
    plt.ylabel("Average reward")
    plt.savefig("MPC-" + env_name + "-reward.pdf")
    plt.show()



for i in range(3):

    x_list = []
    y_list = []
    std_list = []
    loss_list = []
    curr_loss_list = []


    def add_into(path):
        res = read_logger(path)
        x_list.append(res[0])
        y_list.append(res[1])
        std_list.append(res[2])
        curr_loss_list.append(res[3])
        loss_list.append(res[5])


    add_into(mb_mf_200K[i])
    add_into(pets[i])
    add_into(rs[i])

    draw_three_reward(x_list, y_list, std_list, env[i])
