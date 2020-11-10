import os
import numpy as np
import matplotlib.pyplot as plt
import pandas

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
             [1500, 3000, 6000]]


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
            res[2].append(float(array[i + 2][array[i + 2].index("[avg_reward_std]:") + 1]))
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


for task in arg_list:
    for env_index in range(3):
        index = 0
        if task in arg_list:
            index = arg_list.index(task)
        else:
            print("Wrong task!")
            exit()


        path_list = []
        for k in range(len(para_list[index])):
            path = relative_path + "rs-" + env[env_index] + "/seed-0/"
            for i in range(len(arg_list)):
                if i == index:
                    path += arg_list[i] + "-" + str(para_list[i][k]) + "/"
                else:
                    path += arg_list[i] + "-" + str(default_para[i]) + "/"
            path += "random_timesteps-10000/max_timesteps-200000/0.log/0.log/logger"
            path_list.append(path)
            print(path)

        handles = []
        step_size = 5
        legend = []
        colors = ['red', 'royalblue', 'green', 'purple', 'darkorange', 'red', 'orchid', 'darkorange', 'darkcyan']

        for i in range(len(path_list)):
            path = path_list[i]
            res = read_logger(path)
            x = np.array(res[0])
            mean = np.array(res[1])
            std = np.array(res[2])
            minimum = mean - std / 2
            maximum = mean + std / 2
            ma_minimum = MA(minimum, step_size)
            ma_maximum = MA(maximum, step_size)
            ma_mean = MA(mean, step_size)
            curve = plt.fill_between(x, ma_minimum, ma_maximum, where=ma_maximum >= ma_minimum, facecolor=colors[i],
                                     interpolate=True, alpha=0.2)
            plt.plot(x, ma_mean, color=colors[i], linewidth=2)
            legend.append("reward-" + arg_list[index] + "=" + str(para_list[index][i]))
            handles.append(curve)

        plt.xlim(left=0, right=100000)
        plt.ylim(bottom=0)
        x_index = [0, 20000, 40000, 60000, 80000, 100000]
        x_label = ['0', '20K', '40K', '60K', '80K', '100K']
        plt.xticks(x_index, x_label)
        plt.legend(handles, legend)
        plt.xlabel("Timesteps")
        plt.ylabel("Average reward")
        plt.savefig("MPC-rs-" + env[env_index] + "-" + task + "-reward.pdf")
        plt.show()

        legend = []
        handles = []
        step_size = 10
        for i in range(len(path_list)):
            path = path_list[i]
            res = read_logger(path)
            x = np.array(res[0])
            loss = np.array(res[5])
            curr_loss = np.array(res[3])
            ma_loss = MA(loss, step_size)
            ma_curr_loss = MA(curr_loss, step_size)
            # print(x)
            maximum = ma_loss + abs(ma_curr_loss - ma_loss) / 2
            minimum = ma_loss - abs(ma_curr_loss - ma_loss) / 2
            plt.plot(x, ma_loss, color=colors[i], linewidth=2)
            curve = plt.fill_between(x, minimum, maximum, where=maximum >= minimum, facecolor=colors[i],
                                     interpolate=True, alpha=0.2)
            legend.append("loss-" + arg_list[index] + "=" + str(para_list[index][i]))
            handles.append(curve)

        plt.xlim(left=0, right=100000)
        plt.ylim(bottom=0)
        x_index = [0, 20000, 40000, 60000, 80000, 100000]
        x_label = ['0', '20K', '40K', '60K', '80K', '100K']
        plt.xticks(x_index, x_label)
        plt.legend(handles, legend)
        plt.xlabel("Timesteps")
        plt.ylabel("Average loss")
        plt.savefig("MPC-rs-" + env[env_index] + "-" + task + "-loss.pdf")
        plt.show()