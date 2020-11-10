import os
import numpy as np
import matplotlib.pyplot as plt
import pandas

relative_path = "logs/quanser/trpo/"
env = ["BallBalancerSim",
       "CartpoleSwingShort",
       "Qube-100"]

seed_list = [0, 520, 250]

arg_list = ["policy_hidden_layer",
            "policy_hidden_size",
            "batchsize",
            "gamma",
            "vf_stepsize",
            "policy_entcoeff",
            "vf_iters",
            "max_kl",
            "cg_iters",
            "cg_damping",
            "lam"]

file_name = ["BallBalancerSim", "CartpoleSwingShort", "Qube"]
default_para = [3, 128, 1024, 0.99, 0.0003, 0, 3, 0.001, 10, 0.01, 1.0]
para_list = [[1, 2, 3, 5],
             [32, 64, 128, 256],
             [256, 512, 1024, 2048],
             [0.9, 0.95, 0.97, 0.99],
             [0.0001, 0.0003, 0.01, 0.05],
             [0, 0.01, 0.05, 0.2],
             [1, 3, 5, 10],
             [0.0001, 0.001, 0.01, 0.05],
             [10],
             [0.01],
             [0.9, 0.95, 0.97, 1.0]]


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
            path = relative_path + env[env_index] + "-v0/seed-0/mlp/"
            for i in range(len(arg_list)):
                if i == index:
                    path += arg_list[i] + "-" + str(para_list[i][k]) + "/"
                else:
                    path += arg_list[i] + "-" + str(default_para[i]) + "/"
            path += "progress_" + file_name[env_index] + ".csv"
            path_list.append(path)

        print(env[env_index], task)
        handles = []
        step_size = 10
        legend = []
        colors = ['red', 'royalblue', 'green', 'purple', 'darkorange', 'red', 'orchid', 'darkorange', 'darkcyan']
        for i in range(len(path_list)):
            path = path_list[i]
            returns = pandas.read_csv(path)
            mean = np.array(list(returns["EpRewMean"]))
            std = np.array(list(returns["EpRewStd"]))
            x = np.array(list(returns["TimestepsSoFar"]))
            minimum = mean - std / 2
            maximum = mean + std / 2
            ma_minimum = MA(minimum, step_size)
            ma_maximum = MA(maximum, step_size)
            ma_mean = MA(mean, step_size)
            curve = plt.fill_between(x, ma_minimum, ma_maximum, where=ma_maximum >= ma_minimum, facecolor=colors[i],
                                     interpolate=True, alpha=0.2)
            plt.plot(x, ma_mean, color=colors[i], linewidth=2, label=arg_list[index] + "=" + str(para_list[index][i]))
            legend.append(arg_list[index] + "=" + str(para_list[index][i]))
            handles.append(curve)

        plt.xlim(left=0, right=1000000)
        plt.ylim(bottom=0)
        x_index = [0, 200000, 400000, 600000, 800000, 1000000]
        x_label = ['0', '200K', '400K', '600K', '800K', '1M']
        plt.xticks(x_index, x_label)
        plt.legend(handles, legend)
        plt.xlabel("Timesteps")
        plt.ylabel("Average reward")
        plt.savefig("TRPO-" + env[env_index][:4] + "-" + task + ".pdf")
        plt.show()
        # input()
