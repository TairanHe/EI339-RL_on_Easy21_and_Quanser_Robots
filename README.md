# EI339 RL on Easy21 and Quanser Robots
Project of EI339.  

Reinforcement learning implementation on Easy 21 and Quanser Robots.

## Environment Set Up
#### Install MBBL module:

Enter ```Quanser_Robots/mpc/mbbl/```, and run:
```
pip install -e .
```

#### Install Quanser Robots envirinment:

Enter ```Quanser_Robots/mpc/mbbl/env/quanser_env/```, and run:
```
pip install -e .
```

#### Tensorflow version:
```
pip install tensorflow==1.14.0 or pip install tensorflow-gpu==1.14.0
```

## Run Easy 21 
Enter ```Easy_21/```, and run:
```
bash run.sh
```
This script will return the result of ```MC```, ```Value iteration```, ```Policy iteration```, ```Q-learning``` and ```SARSA```.

You can have the 3-D value function result:

<div class="test">
<img src="https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/EASY_21/MCMC_value-crop.png" width="19%" height="19%">
<img src="https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/EASY_21/optimal_value-crop.png" width="19%" height="19%">
<img src="https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/EASY_21/policy_iter_value-crop.png" width="19%" height="19%">
<img src="https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/EASY_21/Q_learning_LR=0.025_DE_optimistic_False_value-crop.png" width="19%" height="19%">
<img src="https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/EASY_21/SARSA_LR=0.025_DE_optimistic_False_value-crop.png" width="19%" height="19%">
</div>



<!-- ![Value Function of MC](https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/EASY_21/MCMC_value.png){:height=19%" width=19%"}
![Value Function of Value iteration](https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/EASY_21/optimal_value.png){:height=19%" width=19%"}
![Value Function of Policy iteration](https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/EASY_21/policy_iter_value.png){:height=19%" width=19%"}
![Value Function of Q-learning](https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/EASY_21/Q_learning_LR=0.025_DE_optimistic_False_value.png){:height=19%" width=19%"}
![Value Function of SARSA](https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/EASY_21/SARSA_LR=0.025_DE_optimistic_False_value.png){:height=19%" width=19%"} -->
 
 You may check more visualized result at ```Easy_21/fig/``` of different hyperparameters and the impact of ```optimistic initialization``` on ```Q-learning``` and ```SARSA``` . 

## Run TRPO on Quanser Robots 
Enter ```Quanser_Robots/my_trpo/scripts```, and run the scripts of different experiment tests. 

First run the default TRPO baselines:
```
bash default.sh
```
Then for example, you may run test on batchsize of TRPO with three random seeds:
```
bash batchsize.sh Qube-100-v0
bash batchsize.sh BallBalancerSim-v0
bash batchsize.sh CartpoleSwingShort-v0
```

Then enter ```Quanser_Robots/my_trpo/```, and run the plot script:
```
python trpo_draw.py
```

You may have visulized result of three environments, for example, batchsize as follows:

<div class="test">
<img src="https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/TRPO/TRPO-Qube-batchsize.png" width="32%" height="32%">
<img src="https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/TRPO/TRPO-Ball-batchsize.png" width="32%" height="32%">
<img src="https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/TRPO/TRPO-Cart-batchsize.png" width="32%" height="32%">
</div>
<!-- ![TRPO_batchsize_Qube](https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/TRPO/TRPO-Qube-batchsize.pdf){:height=32%" width=32%"}
![TRPO_batchsize_Ball](https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/TRPO/TRPO-Ball-batchsize.pdf){:height=32%" width=32%"}
![TRPO_batchsize_Cart](https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/TRPO/TRPO-Cart-batchsize.pdf){:height=32%" width=32%"} -->

We have conducted 9 hyperparamter scripts: ```batchsize.sh```, ```gamma.sh```, ```hidden_layer.sh```, ```hidden_size.sh```, ```lam.sh```, ```max_kl.sh```, ```policy_entcoeff.sh```, ```vf_iters.sh```, ```vf_stepsize.sh```.

## Run MPC on Quanser Robots 
Enter ```Quanser_Robots/mpc/scripts```, and run the scripts of different experiment tests. 

#### MPC-RS (Random Shooting)
Enter ```Quanser_Robots/mpc/scripts/rs```

First run the default MPC-RS (Random Shooting) baselines:

```
bash default.sh
```
Then for example, you may run test on planning depth of MPC-RS:
```
bash planning_depth.sh quanser_qube
bash planning_depth.sh quanser_ball
bash planning_depth.sh quanser_cartpole
```

Then enter ```Quanser_Robots/mpc/mbbl/```, and run the plot script:
```
python mpc_draw.py
```

You may have the visulized results of three environments, for example, planning depth as follows:

<div class="test">
<img src="https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/RS/MPC-rs-quanser_qube-plannging depth-reward.png" width="32%" height="32%">
<img src="https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/RS/MPC-rs-quanser_ball-plannging depth-reward.png" width="32%" height="32%">
<img src="https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/RS/MPC-rs-quanser_cartpole-plannging depth-reward.png" width="32%" height="32%">
</div>

<!-- 

![MPC_planning_depth_Ball](https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/RS/MPC-rs-quanser_ball-plannging depth-reward.pdf){:height=32%" width=32%"}
![MPC_planning_depth_Cart](https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/RS/MPC-rs-quanser_cartpole-plannging depth-reward.pdf){:height=32%" width=32%"} -->
We have conducted 3 hyperparamter scripts: ```num_planning_traj.sh```, ```planning_depth.sh``` and ```timesteps_per_batch.sh```.

#### MPC-MB-MF (Mode-Free Model-Based)
Enter ```Quanser_Robots/mpc/scripts/mb-mf```

First run the default MPC-MB-MF (Mode-Free Model-Based) baselines:

```
bash quanser_qube.sh
bash quanser_ball.sh
bash quanser_cartpole.sh
```

#### MPC-PETS-CEM (Probabilistic Ensembles with Trajectory Sampling)
Enter ```Quanser_Robots/mpc/scripts/pets-cem```

First run the default MPC-PETS-CEM (Probabilistic Ensembles with Trajectory Sampling) baselines:

```
bash quanser_qube.sh
bash quanser_ball.sh
bash quanser_cartpole.sh
```


You can have the visualized comparison between MPC-RS, MPC-MB-MF and MPC-PETS-CEM:

<div class="test">
<img src="https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/MPC/MPC-quanser_qube-reward.png" width="32%" height="32%">
<img src="https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/MPC/MPC-quanser_ball-reward.png" width="32%" height="32%">
<img src="https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/MPC/MPC-quanser_cartpole-reward.png" width="32%" height="32%">
</div>

<!-- ![MPC_planning_depth_Qube](https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/MPC/MPC-quanser_qube-reward.pdf){:height=32%" width=32%"}
![MPC_planning_depth_Qube](https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/MPC/MPC-quanser_ball-reward.pdf){:height=32%" width=32%"}
![MPC_planning_depth_Qube](https://github.com/TairanHe/EI339_RL_on_Easy21_and_Quanser_Robots/blob/master/fig/MPC/MPC-quanser_cartpole-reward.pdf){:height=32%" width=32%"}
 -->
<!-- ##Video demo
[Qube_Qube](https://www.baidu.com)
[Qube_Qube](https://www.baidu.com)
[Qube_Qube](https://www.baidu.com) -->
## Referrence

This code repo is based on other public implementations:

- [OpenAI baselines](https://github.com/openai/baselines) 
- [MBBL](https://github.com/WilsonWangTHU/mbbl) 




