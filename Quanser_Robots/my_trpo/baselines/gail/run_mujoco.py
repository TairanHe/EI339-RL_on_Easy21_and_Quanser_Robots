'''
Disclaimer: this code is highly based on trpo_mpi at @openai/baselines and @openai/imitation
'''

# import argparse
# import os.path as osp
# import logging
# from mpi4py import MPI
# from tqdm import tqdm

import argparse
import os.path as osp
import os,sys,inspect
import logging
from mpi4py import MPI
from tqdm import tqdm
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)


import numpy as np
import gym

from baselines.gail import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.dataset_plus import Dataset
from baselines.common.misc_util import boolean_flag
from baselines import bench
from baselines import logger
from baselines.gail.adversary import TransitionClassifier

#from baselines.run import get_exp_data
from baselines.ebil.utils import get_dirs, Data_dir, get_exp_data

import pickle
def get_exp_data2(expert_path):
    print("-------------------enter_data2\n")
    with open(expert_path, 'rb') as f:
        data = pickle.loads(f.read())

        data["actions"] = np.squeeze(data["actions"])
        data["observations"] = data["observations"]

        # print(data["observations"].shape)
        # print(data["actions"].shape)
        return [data["observations"], data["actions"]]

Log_dir = osp.expanduser("~/workspace/log/mujoco")
Checkpoint_dir = osp.expanduser("~/workspace/checkpoint/mujoco")

Log_dir = osp.expanduser("../../logs/mujoco/red/")
GAIL_Log_dir = osp.expanduser("../../logs/mujoco/gail/")
GMMIL_Log_dir = osp.expanduser("../../logs/mujoco/gmmil/")
AIRL_Log_dir = osp.expanduser("../../logs/mujoco/airl/")

Model_dir = osp.expanduser("../../models/mujoco/red")
GAIL_Model_dir = osp.expanduser("../../models/mujoco/gail")
GMMIL_Model_dir = osp.expanduser("../../models/mujoco/gmmil")
AIRL_Model_dir = osp.expanduser("../../models/mujoco/airl/")

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--env_id', help='environment ID', default="Hopper-v2")
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--model_dir', help='the directory to save model', default=GAIL_Model_dir)
    parser.add_argument('--log_dir', help='the directory to save log file', default=GAIL_Log_dir)
    parser.add_argument('--data_dir', help='the directory to save exp data', default=Data_dir)
    parser.add_argument('--expert_path', type=str, default='../../data/Reacher-v2.pkl')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    #parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=5)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--adversary_hidden_size', type=int, default=100)
    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['trpo', 'ppo'], default='trpo')
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=100)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=5e6)
    # Behavior Cloning
    boolean_flag(parser, 'pretrained', default=False, help='Use BC to pretrain')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e4)


    # My GAIL
    parser.add_argument('--num_trajs', help='number of exp data', type=int, default=4)
    parser.add_argument('--num_iters', help='number of iters per episode', type=int, default=6e3)
    return parser.parse_args()


def get_task_name(args):
    task_name = args.algo + "_gail."
    if args.pretrained:
        task_name += "with_pretrained."
    task_name += args.env_id.split("-")[0]
    # task_name = task_name + ".g_step_" + str(args.g_step) + ".d_step_" + str(args.d_step) + \
    #     ".policy_entcoeff_" + str(args.policy_entcoeff) + ".adversary_entcoeff_" + str(args.adversary_entcoeff)
    # task_name += ".seed_" + str(args.seed)
    return task_name


def main(args):
    set_global_seeds(args.seed)
    env = gym.make(args.env_id)

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)
    # env = bench.Monitor(env, logger.get_dir() and
    #                     osp.join(logger.get_dir(), "monitor.json"))
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    if args.log_dir != Log_dir:
        log_dir = osp.join(Log_dir, args.log_dir)
        save_dir = osp.join(Checkpoint_dir, args.log_dir)
    else:
        log_dir = Log_dir
        save_dir = Checkpoint_dir

    logger.configure(dir=log_dir, log_suffix=task_name, format_strs=["log", "stdout"])

    if args.task == 'train':

        log_dir, data_dir, policy_model_dir, __, _ = get_dirs(args)
        print("log_dir: ", log_dir)
        print("model_dir: ", policy_model_dir)
        #exp_data = get_exp_data2(osp.join(osp.dirname(osp.realpath(__file__)), "../../data/mujoco/%s.pkl" % args.env_id))
        
        data_path = data_dir+'/expert_sample'
        # eric version
        exp_data = get_exp_data(data_path, args.num_trajs)
        dataset = Dataset(exp_data)
        reward_giver = TransitionClassifier(env, args.adversary_hidden_size, entcoeff=args.adversary_entcoeff)
        train(env,
              args.seed,
              policy_fn,
              reward_giver,
              dataset,
              args.algo,
              args.g_step,
              args.d_step,
              args.policy_entcoeff,
              args.num_timesteps,
              args.num_iters,
              args.save_per_iter,
              save_dir,
              args.pretrained,
              args.BC_max_iter,
              task_name
              )
    elif args.task == 'evaluate':
        runner(env,
               policy_fn,
               args.load_model_path,
               timesteps_per_batch=1024,
               number_trajs=10,
               stochastic_policy=args.stochastic_policy,
               save=args.save_sample
               )
    else:
        raise NotImplementedError
    env.close()


def train(env, seed, policy_fn, reward_giver, dataset, algo,
          g_step, d_step, policy_entcoeff, num_timesteps, num_iters, save_per_iter,
          checkpoint_dir, pretrained, BC_max_iter, task_name=None):

    pretrained_weight = None
    if pretrained and (BC_max_iter > 0):
        # Pretrain with behavior cloning
        from baselines.gail import behavior_clone
        pretrained_weight = behavior_clone.learn(env, policy_fn, dataset,
                                                 max_iters=BC_max_iter)

    if algo == 'trpo':
        from baselines.gail import trpo_mpi
        # Set up for MPI seed
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        env.seed(workerseed)
        trpo_mpi.learn(env, policy_fn, reward_giver, dataset, rank,
                       pretrained=pretrained, pretrained_weight=pretrained_weight,
                       g_step=g_step, d_step=d_step,
                       entcoeff=policy_entcoeff,
                       max_iters=num_iters,
                       ckpt_dir=checkpoint_dir,
                       save_per_iter=save_per_iter,
                       timesteps_per_batch=1024,
                       max_kl=0.01, cg_iters=10, cg_damping=0.1,
                       gamma=0.99, lam=0.97, #0.995 as default
                       vf_iters=5, vf_stepsize=1e-3,
                       task_name=task_name)
    else:
        raise NotImplementedError


def runner(env, policy_func, load_model_path, timesteps_per_batch, number_trajs,
           stochastic_policy, save=False, reuse=False):

    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=reuse)
    U.initialize()
    # Prepare for rollouts
    # ----------------------------------------
    pi.load_policy(load_model_path)

    obs_list = []
    acs_list = []
    len_list = []
    ret_list = []
    for _ in tqdm(range(number_trajs)):
        traj = traj_1_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy)
        obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
        obs_list.append(obs)
        acs_list.append(acs)
        len_list.append(ep_len)
        ret_list.append(ep_ret)
    if stochastic_policy:
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    if save:
        filename = load_model_path.split('/')[-1] + '.' + env.spec.id
        np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list),
                 lens=np.array(len_list), rets=np.array(ret_list))
    avg_len = sum(len_list)/len(len_list)
    avg_ret = sum(ret_list)/len(ret_list)
    print("Average length:", avg_len)
    print("Average return:", avg_ret)
    print("std:", np.std(ret_list))
    return avg_len, avg_ret


# Sample one expert_trajectory (until expert_trajectory end)
def traj_1_generator(pi, env, horizon, stochastic):

    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset()
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []

    while True:
        ac, vpred = pi.act(stochastic, ob)
        obs.append(ob)
        news.append(new)
        acs.append(ac)

        ob, rew, new, _ = env.step(ac)
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            break
        t += 1

    obs = np.array(obs)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len}
    return traj


if __name__ == '__main__':
    args = argsparser()
    main(args)
