'''
Disclaimer: this code is highly based on trpo_mpi at @openai/baselines and @openai/imitation
'''

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
import quanser_robots
from utils import *
from baselines.quanser_main import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines import bench
from baselines import logger
import tensorflow as tf


import pickle




TRPO_Log_dir = osp.expanduser("../logs/quanser/trpo/")


TRPO_Model_dir = osp.expanduser("../models/quanser/trpo")


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of TRPO")
    parser.add_argument('--env_id', help='environment ID', default="Hopper-v2")
    parser.add_argument('--seed', help='random seed', type=int, default=0)
    parser.add_argument('--model_dir', help='the directory to save model', default=TRPO_Model_dir)
    parser.add_argument('--log_dir', help='the directory to save log file', default=TRPO_Log_dir)
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    # Network Configuration (Using MLP Policy)

    # Algorithms Configuration
    parser.add_argument('--network', help='type of policy network', default="mlp")
    parser.add_argument('--policy_hidden_layer', type=int, default=3)
    parser.add_argument('--policy_hidden_size', type=int, default=128)
    parser.add_argument('--batchsize', help='timesteps per batch', type=int, default=1024)
    parser.add_argument('--gamma', help='Discount factor', type=float, default=0.99)
    parser.add_argument('--vf_stepsize', type=float, default=3e-4)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)
    parser.add_argument('--vf_iters', type=int, default=3)
    parser.add_argument('--max_kl', type=float, default=0.001)
    parser.add_argument('--cg_iters', type=int, default=10)
    parser.add_argument('--cg_damping', type=int, default=0.01)
    parser.add_argument('--lam', type=float, default=1.0)
    boolean_flag(parser, 'fixed_var', default=False, help='Fixed policy variance')
    boolean_flag(parser, 'popart', default=True, help='Use popart on V function')

    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Traing Configuration
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=1e6)
    parser.add_argument('--num_iters', help='number of iters per episode', type=int, default=0)





    return parser.parse_args()


def get_task_name(args):
    task_name = args.env_id.split("-")[0]
    return task_name


def modify_args(args):


    return args




def main(args):
    set_global_seeds(args.seed)
    env = gym.make(args.env_id)
    env.seed(args.seed)



    gym.logger.setLevel(logging.WARN)

    args = modify_args(args)
    log_dir, policy_model_dir = get_dirs(args)
    print("log_dir: ", log_dir)
    print("model_dir: ", policy_model_dir)


    # def policy_fn(name, ob_space, ac_space,):
    #     return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
    #                                 hid_size=args.policy_hidden_size, num_hid_layers=2, popart=args.popart, gaussian_fixed_var=args.fixed_var)

    if args.task == 'train':


        task_name = get_task_name(args)
        logger.configure(dir=log_dir, log_suffix='_'+task_name, format_strs=["log", "stdout", "tensorboard", "csv"])


        train(env=env,
              seed=args.seed,
              policy_entcoeff=args.policy_entcoeff,
              num_timesteps=args.num_timesteps,
              num_iters=args.num_iters,
              # save_dir,
              checkpoint_dir=policy_model_dir,
              gamma=args.gamma,
              task_name=task_name
              )
    elif args.task == 'evaluate':
        runner(env,
               args.load_model_path,
               timesteps_per_batch=1024,
               number_trajs=10,
               stochastic_policy=args.stochastic_policy,
               save=args.save_sample
               )
    else:
        raise NotImplementedError
    env.close()


def train(env, seed, policy_entcoeff, num_timesteps, num_iters,
          checkpoint_dir,gamma, task_name=None):


    from baselines.trpo_mpi import trpo_mpi
    # Set up for MPI seed
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env.seed(workerseed)
    trpo_mpi.learn(network=args.network, env=env,
                   total_timesteps=num_timesteps,
                   ent_coef=policy_entcoeff,
                   max_iters = num_iters,
                   ckpt_dir=checkpoint_dir,
                   timesteps_per_batch=args.batchsize,
                   max_kl=args.max_kl, cg_iters=args.cg_iters, cg_damping=args.cg_damping,
                   gamma=gamma, lam=0.97,
                   vf_iters=args.vf_iters, vf_stepsize=args.vf_stepsize,
                   task_name=task_name,
                   num_layers=args.policy_hidden_layer,
                   num_hidden=args.policy_hidden_size)


def runner(env, policy_func, load_model_path, timesteps_per_batch, number_trajs,
           stochastic_policy, save=False):

    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)
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


# Sample one trajectory (until trajectory end)
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
