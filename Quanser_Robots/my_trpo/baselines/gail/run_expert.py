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

from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines import bench
from baselines import logger
from baselines.rnd_gail.merged_critic import make_critic

from baselines.rnd_gail.mujoco_main import get_exp_data2, modify_args, get_task_name
from baselines.ebil.utils import get_exp_data, get_dirs
from baselines.rnd_gail.mujoco_main import Log_dir, GAIL_Log_dir, GMMIL_Log_dir, Model_dir, GAIL_Model_dir, GMMIL_Model_dir



def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v2')
    parser.add_argument('--pi', help="model file", type=str, default='2')
    parser.add_argument('--render', help='Save to video', default=0, type=int)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--model_dir', help='the directory to save model', default=Model_dir)
    parser.add_argument('--num_trajs', help='number of exp data', type=int, default=4)
    parser.add_argument('--sample_trajs', help='number of exp data', type=int, default=50)
    parser.add_argument('--log_dir', help='the directory to save log file', default=Log_dir)
    parser.add_argument('--reward', help='Reward Type', type=int, default=0)
    parser.add_argument('--gamma', help='Discount factor', type=float, default=0.97)
    parser.add_argument('--max_kl', type=float, default=0.01)
    boolean_flag(parser, 'pretrained', default=False, help='Use BC to pretrain')
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=3)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1) 
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--adversary_hidden_size', type=int, default=100)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Traing Configuration
    boolean_flag(parser, 'popart', default=True, help='Use popart on V function')
    parser.add_argument('--UsePretrained', help='Use BC as pretrained or not', type=int, default=0)
    parser.add_argument('--indice', help='Which model you would like to check', type=str, default='best')
    return parser.parse_args()

def main(args):
    set_global_seeds(args.seed)
    env = gym.make(args.env_id)
    if args.render:
        vid_dir = osp.expanduser("~/Videos")

        env.env._get_viewer("rgb_array")
        env.env.viewer_setup()

        env = gym.wrappers.Monitor(env, vid_dir, video_callable=lambda ep: True, force=True, mode="evaluation")

    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    if "gail" in args.pi:
        # from baselines.gail import mlp_policy
        # def policy_fn(name, ob_space, ac_space, reuse=False):
        #     return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        #                                 reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)

        from baselines.rnd_gail import mlp_policy
        def policy_fn(name, ob_space, ac_space,):
            return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                        hid_size=args.policy_hidden_size, num_hid_layers=2, popart=args.popart)
    else:
        from baselines.rnd_gail import mlp_policy
        def policy_fn(name, ob_space, ac_space,):
            return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                        hid_size=args.policy_hidden_size, num_hid_layers=2, popart=args.popart)

    args, rnd_iter, dyn_norm = modify_args(args)
    task_name = get_task_name(args)
    log_dir, _, policy_model_dir, __, _ = get_dirs(args)    
    print("policy_model_dir: ",policy_model_dir)
    #exit(0)

    logger.configure(dir=log_dir, log_suffix="_eval" +task_name+'_'+ args.pi, format_strs=["log", "stdout"])

    load_path = osp.join(policy_model_dir, task_name+'_'+args.indice)

    #load_path = osp.join(policy_model_dir, task_name+'_'+ args.pi)
    

    runner(env,
           policy_fn,
           load_path,
           timesteps_per_batch=1000,
           number_trajs=args.sample_trajs,
           stochastic_policy=args.stochastic_policy,
           )
    env.close()


def runner(env, policy_func, load_model_path, timesteps_per_batch, number_trajs,
           stochastic_policy):

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
    avg_len = np.mean(len_list)
    avg_ret = np.mean(ret_list)
    print(ret_list)
    logger.info(avg_len)
    logger.info(avg_ret)
    logger.info(np.std(ret_list))
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
