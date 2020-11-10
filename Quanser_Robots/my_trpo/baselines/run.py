import sys
import multiprocessing
import os.path as osp
import os
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np
import pickle
from tqdm import tqdm

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            ret = True
            if args.play:# or args.sample:
               ret = False
            env = VecNormalize(env, ret=ret)

    return env


def get_env_type(args):
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}



def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args) 
    
    extra_args['load_path'] = args.load_path
    if args.load:
        load_path = osp.join(args.load_path, args.env)
        load_path = osp.join(load_path, 'seed-'+str(args.seed))
        load_path = osp.join(load_path, 'expert_final')
        extra_args['load_path'] = str(load_path)
    
    log_path = args.log_path
    if args.log_path is not None:
        log_path = osp.join(args.log_path, args.env)
        log_path = osp.join(log_path, 'seed-'+str(args.seed))
     
    if args.extra_import is not None:
        import_module(args.extra_import)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure(log_path)
    else:
        logger.configure(log_path, format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    model, env = train(args, extra_args)

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        save_path = osp.join(save_path, str(args.env))
        save_path = osp.join(save_path, 'seed-'+str(args.seed))
        save_path = osp.join(save_path, 'expert_final')
        logger.log('Saving the final model to {}'.format(save_path))
        model.save(save_path)

    if args.play:
        logger.log("Running trained model")
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
        play_times = 0
        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs,S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rew, done, _ = env.step(actions)
            episode_rew += rew
            # env.render()
            done = done.any() if isinstance(done, np.ndarray) else done
            if done:
                for i in np.nonzero(done)[0]:
                    print('episode_rew=%f' % episode_rew[i])
                    episode_rew[i] = 0
                obs = env.reset()
                play_times+=1
                if play_times > 10:
                    break

    if args.sample:
        logger.log("Sampling trajectries with trained model")
        obs_list = []
        acs_list = []
        len_list = []
        ret_list = []
        for _ in tqdm(range(args.num_trajs)):
            traj = traj_1_generator(model, env, args.timesteps)
            obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
            obs_list.append(obs)
            acs_list.append(acs)
            len_list.append(ep_len)
            ret_list.append(ep_ret)
        avg_len = np.mean(len_list)
        avg_ret = np.mean(ret_list)
        logger.info(avg_len)
        logger.info(avg_ret)
        logger.info(np.std(ret_list))
        
        if args.save_sample_data:
            filename = osp.join(args.expert_data_path, args.env)
            filename = osp.join(filename, 'seed-'+str(args.seed))
            if not os.path.exists(filename):
                os.makedirs(filename)
            filename = osp.join(filename, 'expert_sample')
            data = {}
            data["acts"] = np.array(acs_list)
            data["obs"] = np.array(obs_list)
            data["lens"] = np.array(len_list)
            data["rets"] = np.array(ret_list)
            print("saving to {}".format(filename))
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
        avg_len = sum(len_list)/len(len_list)
        avg_ret = sum(ret_list)/len(ret_list)

        print("Average length:", avg_len)
        print("Average return:", avg_ret)
        print("std:", np.std(ret_list))

    env.close()

    return model

def traj_1_generator(pi, env, horizon=None):
    if horizon is None:
        horizon = 99999
    t = 0    
    ob = env.reset()

    state = pi.initial_state if hasattr(pi, 'initial_state') else None
        
    cur_ep_len = 0  # len of current episode
    cur_ep_ret = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1) # return in current episode
    ret = 0
    obs, rews, acs, dones = [], [], [], []
    while True:
        if state is not None:
            ac, _, state, _ = pi.step(ob,S=state, M=dones)
        else:
            ac, _, _, _ = pi.step(ob)
        
        obs.append(ob[0])
        acs.append(ac[0])

        ob, rew, done, _ = env.step(ac)
        rews.append(rew)
        dones.append(done)
        cur_ep_ret += rew
        cur_ep_len += 1
        done_any = done.any() if isinstance(done, np.ndarray) else done
        if done_any or t>=horizon:
            for i in np.nonzero(done)[0]:
                cur_ep_ret[i] = _[i]['episode']['r'] 
                print('episode_rew=%f' % cur_ep_ret[i])
                ret = cur_ep_ret[i]
                break
            # print('episode rew = {}'.format(cur_ep_ret))
            break
        t+=1
    traj = {"ob": obs, "rew": rews, "done": dones, "ac": acs,
            "ep_ret": ret, "ep_len": cur_ep_len}
    return traj

if __name__ == '__main__':
    main(sys.argv)
