# -----------------------------------------------------------------------------
#   @brief: main fucntion
# -----------------------------------------------------------------------------

from mbbl.config import base_config
from mbbl.config import init_path
from mbbl.config import rs_config
from mbbl.util.base_main import train
from mbbl.util.common import logger


def main():
    parser = base_config.get_base_config()
    parser = rs_config.get_rs_config(parser)
    args = base_config.make_parser(parser)

    if args.write_log:
        if args.output_dir == None:
            args.output_dir = "log"
        else:
            pass
        log_path = str(args.output_dir)+'/rs-' + str(args.task) + '/seed-' + str(args.seed) +'/num_planning_traj-' + str(args.num_planning_traj) +'/plannging depth-' + str(args.planning_depth) +'/timesteps_per_batch-' + str(args.timesteps_per_batch) +'/random_timesteps-' + str(args.random_timesteps) +'/max_timesteps-' + str(args.max_timesteps)+'/'
        logger.set_file_handler(path=log_path, prefix='', time_str="0")

    print('Training starts at {}'.format(init_path.get_abs_base_dir()))
    from mbbl.trainer import shooting_trainer
    from mbbl.sampler import singletask_sampler
    from mbbl.worker import rs_worker
    from mbbl.network.policy.random_policy import policy_network

    if args.gt_dynamics:
        from mbbl.network.dynamics.groundtruth_forward_dynamics import \
            dynamics_network
    else:
        from mbbl.network.dynamics.deterministic_forward_dynamics import \
            dynamics_network

    if args.gt_reward:
        from mbbl.network.reward.groundtruth_reward import reward_network
    else:
        from mbbl.network.reward.deterministic_reward import reward_network

    train(shooting_trainer, singletask_sampler, rs_worker,
          dynamics_network, policy_network, reward_network, args)


if __name__ == '__main__':
    main()
