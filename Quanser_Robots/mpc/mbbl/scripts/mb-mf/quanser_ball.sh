CUDA_VISIBLE_DEVICES=0 python ../../main/mbmf_main.py --output_dir ../../log --num_planning_traj 10 --planning_depth 10 --random_timesteps 10000 --timesteps_per_batch 3000 --dynamics_epochs 30 --num_workers 1 --mb_timesteps 7000 --dagger_epoch 300 --dagger_timesteps_per_iter 1750 --max_timesteps 1000000 --seed 0 --dynamics_batch_size 500 --task quanser_ball


