#default
#      CUDA_VISIBLE_DEVICES=0 python ../../main/rs_main.py --output_dir ../../log --num_planning_traj 10 --planning_depth 10 --random_timesteps 10000 --timesteps_per_batch 3000 --num_workers 1 --max_timesteps 200000 --seed 0 --task $1

for num_planing_traj in 20; do
      CUDA_VISIBLE_DEVICES=0 python ../../main/rs_main.py --output_dir ../../log --num_planning_traj ${num_planing_traj} --planning_depth 10 --random_timesteps 10000 --timesteps_per_batch 3000 --num_workers 1 --max_timesteps 200000 --seed 0 --task $1
done
