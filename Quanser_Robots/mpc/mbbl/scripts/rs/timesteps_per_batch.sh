#default
#      CUDA_VISIBLE_DEVICES=0 python ../../main/rs_main.py --output_dir ../../log --num_planning_traj 10 --planning_depth 10 --random_timesteps 10000 --timesteps_per_batch 3000 --num_workers 1 --max_timesteps 200000 --seed 0 --task $1

for timesteps_per_batch in 1500 6000; do
      CUDA_VISIBLE_DEVICES=0 python ../../main/rs_main.py --output_dir ../../log --num_planning_traj 10 --planning_depth 10 --random_timesteps 10000 --timesteps_per_batch ${timesteps_per_batch} --num_workers 1 --max_timesteps 200000 --seed 0 --task $1
done
