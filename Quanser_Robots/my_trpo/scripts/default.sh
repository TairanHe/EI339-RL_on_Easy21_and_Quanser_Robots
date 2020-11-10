# see how the ppo works for the environments in terms of the performance:
# batch size
# max_timesteps 1e8


for seed in 0 520 250; do
      CUDA_VISIBLE_DEVICES=0 python ../baselines/quanser_main/quanser_main.py --seed=${seed} --env_id=$1
done

