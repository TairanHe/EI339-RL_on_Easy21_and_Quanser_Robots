# see how the trpo works for the environments in terms of the performance:
# max_timesteps 2e6

#default set:
#seed-X/mlp/policy_hidden_layer-3/policy_hidden_size-128/gamma-0.99/vf_stepsize-0.0003/policy_entcoeff-0/vf_iters-3/max_kl-0.001/cg_iters-10/cg_damping-0.01
#/lam-1.0


for seed in 0 520 250; do
  for vf_stepsize in 0.0001 0.01 0.05; do
      CUDA_VISIBLE_DEVICES=0 python ../baselines/quanser_main/quanser_main.py --seed=${seed} --vf_stepsize=${vf_stepsize} --env_id=$1
  done
done

