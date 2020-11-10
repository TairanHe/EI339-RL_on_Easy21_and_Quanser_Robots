import numpy as np
import pickle
import tensorflow as tf
import os,sys,inspect
import os.path as osp
def get_dirs(args):
    log_dir, policy_model_dir = None, None
    if 'log_dir' in args.__dict__:
        log_dir = args.log_dir
        log_dir = osp.join(log_dir, args.env_id)
        log_dir = osp.join(log_dir, 'seed-' + str(args.seed))
        log_dir = osp.join(log_dir, args.network)
        log_dir = osp.join(log_dir, "policy_hidden_layer-" + str(args.policy_hidden_layer))
        log_dir = osp.join(log_dir, "policy_hidden_size-" + str(args.policy_hidden_size))
        log_dir = osp.join(log_dir, "batchsize-" + str(args.batchsize))
        log_dir = osp.join(log_dir, "gamma-"+str(args.gamma))
        log_dir = osp.join(log_dir, "vf_stepsize-" + str(args.vf_stepsize))
        log_dir = osp.join(log_dir, "policy_entcoeff-"+str(args.policy_entcoeff))
        log_dir = osp.join(log_dir, "vf_iters-" + str(args.vf_iters))
        log_dir = osp.join(log_dir, "max_kl-" + str(args.max_kl))
        log_dir = osp.join(log_dir, "cg_iters-" + str(args.cg_iters))
        log_dir = osp.join(log_dir, "cg_damping-" + str(args.cg_damping))
        log_dir = osp.join(log_dir, "lam-" + str(args.lam))
        print(log_dir)

    if 'model_dir' in args.__dict__:
        model_dir = args.model_dir
        model_dir = osp.join(model_dir, args.env_id)
        model_dir = osp.join(model_dir, 'seed-' + str(args.seed))
        model_dir = osp.join(model_dir, args.network)

        model_dir = osp.join(model_dir, "policy_hidden_layer-" + str(args.policy_hidden_layer))
        model_dir = osp.join(model_dir, "policy_hidden_size-" + str(args.policy_hidden_size))
        model_dir = osp.join(model_dir, "batchsize-" + str(args.batchsize))
        model_dir = osp.join(model_dir, "gamma-"+str(args.gamma))
        model_dir = osp.join(model_dir, "vf_stepsize-" + str(args.vf_stepsize))
        model_dir = osp.join(model_dir, "policy_entcoeff-"+str(args.policy_entcoeff))
        model_dir = osp.join(model_dir, "vf_iters-" + str(args.vf_iters))
        model_dir = osp.join(model_dir, "max_kl-" + str(args.max_kl))
        model_dir = osp.join(model_dir, "cg_iters-" + str(args.cg_iters))
        model_dir = osp.join(model_dir, "cg_damping-" + str(args.cg_damping))
        model_dir = osp.join(model_dir, "lam-" + str(args.lam))
        policy_model_dir = model_dir
    else:
        if 'policy_model_dir' in args.__dict__:
            policy_model_dir = args.policy_model_dir
            policy_model_dir = osp.join(policy_model_dir, args.env_id)
            policy_model_dir = osp.join(policy_model_dir, 'seed-' + str(args.seed))
            policy_model_dir = osp.join(policy_model_dir, args.network)
            policy_model_dir = osp.join(policy_model_dir, "policy_hidden_layer-" + str(args.policy_hidden_layer))
            policy_model_dir = osp.join(policy_model_dir, "policy_hidden_size-" + str(args.policy_hidden_size))
            policy_model_dir = osp.join(policy_model_dir, "batchsize-" + str(args.batchsize))
            policy_model_dir = osp.join(policy_model_dir, "gamma-" + str(args.gamma))
            policy_model_dir = osp.join(policy_model_dir, "vf_stepsize-" + str(args.vf_stepsize))
            policy_model_dir = osp.join(policy_model_dir, "policy_entcoeff-" + str(args.policy_entcoeff))
            policy_model_dir = osp.join(policy_model_dir, "vf_iters-" + str(args.vf_iters))
            policy_model_dir = osp.join(policy_model_dir, "max_kl-" + str(args.max_kl))
            policy_model_dir = osp.join(policy_model_dir, "cg_iters-" + str(args.cg_iters))
            policy_model_dir = osp.join(policy_model_dir, "cg_damping-" + str(args.cg_damping))
            policy_model_dir = osp.join(policy_model_dir, "lam-" + str(args.lam))



    return log_dir,policy_model_dir