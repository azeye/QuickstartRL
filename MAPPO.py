#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from config import get_config
from mpe.MPE_env import MPEEnv
from mpe_runner import MPERunner as Runner

"""Train script for MPEs."""

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    # env init
    env = MPEEnv(all_args)
    eval_envs = None
    num_agents = all_args.num_agents
    env.close()

    # run experiments
    config = {
        "all_args": all_args,
        "envs": envs,
        "num_agents": num_agents,

    }
    runner = Runner(config)
    runner.run()




    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
