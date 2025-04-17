# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pathlib import Path
BASEDIR = Path(__file__).parent.parent.parent
import sys
if str(BASEDIR) not in sys.path:
    sys.path.append(str(BASEDIR))

import isaacgym
from isaacgym import gymapi
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from simulation.utils.reformat import omegaconf_to_dict, print_dict
from simulation.utils.misc import set_np_formatting, set_seed
from simulation.im2gym.tasks.domain import Domain
from simulation.im2gym.tasks import immgym_task_map
from simulation.im2gym.algos.policy import Model
from simulation.im2gym.algos.ppo import Runner, PPO

## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)

# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
# num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)

WANDB = True
if WANDB:
    import wandb, datetime

@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)  ############ question -> seed = 42 -> deterministic??

    print("-------------------")
    rlg_config_dict = omegaconf_to_dict(cfg.task)
    # dump config dict
    # experiment_dir = os.path.join('runs', cfg.train.params.config.name)
    # os.makedirs(experiment_dir, exist_ok=True)
    # with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
    #     f.write(OmegaConf.to_yaml(cfg))

    env: Domain = immgym_task_map[cfg.task_name](rlg_config_dict, cfg["sim_device"], cfg["graphics_device_id"], cfg["headless"], cfg["use_states"], record=cfg['record'])
    env_cfg = {}
    env_cfg["num_obs"] = env.num_obs # 82
    env_cfg["num_states"] = env.num_states # 115
    env_cfg["num_student_obs"] = env.num_student_obs
    env_cfg["num_acts"] = env.num_acts
    model = Model(cfg, env_cfg)
    algo = PPO(cfg, model)
    log_name=''
    log_name+=cfg["train"]["method"]
    log_name+=f'_adpative_{cfg["task"]["env"]["adaptive_residual_scale"]["activate"]}'
    if cfg["task"]["env"]["adaptive_residual_scale"]["activate"]:
        log_name+=f'_from_{cfg["task"]["env"]["initial_residual_scale"]}'
        log_name+=f'_to_{cfg["task"]["env"]["adaptive_residual_scale"]["minimum"]}'
    else:
        log_name+=f'_with_{cfg["task"]["env"]["initial_residual_scale"]}'
    if cfg.task_name =="Bump" and cfg["task"]["env"]["geometry"]["both_side"]:
        log_name+="_both_side"

    if WANDB and not cfg.test and not cfg.play:
        wandb.init(project='hardware-CRM', 
                   entity='im2-hardware',
                   config=omegaconf_to_dict(cfg),
                   name=f'{cfg.task_name}_{datetime.datetime.now()}',
                   sync_tensorboard=True)
    runner = Runner(cfg, env, algo, model)

    #################################### cfg.play??

    if len(cfg.checkpoint) > 0:
        runner.load(cfg.checkpoint)
        
    if cfg.play:
        runner.play()
    elif cfg.test:
        if len(cfg.checkpoint) == 0:
            raise RuntimeError("play mode requires pre-trained model")
        print("_______________________testing______________________")
        runner.test()  
    else:
        runner.train()

        if cfg["phase2"]:
            gym = env.destroy_sim()
            del env
            print("-------------------------We are in phase2 now!------------------------")
            cfg["task"]["env"]["adaptive_residual_scale"]["activate"]=False
            cfg["task"]["env"]["initial_residual_scale"]=[0.02,0.03]
            cfg["task"]["task"]["env_randomize"]=True
            cfg["task"]["task"]["torque_randomize"]=True
            cfg["task"]["task"]["observation_randomize"]=True
            cfg["task"]["env"]["adaptive_dof_pos_limit"]["activate"]=True
            rlg_config_dict = omegaconf_to_dict(cfg.task)
            newenv = immgym_task_map[cfg.task_name](rlg_config_dict, cfg["sim_device"], cfg["graphics_device_id"], cfg["headless"], cfg["use_states"], gym=gym)
            # trun on DR + adaptive dof_pos_limit set initial residual, turn off adpative residual 
            previous_writer = runner.writer
            previous_frame = runner.frames
            newrunner = Runner(cfg, newenv, algo, model, writer=previous_writer, startframe=previous_frame)
            del runner
            newrunner.train()


if __name__ == "__main__":
    launch_rlg_hydra()
