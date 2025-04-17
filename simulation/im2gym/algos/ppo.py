from cmath import inf
from typing import Tuple, Dict, Optional
from models.policy import *

# from simulation.im2gym.tasks.domain import Domain
from simulation.im2gym.tasks.sysid import Sysid
from simulation.im2gym.algos.utils import *
from simulation.im2gym.algos.schedulers import AdaptiveScheduler, policy_kl, LinearScheduler
from simulation.utils.generator import generatorBuilder
from simulation.im2gym.algos.policy import Model
import torch

from torch.utils.tensorboard import SummaryWriter

from isaacgym.torch_utils import normalize
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import shutil
import os
import random

from simulation.im2gym.algos.set_initial_position import setFixedInitPosition, PreCalculatedIK, PreCalculatedRandom

SYSID_SAVE_DIR = 'sysID_progress/'
np.set_printoptions(precision=5, suppress=True)
torch.set_printoptions(precision=5, sci_mode=False, linewidth=200)

class PPO:
    """
    PPO algorithm for post contact policy (currently)
    """
    def __init__(self, cfg, model: Model):
        """
        cfg: configuration dictinary
        model: model of policy
        """
        config = cfg["train"]["params"]["config"]

        #PPO parameters
        self.lr = config["learning_rate"]
        self.gamma = config["gamma"]
        self.cliprange = config["e_clip"]
        self.ent_coef = config["entropy_coef"]
        self.vf_coef = config["critic_coef"]
        self.bounds_loss_coef = config["bounds_loss_coef"]
        self.truncate_grads = config["truncate_grads"]
        self.grad_norm = config["grad_norm"]
        self.asymmetric_obs=cfg["use_states"]

        self.model = model
        self.optimizer = torch.optim.Adam(self.model.policy.parameters(), lr=self.lr)

        if self.asymmetric_obs:
            self.value_optimizer = torch.optim.Adam(self.model.value_network.parameters(), lr=self.lr)

        self.dist = DiagGaussianPd()

    def train(self, observations, actions, returns, values, advantages, neglogpac_old, mu_old, logstd_old, state=None):
        """
        observations: observations from rollout.
        actions: actions made during rollout.
        returns: return get during rollout.
        values: value estimation for state in rollout.
        advantages: estimated advantages using GAE for rollout. 
        neglogpac_old, mu_old, logstd_old: estimated values by policy before current step update.
        """
        if self.asymmetric_obs:
            mu, logstd = self.model.policy(observations)
            vpred = self.model.value_network(state)
        else:
            mu, logstd, vpred = self.model.policy(observations)
        neglogpac = self.dist.neglogp(mu, logstd, actions)
        entropy = torch.mean(self.dist.entropy())

        vpredclipped = values + torch.clip(vpred - values, -self.cliprange, self.cliprange)
        vf_losses1 = torch.square(vpred - returns)
        vf_losses2 = torch.square(vpredclipped - returns)
        vf_loss = 0.5 * torch.mean(torch.maximum(vf_losses1, vf_losses2))

        # vf_loss = 0.5 * torch.mean(torch.square(vpred - returns))

        ratio = torch.exp(neglogpac_old - neglogpac)
        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clip(ratio, 1 - self.cliprange, 1 + self.cliprange)
        pg_loss = torch.mean(torch.maximum(pg_losses1, pg_losses2))

        mu_loss_high = torch.square(torch.clamp_min(mu - 1.1, 0.0))
        mu_loss_low = torch.square(torch.clamp_max(mu + 1.1, 0.0))
        b_loss = torch.mean((mu_loss_low + mu_loss_high).sum(dim=-1))

        loss = pg_loss - entropy * self.ent_coef  + b_loss * self.bounds_loss_coef
        vf_loss = vf_loss * self.vf_coef
        if not self.asymmetric_obs:
            loss += vf_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        if self.truncate_grads:
            torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), self.grad_norm)
        self.optimizer.step()

        if self.asymmetric_obs:
            self.value_optimizer.zero_grad()
            vf_loss.backward()
            if self.truncate_grads:
                torch.nn.utils.clip_grad_norm_(self.model.value_network.parameters(), self.grad_norm)
            self.value_optimizer.step()

        with torch.no_grad():
            kl_dist = policy_kl(mu.detach(), torch.exp(logstd.detach()), mu_old, torch.exp(logstd_old))

        return kl_dist, mu.detach(), logstd.detach()



class Runner:
    """
    train = training
    test = test with pre-sampled test set
    play = execute policy with online sampled data. (not implemented yet)
    """
    def __init__(self, cfg: Dict, env, algo: PPO, model: Model,
                 writer: Optional[SummaryWriter] = None, startframe :Optional[int] = None):
        self.env = env
        self.algo = algo
        self.model = model

        env_list = ["Card", "HiddenCard", "Flip_chamfer", "Hole", "Reorientation", "Bookshelf", "Bump", "Hole_wide", "Throw", "Throw_left"]
        env_dict = dict(zip(env_list, range(len(env_list))))

        # parameters for training
        self.cfg = cfg
        self.rl_device = cfg["rl_device"]
        self.seed = self.cfg["train"]["params"]["seed"]
        config = self.cfg["train"]["params"]["config"]
        self.joint_training = config["joint"]["activate"]
        self.nminiepochs = config["mini_epochs"]
        self.nactors = config["num_actors"]
        self.nsteps = config["horizon_length"]
        self.nminibatches = config["minibatch_size"]
        self.max_epochs = config["max_epochs"]
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.normalize_advantage = config["normalize_advantage"]
        self.normalize_input = config["normalize_input"]
        self.normalize_value = config["normalize_value"]
        self.kl_threshold = config["kl_threshold"]
        self.lr = config["learning_rate"]
        self.value_lr = self.lr
        self.reward_shaper = config["reward_shaper"]["scale_value"]
        self.asymmetric_obs = cfg["use_states"]

        self.initial_dof_vel_limit = cfg["task"]["env"]["initial_dof_vel_limit"]
        self.adaptive_dof_vel_limit = cfg["task"]["env"]["adaptive_dof_vel_limit"]["activate"]
        if self.adaptive_dof_vel_limit:
            self.dof_vel_limit_threshold = cfg["task"]["env"]["adaptive_dof_vel_limit"]["threshold_success_rate"]
            self.dof_vel_limit_bucket = cfg["task"]["env"]["adaptive_dof_vel_limit"]["bucket"]
            self.dof_vel_limit_maximum = cfg["task"]["env"]["adaptive_dof_vel_limit"]["maximum"]
            self.dof_vel_limit_offset = (self.dof_vel_limit_maximum - self.initial_dof_vel_limit) / self.dof_vel_limit_bucket
            self.dof_vel_limit_update_count = 0
        
        self.initial_dof_pos_limit = cfg["task"]["env"]["initial_dof_pos_limit"]
        self.adaptive_dof_pos_limit = cfg["task"]["env"]["adaptive_dof_pos_limit"]["activate"]
        if self.adaptive_dof_pos_limit:
            self.dof_pos_limit_threshold = cfg["task"]["env"]["adaptive_dof_pos_limit"]["threshold_success_rate"]
            self.dof_pos_limit_bucket = cfg["task"]["env"]["adaptive_dof_pos_limit"]["bucket"]
            self.dof_pos_limit_maximum = cfg["task"]["env"]["adaptive_dof_pos_limit"]["maximum"]
            self.dof_pos_limit_offset = (self.dof_pos_limit_maximum - self.initial_dof_pos_limit) / self.dof_pos_limit_bucket
            self.dof_pos_limit_update_count = 0
       
        self.adaptive_residual_scale = cfg["task"]["env"]["adaptive_residual_scale"]["activate"]
        if self.adaptive_residual_scale:
            self.initial_residual_scale = cfg["task"]["env"]["initial_residual_scale"]
            self.residual_scale_bucket = cfg["task"]["env"]["adaptive_residual_scale"]["bucket"]
            self.threshold_success_rate = cfg["task"]["env"]["adaptive_residual_scale"]["threshold_success_rate"]
            self.minimum_residual_scale = cfg["task"]["env"]["adaptive_residual_scale"]["minimum"]

            def calculate_ratio(min_residual_scale, initial_residual_scale, residual_scale_bucket, indices):
                return np.array([np.power((min_residual_scale[i] / initial_residual_scale[i]), 1./residual_scale_bucket) for i in indices])

            # Assuming self.minimum_residual_scale, self.initial_residual_scale, and self.residual_scale_bucket are already defined

            self.position_ratio = calculate_ratio(self.minimum_residual_scale, self.initial_residual_scale, self.residual_scale_bucket, range(6))
            self.kp_ratio = calculate_ratio(self.minimum_residual_scale, self.initial_residual_scale, self.residual_scale_bucket, range(6, 12))
            self.kd_ratio = calculate_ratio(self.minimum_residual_scale, self.initial_residual_scale, self.residual_scale_bucket, range(12, 18))

            self.residual_update_count = 0

        # parameters for logs
        self.rewards_mean = AverageMeter(1, 100).to(self.rl_device)
        self.each_reward_mean: Dict[str, AverageMeter] = dict()
        reward_terms = self.cfg['train']['rewards']['names'] \
                            if 'rewards' in self.cfg['train'] \
                            else ['object', 'inductive', 'energy']
        for reward_name in reward_terms:
            self.each_reward_mean[reward_name] = AverageMeter(1, 100).to(self.rl_device)
        self.success_mean = AverageMeter(1, 100).to(self.rl_device)
        self.reg_mean = AverageMeter(1, 200).to(self.rl_device)
        self.last_mean_reward = -inf
        self.config_name = config["name"]
        self.save_freq = config["save_frequency"]
        self.save_best_after = config["save_best_after"]
        exp_name = datetime.now().strftime("%y-%m-%d-%H-%M") + f'-seed-{self.seed}'
        
        if not cfg["test"]:
            self.save_dir = Path(__file__).parents[3].joinpath("trains", config["name"], self.cfg["train"]["method"], exp_name)
            # if self.save_dir.exists(): shutil.rmtree(self.save_dir) # Remove existing contents in the directory
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.writer = writer if writer else SummaryWriter(os.path.join(self.save_dir, 'events'), max_queue=1000, flush_secs=600) 
        else:
            self.writer = None

        # assign seeds and buffer
        self.nbatch = self.nactors * self.nsteps
        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
        self.reset = torch.zeros(self.nactors, device=self.rl_device)
        self.scheduler = AdaptiveScheduler(self.kl_threshold)
        self.value_scheduler = LinearScheduler(self.lr, max_steps=10000)
        self.reward_buf = torch.zeros(self.nactors, 1, dtype=torch.float, device=self.rl_device)

        # exponentiation of gamma i.e. gamma^t
        self.gamma_exp = torch.ones(self.nactors, 1, dtype=torch.float, device=self.rl_device)
             
        self.batchsize = self.nsteps * self.nminibatches
        self.frames = startframe if startframe else 0

        self.IK_query_size = 22000
        self.generator = generatorBuilder(config["name"], writer=self.writer,
            map=None, IK_query_size=self.IK_query_size, device=self.rl_device, geometry=self.cfg["task"]["env"]["geometry"])

        self.robot_init = self.cfg["task"]["env"]["reset_distribution"]["robot_initial_state"]["type"]

        # self.initial_robot_position = [0.4, 0., 0., 0.05, -0.3, 0.]
        self.initial_robot_position = [0.217, -0.312, -0.057, 0.152, 1.431, -0.643]

        self.hand = self.cfg["task"]["env"]["hand"]
        if self.hand == "both":
            self.initial_robot_position = self.env.get_initial_joint_position(1).squeeze()

        if self.robot_init == "pre_calculated_ik":
            self.sP = PreCalculatedIK()
        
        elif self.robot_init == "pre_calculated_random":
            self.sP = PreCalculatedRandom()
        
        elif self.robot_init == "fixed_init_position":
            self.sP = setFixedInitPosition(self.initial_robot_position)
        
        

    def save(self, dir):
        state = {}
        state['post'] = self.model.policy.state_dict()
        state['post_optimizer'] = self.algo.optimizer.state_dict()
        if self.asymmetric_obs:
            state['post_value'] = self.model.value_network.state_dict()
            state['post_value_optimizer'] = self.algo.value_optimizer.state_dict()

        state['last_mean_rewards'] = self.last_mean_reward
        state['frames'] = self.frames
        save_checkpoint(dir, state)

    def load(self, checkpoint):
        state = load_checkpoint(checkpoint)

        post_state = state.pop('post')
        post_optim_state = state.pop('post_optimizer')
        if self.asymmetric_obs:
            post_value_state = state.pop('post_value')
            post_value_optim_state = state.pop('post_value_optimizer')
            self.model.value_network.load_state_dict(post_value_state)
            self.algo.value_optimizer.load_state_dict(post_value_optim_state)
        self.last_mean_reward = state.pop('last_mean_rewards')
        self.frames = state.pop('frames')
        self.model.policy.load_state_dict(post_state)
        self.algo.optimizer.load_state_dict(post_optim_state)

        print(f"restored running mean for value(count, mean, std):\
         {self.model.policy.value_mean_std.count} {self.model.policy.value_mean_std.running_mean} {self.model.policy.value_mean_std.running_var}")
        print(f"restored running mean for obs(count, mean, std):\
         {self.model.policy.running_mean_std.count} {self.model.policy.running_mean_std.running_mean} {self.model.policy.running_mean_std.running_var}")

    def train(self):

        reset_count = 0
        self._initialize_robot_joint_position()
        nupdates = self.max_epochs
        obs_dict = self.env.reset()
        self.observation = torch.clone(obs_dict["obs"])
        if self.asymmetric_obs:
            self.state = torch.clone(obs_dict["states"])
            self.regs = torch.zeros(self.nactors, dtype=torch.float, device=self.rl_device)
        else: 
            self.state = None
        last_update = self.frames
        for update in range(nupdates):
            observations, actions, values, neglogpacs, rewards, resets, mus, logstds = [], [], [] ,[], [], [], [], []
            if self.asymmetric_obs:
                states = []
            self.algo.model.policy.eval()
            if self.asymmetric_obs:
                self.algo.model.value_network.eval()
            for step in range(self.nsteps):
                action, value, neglogp, mu, logstd = self.model.step(self.observation, self.state)
                observations.append(self.observation) # observation:(env, obs)
                resets.append(self.reset) # reset: (env,)
                actions.append(action) # action: (env, action)
                values.append(value) # value: (env,)
                neglogpacs.append(neglogp) # neglogp: (env,)
                mus.append(mu) # mu: (env, action)
                logstds.append(logstd) # logstd: (env, action)
                if self.asymmetric_obs:
                    states.append(self.state)
                obs_dict, reward, reset, others = self.env.step(action)

                self.observation = torch.clone(obs_dict["obs"])
               
                self.reward_buf *=self.gamma
                self.reward_buf += (reward.unsqueeze(1)) #R_t=r_(t)+gamma*r_(t+1)+gamma^2*r_(t+2)+... 
                reward = torch.clone(reward) * self.reward_shaper # reward scaling
                rewards.append(reward) # reward: (env,)
                self.reset = torch.clone(reset)
                env_reset_indices = self.reset.view(self.nactors, 1).all(dim=1).nonzero(as_tuple=False)
                self.rewards_mean.update(self.reward_buf[env_reset_indices])
                if 'rewards' not in self.cfg['train']:
                    self.each_reward_mean['object'].update(self.env.object_rewards[env_reset_indices])
                    self.each_reward_mean['inductive'].update(self.env.inductive_rewards[env_reset_indices])
                    self.each_reward_mean['energy'].update(self.env.energy_rewards[env_reset_indices])
                else:
                    for reward_name, reward_meter in self.each_reward_mean.items():
                        reward_meter.update(self.env.reward_terms[reward_name][env_reset_indices])
                self.success_mean.update(self.env.env_succeed[env_reset_indices])
                if self.asymmetric_obs:
                    self.state = torch.clone(obs_dict["states"])
                    # self.reg = torch.clone(others["regularization"])
                    # self.regs += self.reg
                    # self.reg_mean.update((self.regs[env_reset_indices]/self.env.progress_buf[env_reset_indices]).unsqueeze(1))
                    # self.regs[env_reset_indices]=0
                
                reset_count += torch.count_nonzero(self.reset).item()
                if reset_count >= self.nactors:
                    self._initialize_robot_joint_position()
                    reset_count = 0

                on_running = 1.0 - self.reset.float()
                self.reward_buf = self.reward_buf * on_running.unsqueeze(1)

            self.frames += self.batchsize
            self.writer.add_scalar('performance/reward', self.rewards_mean.get_mean()[0], self.frames)
            for reward_name, reward_meter in self.each_reward_mean.items():
                self.writer.add_scalar(f'performance/{reward_name}_reward', reward_meter.get_mean()[0], self.frames)
            # self.writer.add_scalar('performance/object_reward', self.object_rewards_mean.get_mean()[0], self.frames)
            # self.writer.add_scalar('performance/inductive_reward', self.inductive_rewards_mean.get_mean()[0], self.frames)
            # self.writer.add_scalar('performance/energy_reward', self.energy_rewards_mean.get_mean()[0], self.frames)
            self.writer.add_scalar('performance/success rate', self.success_mean.get_mean()[0], self.frames)
            if self.asymmetric_obs:
                self.writer.add_scalar('regularization/gain', self.reg_mean.get_mean()[0], self.frames)
            # if self.success_mean.get_mean()[0]>0.95 and not self.adaptive_dof_pos_limit:
            #     if self.env.position_scale[0] <= self.env.minimum_residual_scale[0]:
            #         checkpoint_name=self.config_name+'_last'+'_rew_'+str(current_mean_reward)
            #         self.save(os.path.join(self.save_dir, checkpoint_name)) 
            #         break
            if self.adaptive_dof_pos_limit:
                self.dof_pos_limit_update_count += 1
                if self.dof_pos_limit_update_count>=100 and self.success_mean.get_mean()[0] > self.dof_pos_limit_threshold:
                    if self.env.add_dof_position_limit_offset(self.dof_pos_limit_offset):
                        self.success_mean.clear()
                        self.dof_pos_limit_update_count = 0
                        self.dof_vel_limit_update_count = 0
                        self.residual_update_count = 0
            
            if self.adaptive_dof_vel_limit:
                self.dof_vel_limit_update_count += 1
                if self.dof_vel_limit_update_count>=100 and self.success_mean.get_mean()[0] > self.dof_vel_limit_threshold:
                    if self.env.add_dof_velocity_limit_offset(self.dof_vel_limit_offset):
                        self.success_mean.clear()
                        self.dof_pos_limit_update_count = 0
                        self.dof_vel_limit_update_count = 0
                        self.residual_update_count = 0

            if self.adaptive_residual_scale:
                self.residual_update_count += 1
                if self.residual_update_count >= 100 and self.success_mean.get_mean()[0] > self.threshold_success_rate:
                    if self.env.update_residual_scale(self.position_ratio, self.kp_ratio, self.kd_ratio):
                        self.success_mean.clear()
                        self.dof_pos_limit_update_count = 0
                        self.dof_vel_limit_update_count = 0
                        self.residual_update_count = 0

            observations = torch.stack(observations, dim=0) # observations: (steps, env, obs)
            actions = torch.stack(actions, dim=0) # actions: (steps, env, action)
            values = torch.stack(values, dim=0) # values: (steps, env)
            neglogpacs = torch.stack(neglogpacs, dim=0) # neglogpacs: (steps, env)
            rewards = torch.stack(rewards, dim=0) # rewards: (steps, env)
            resets = torch.stack(resets, dim=0) # resets: (steps, env)
            mus = torch.stack(mus, dim=0) # mus: (steps, env, action)
            logstds = torch.stack(logstds, dim=0) # logstds: (steps, env, action)
            if self.asymmetric_obs:
                states=torch.stack(states, dim=0)

            last_value = self.model.value(self.observation, self.state) # last_value: (env,)
            advantages = torch.zeros_like(rewards) # advantages: (steps, env)
            lastgaelam = 0
            for t in reversed(range(self.nsteps)):
                if t == self.nsteps - 1:
                    nextnonterminal = 1.0 - self.reset # nonterminal: (env,)
                    nextvalue = last_value # nextvalue: (env,)
                else:
                    nextnonterminal = 1.0 - resets[t + 1]
                    nextvalue = values[t + 1] # values: (steps, env)
                delta = rewards[t] + self.gamma * nextvalue * nextnonterminal - values[t]
                advantages[t] = delta + self.gamma * self.tau * nextnonterminal * lastgaelam
                lastgaelam = advantages[t]
            returns = advantages + values # returns: (steps, env)

            observations = swap_and_flatten(observations)
            actions = swap_and_flatten(actions)
            returns = swap_and_flatten(returns)
            values = swap_and_flatten(values)
            neglogpacs = swap_and_flatten(neglogpacs)
            mus = swap_and_flatten(mus)
            logstds = swap_and_flatten(logstds)
            if self.asymmetric_obs:
                states = swap_and_flatten(states)
            else:
                states = torch.zeros((self.nbatch), device=self.rl_device)

            self.algo.model.policy.train()
            if self.asymmetric_obs:
                self.algo.model.value_network.train()

            advantages = returns - values

            if self.normalize_value:
                if self.asymmetric_obs:
                    self.algo.model.value_network.value_mean_std.train()
                    values = torch.squeeze(self.algo.model.value_network.value_mean_std(torch.unsqueeze(values, dim=-1)))
                    returns = torch.squeeze(self.algo.model.value_network.value_mean_std(torch.unsqueeze(returns, dim=-1)))
                    self.algo.model.value_network.value_mean_std.eval()
                else:
                    self.algo.model.policy.value_mean_std.train()
                    values = torch.squeeze(self.algo.model.policy.value_mean_std(torch.unsqueeze(values, dim=-1)))
                    returns = torch.squeeze(self.algo.model.policy.value_mean_std(torch.unsqueeze(returns, dim=-1)))
                    self.algo.model.policy.value_mean_std.eval()

            if self.normalize_advantage:
                advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-8)
            
            for miniepoch in range(self.nminiepochs):
                # permutation = torch.randperm(self.nbatch)
                permutation = torch.arange(self.nbatch, device=self.cfg["rl_device"])
                for start in range(0, self.nbatch, self.nminibatches):
                    end = start + self.nminibatches
                    indices = permutation[start:end]
                    kl_dist, cmu, clogstd = self.algo.train(
                        observations[indices], actions[indices], returns[indices], values[indices], 
                        advantages[indices], neglogpacs[indices], mus[indices], logstds[indices], states[indices]
                    )
                    mus[indices] = cmu
                    logstds[indices] = clogstd
                    self.lr, _ = self.scheduler.update(self.lr, 0, 0, 0, kl_dist.item())
                    for param_group in self.algo.optimizer.param_groups:
                        param_group['lr'] = self.lr
                    if self.asymmetric_obs:
                        self.value_lr, _ = self.value_scheduler.update(self.value_lr, 0, update, 0, 0)
                        for param_group in self.algo.value_optimizer.param_groups:
                            param_group['lr'] = self.value_lr
                if self.normalize_input:
                    self.algo.model.policy.running_mean_std.eval()
                    if self.asymmetric_obs:
                        self.algo.model.value_network.running_mean_std.eval()
            if self.rewards_mean.current_size>0:
                current_mean_reward=self.rewards_mean.get_mean()[0]

                if self.save_freq>0:
                    if (update%self.save_freq==0):
                        checkpoint_name=self.config_name+'_ep_'+str(update)+'_rew_'+str(current_mean_reward)
                        self.save(os.path.join(self.save_dir, checkpoint_name))
        self.writer.close()
        if self.joint_training or self.uniform_random_contact:
            self.map.close()

    def _initialize_robot_joint_position(self):
        num_samples = self.nactors * 2
        T_O, T_G, q_R = self._get_fixed_initial_joint_position(num_samples)
        self.env.push_data(T_O, T_G, q_R)

    def _get_fixed_initial_joint_position(self, num_samples: int) -> Tuple[torch.Tensor, ...]:
        T_O, T_G = self.generator.sample(num_samples)
        q_R = self.env.get_initial_joint_position(num_samples)

        n_env = self.cfg['num_envs'] #24576 #128
        if n_env == '':
            n_env = 24576
        EE_position = T_O[:n_env, :3].clone().detach()
        EE_position[:, 2] = 0.52
        
        # joint_pose = self.sP.init_pos(EE_position, n_env)

        if self.robot_init == "pre_calculated_ik":
            joint_pose = self.sP.pre_calculated_IK(EE_position)
        
        if self.robot_init == "pre_calculated_random":
            joint_pose = self.sP.pre_calculated_random(n_env)
        
        if self.robot_init == "fixed_init_position":
            joint_pose = self.sP.fixed_init_pos(n_env)
        
        q_R[:] = torch.concatenate((joint_pose, joint_pose), 0)

        # q_R[:] = self.initial_robot_position
        return T_O, T_G, q_R

    # def _get_random_initial_joint_position(self, num_samples: int) -> Tuple[torch.Tensor, ...]:
    #     q_R = torch.zeros((num_samples, 9), dtype=torch.float, device=self.rl_device)
    #     T_O = torch.zeros((num_samples, 7), dtype=torch.float, device=self.rl_device)
    #     T_G = torch.zeros((num_samples, 7), dtype=torch.float, device=self.rl_device)
    #     sample_count = 0
    #     while sample_count < num_samples:
    #         _initial_object_pose, _goal_object_pose = self.generator.sample(size=100)
    #         jointPositionAndFeasiblity = self.joint_position_sampler.get_random_collision_free_joint_position(
    #             _initial_object_pose, self.seed
    #         )
    #         jointPosition = jointPositionAndFeasiblity[:, :-1]
    #         feasiblity = (jointPositionAndFeasiblity[:, -1] > 0.5)
    #         num_feasible_samples = torch.count_nonzero(feasiblity).item()
    #         start = sample_count
    #         end = sample_count + num_feasible_samples
    #         if end > num_samples: end = num_samples
    #         q_R[start:end] = jointPosition[feasiblity][:(end - start)]
    #         T_O[start:end] = _initial_object_pose[feasiblity][:(end - start)]
    #         T_G[start:end] = _goal_object_pose[feasiblity][:(end - start)]
    #         sample_count += num_feasible_samples
    #     return T_O, T_G, q_R


    def play(self):
        print("Play mode is for test environment")
        T_O, T_G = self.generator.sample(self.IK_query_size)
        self.env.push_data(T_O, T_G, torch.zeros((T_O.shape[0], 6), device=self.rl_device))
        # Array indicating whether current episode is reset from current version of pi_pre  
        obs_dict = self.env.reset()
        self.observation = torch.clone(obs_dict["obs"])
        if self.asymmetric_obs:
            self.state = torch.clone(obs_dict["states"])
        else:
            self.state = torch.zeros((self.nactors))         
        nplay = self.max_epochs *self.nsteps
        for _ in range(nplay):
            action, value, neglogp, mu, logstd = self.model.step(self.observation, self.state)
            action[:] = 0
            obs_dict, reward, reset, _ = self.env.step(action)
            self.observation = torch.clone(obs_dict["obs"])
            if self.asymmetric_obs:
                self.state = torch.clone(obs_dict["states"])

    def test(self):
        self.algo.model.policy.eval()
        self._initialize_robot_joint_position()
        self.env.extract = False 
        obs_dict = self.env.reset()
        self.observation = torch.clone(obs_dict["obs"])
        if self.asymmetric_obs:
            self.state = torch.clone(obs_dict["states"])
        else:
            self.state = torch.zeros((self.nactors))
        reset_count = 0
        success_count = 0
        episode_count = 0
        episode_suc_count = 0
        while True:
            action, value, neglogp, mu, logstd = self.model.step(self.observation, self.state)
            obs_dict, reward, reset, _ = self.env.step(action)
            reset_count += torch.count_nonzero(reset)
            success_count += torch.count_nonzero(self.env.env_succeed)
            env_reset_indices = reset.view(self.nactors, 1).all(dim=1).nonzero(as_tuple=False)
            env_suc_indices = self.env.env_succeed.view(self.nactors, 1).all(dim=1).nonzero(as_tuple=False)
            episode_count += torch.sum(self.env.progress_buf[env_reset_indices])
            episode_suc_count += torch.sum(self.env.progress_buf[env_suc_indices])
            if reset_count > 10000: break
            self.observation = torch.clone(obs_dict["obs"])
            if self.asymmetric_obs:
                self.state = torch.clone(obs_dict["states"])
        print(f"all tries {reset_count}, succeed {success_count} times and total episodes are {episode_count} and for success {episode_suc_count}")
        if self.joint_training:
            self.map.close()

class SysidRunner:
    def __init__(self, cfg, env: Sysid, real_data):
        self.env = env

        # parameters for training
        self.cfg = cfg
        self.rl_device = cfg["rl_device"]

        if 'joint_idx' in real_data and len(real_data['joint_idx']) > 0:
            self.joint_idx = torch.tensor(real_data['joint_idx'], dtype=torch.int32, device=self.rl_device)[0]
        else:
            self.joint_idx = None
        self.init_poses = torch.tensor(real_data['init_pos'], dtype=torch.float32, device=self.rl_device)
        self.torques = torch.tensor(real_data['torques'], dtype=torch.float32, device=self.rl_device)
        self.joint_pos_real = torch.tensor(real_data['joint_pos'], dtype=torch.float32, device=self.rl_device)
        self.joint_vel_real = torch.tensor(real_data['joint_vel'], dtype=torch.float32, device=self.rl_device)

        os.makedirs(SYSID_SAVE_DIR, exist_ok=True)
        
    def run_sim(self, params, vis: str=None) -> torch.Tensor:
        # init_poses = 0.1*torch.ones(self.env.num_env_per_param, 6, dtype=torch.float32, device=self.rl_device)
        # params = torch.zeros(self.env.num_params, 6, 3, dtype=torch.float32, device=self.rl_device)
        
        # action = torch.zeros(self.env.num_envs, 18, dtype=torch.float32, device=self.rl_device)
        self.env.reset_sysid(self.init_poses, params, self.joint_idx)

        # # Array indicating whether current episode is reset from current version of pi_pre  
        # obs_dict = self.env.reset()

        joint_pos_sim = []
        joint_vel_sim = []

        # self collision test
        # for i in range(1000000):
        #     action_repeated = self.action[:, 0, :].repeat(self.env.num_params, 1)
        #     action_repeated = torch.zeros_like(action_repeated)
        #     obs_dict = self.env.step(action_repeated)
        init_pos_repeated = self.init_poses.repeat(self.env.num_params, 1)
        # joint_pos_sim.append(init_pos_repeated)

        for i in range((int)(self.torques.shape[1]/10)):
            torque_repeated = self.torques[:, i*10:i*10+10, :].repeat(self.env.num_params, 1, 1)
            ten_joint_poses, ten_joint_vels = self.env.step_sysid(torque_repeated, self.joint_idx, init_pos_repeated)
            joint_pos_sim.append(torch.clone(ten_joint_poses)) # 10, N_env, N_dof
            joint_vel_sim.append(torch.clone(ten_joint_vels))

        joint_pos_sim = torch.cat(joint_pos_sim, dim=0) # 10*N_step N_env, N_dof
        joint_pos_sim = torch.transpose(joint_pos_sim, 0, 1) # N_env, 10*N_step, N_dof

        joint_pos_diff = (self.joint_pos_real.repeat(self.env.num_params, 1, 1) - joint_pos_sim)**2     # N_env, N_step, N_dof

        joint_vel_sim = torch.cat(joint_vel_sim, dim=0) # 10*N_step N_env, N_dof
        joint_vel_sim = torch.transpose(joint_vel_sim, 0, 1) # N_env, 10*N_step, N_dof

        

        def estimate_acceleration(velocities, dt=1./200.):
            """
            Estimate acceleration given positions and velocities using finite differences.
            
            Args:
                velocities (torch.Tensor): Tensor of shape (N,) representing velocities at each timestep.
                dt (float): Timestep size.
                
            Returns:
                torch.Tensor: Estimated accelerations of shape (N,).
            """
            # Central differences for internal points
            acc = torch.zeros_like(velocities)
            acc[:, 1:-1, :] = (velocities[:, 2:, :] - velocities[:, :-2, :]) / (2 * dt)
            
            # Forward difference for the first point
            acc[:, 0, :] = (velocities[:, 1, :] - velocities[:, 0, :]) / dt
            
            # Backward difference for the last point
            acc[:, -1, :] = (velocities[:, -1, :] - velocities[:, -2, :]) / dt
            
            return acc
        
        def moving_average_smooth_multidim(data, window_size=10):
            """
            Smooth the data along the second dimension using a moving average filter.
            
            Args:
                data (torch.Tensor): Input data of shape (B, T, D).
                window_size (int): Size of the moving average window.
                
            Returns:
                torch.Tensor: Smoothed data of shape (B, T, D).
            """
            # Create a uniform kernel for the moving average
            kernel = torch.ones(window_size, device='cuda') / window_size  # Shape: (window_size,)
            
            # Reshape kernel for group convolution
            kernel = kernel.view(1, 1, -1)  # Shape: (1, 1, window_size)
            
            # Expand kernel to match the number of channels
            kernel = kernel.expand(data.size(2), -1, -1)  # Shape: (D, 1, window_size)
            
            # Apply convolution along the second dimension
            smoothed_data = torch.conv1d(
                data.permute(0, 2, 1),  # Permute to (B, D, T)
                kernel,  # Expanded kernel
                padding = window_size // 2,
                groups=data.size(2)  # Ensure channel-wise convolution
            ).permute(0, 2, 1)  # Return to original shape (B, T, D)
            
            return smoothed_data
        
        import torch.nn.functional as F

        def gaussian_smooth_multidim(data, window_size=10, sigma=2.0):
            """
            Smooth the data along the second dimension using a Gaussian filter.
            
            Args:
                data (torch.Tensor): Input data of shape (B, T, D).
                window_size (int): Size of the Gaussian kernel window.
                sigma (float): Standard deviation for the Gaussian kernel.
                
            Returns:
                torch.Tensor: Smoothed data of shape (B, T, D).
            """
            # Create a Gaussian kernel
            kernel = torch.exp(-0.5 * (torch.arange(-(window_size // 2), window_size // 2 + 1).float()**2) / sigma**2)
            kernel = kernel / kernel.sum()  # Normalize to ensure sum equals 1
            kernel = kernel.to(data.device)

            # Reshape kernel for group convolution
            kernel = kernel.view(1, 1, -1)  # Shape: (1, 1, window_size)
            
            # Expand kernel to match the number of channels
            kernel = kernel.expand(data.size(2), -1, -1)  # Shape: (D, 1, window_size)
            
            # Apply convolution along the second dimension
            smoothed_data = F.conv1d(
                data.permute(0, 2, 1),  # Permute to (B, D, T)
                kernel,  # Expanded kernel
                padding=(window_size - 1) // 2,  # Dynamically adjust padding
                groups=data.size(2)  # Ensure channel-wise convolution
            ).permute(0, 2, 1)  # Return to original shape (B, T, D)
            
            # Truncate to original size if necessary
            if smoothed_data.shape[1] > data.shape[1]:
                smoothed_data = smoothed_data[:, :data.shape[1], :]
            
            return smoothed_data
        
        acc_real = gaussian_smooth_multidim(estimate_acceleration(self.joint_vel_real.repeat(self.env.num_params, 1, 1)), window_size=20, sigma=3.0)
        acc_sim = gaussian_smooth_multidim(estimate_acceleration(joint_vel_sim),  window_size=20, sigma=3.0)
        # acc_tough = estimate_acceleration(self.joint_vel_real.repeat(self.env.num_params, 1, 1))

        vel_real_smooth = gaussian_smooth_multidim(self.joint_vel_real.repeat(self.env.num_params, 1, 1))
        vel_sim_smooth = gaussian_smooth_multidim(joint_vel_sim)

        joint_vel_diff = (vel_real_smooth - vel_sim_smooth)**2
        
        # joint_vel_diff = (self.joint_vel_real.repeat(self.env.num_params, 1, 1) - joint_vel_sim)**2     # N_env, N_step, N_dof

        joint_acc_diff = (acc_real-acc_sim)**2

        if self.joint_idx is None:
            # joint_pos_diff = torch.sum(joint_pos_diff, axis=1)
            joint_vel_diff = torch.sum(joint_vel_diff, axis=1)
            joint_acc_diff = torch.sum(joint_acc_diff, axis=1)
        else:
            # joint_pos_diff = torch.sum(joint_pos_diff, axis=1)[:, self.joint_idx] # N_env, N_dof
            joint_vel_diff = torch.sum(joint_vel_diff, axis=1)[:, self.joint_idx] # N_env, N_dof
            joint_acc_diff = torch.sum(joint_acc_diff, axis=1)[:, self.joint_idx] # N_env, N_dof

        # joint_pos_diff = joint_pos_diff.reshape(self.env.num_params, -1) # self.env.num_params, N_env_per_param
        joint_vel_diff = joint_vel_diff.reshape(self.env.num_params, -1)[:, 1:] # self.env.num_params, N_env_per_param
        joint_acc_diff = joint_acc_diff.reshape(self.env.num_params, -1)[:, 1:] # self.env.num_params, N_env_per_param

    
        # params_cost = torch.mean(joint_pos_diff, axis=1) # self.env.num_params
        params_cost = torch.mean(joint_vel_diff, axis=1)/1300. + torch.mean(joint_acc_diff, axis=1)/360000. # self.env.num_params
        
        # params_cost = torch.mean(joint_acc_diff, axis=1) # self.env.num_params

        if vis is not None:

            # vis_idx = torch.argmin(params_cost)
            vis_idx = -1
            for vis_idx in range(1, 20):
                N_env = self.env.num_params

                import matplotlib.pyplot as plt
                joint_idx = self.joint_idx

                plt.figure(figsize=(12, 8))

                # x = list(range(len(self.joint_pos_real[vis_idx, :, joint_idx])))
                # plt.plot(x, self.joint_pos_real[vis_idx, :, joint_idx].detach().cpu().numpy()*180/np.pi, label='Real Joint Positions')
                # plt.plot(x, joint_pos_sim[vis_idx, :, joint_idx].detach().cpu().numpy()*180/np.pi, label='Simulated Joint Positions')

                x = list(range(len(self.joint_vel_real[vis_idx, :, joint_idx])))
                plt.plot(x, self.joint_vel_real[vis_idx, :, joint_idx].detach().cpu().numpy(), label='Real Joint Vel (rad/s)')
                plt.plot(x, joint_vel_sim[vis_idx, :, joint_idx].detach().cpu().numpy(), label='Simulated Joint Vel (rad/s)')

                # x = list(range(len(acc_real[vis_idx, :, joint_idx])))
                # plt.plot(x, acc_real[vis_idx, :, joint_idx].detach().cpu().numpy(), label='Smoothed Real joint acc (rad/s^2)')
                # plt.plot(x, acc_sim[vis_idx, :, joint_idx].detach().cpu().numpy(), label='Smoothed Simulated joint acc (rad/s^2)')
                # plt.plot(x, acc_tough[vis_idx, 1:-1, joint_idx].detach().cpu().numpy(), label='Real joint acc (rad/s^2)')
                
                # torque_seq = self.torques[vis_idx, :, joint_idx].detach().cpu().numpy()
                # plt.plot(x, 10.*torque_seq, label='torque sequence 10x (Nm)')

                plt.legend()
                plt.title(f'sysID joint {joint_idx+1}')
                plt.tight_layout()
                plt.savefig(SYSID_SAVE_DIR + f'sysid_cnt{vis}_joint{joint_idx+1}_scenario{vis_idx}.png')

                plt.cla()
                plt.clf()
                plt.close()

        return params_cost # params_cost_opt, params_cost_val


class PPODataset():
    def __init__(self,  device):
        self.device = device
        self._data = {}

    def update(self, key, value):
        if key in self._data:
            old_value = self._data[key]
            self._data[key] = torch.cat([old_value, value], 0)
        else:
            self._data[key] = value

    def __len__(self):
        return len(self._data[list(self._data.keys())[0]])
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            d = {}
            for k, v in self._data.items():
                d[k] = v[key.start:key.stop]
        return d

    def saf(self):
        for key, value in self._data.items():
            print(key,value.shape)
            self._data[key] = swap_and_flatten(value)

    def reset(self):
        self._data = {}
        
    def put(self, key, value):
        self._data[key] = value

    @property
    def data(self):
        return self._data
