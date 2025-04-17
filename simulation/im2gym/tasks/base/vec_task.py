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

from typing import Dict, Any, Tuple, List

import gym
from gym import spaces

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import to_torch
from isaacgym.gymutil import get_property_setter_map, get_property_getter_map, get_default_setter_args, apply_random_samples, generate_random_samples

import torch
from torch.distributions.distribution import Distribution
from torch.distributions.cauchy import Cauchy
from torch.distributions.laplace import Laplace
import numpy as np
import operator, random
from copy import deepcopy

import sys

import abc
from abc import ABC
# from simulation.im2gym.tasks.base.image_randomizer import ImageRandomizer
from collections import OrderedDict
from bisect import bisect


class Env(ABC):
    def __init__(self, config: Dict[str, Any], sim_device: str, graphics_device_id: int, headless: bool):
        """Initialise the env.

        Args:
            config: the configuration dictionary.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
        """

        split_device = sim_device.split(":")
        self.device_type = split_device[0]
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0

        self.device = "cpu"
        if config["sim"]["use_gpu_pipeline"]:
            if self.device_type.lower() == "cuda" or self.device_type.lower() == "gpu":
                self.device = "cuda" + ":" + str(self.device_id)
            else:
                print("GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline.")
                config["sim"]["use_gpu_pipeline"] = False

        self.rl_device = config.get("rl_device", "cuda:0")

        # Rendering
        # if training in a headless mode
        self.headless = headless

        enable_camera_sensors = config["env"].get("enableCameraSensors", False)
        self.graphics_device_id = graphics_device_id
        if enable_camera_sensors == False and self.headless == True:
            self.graphics_device_id = -1

        self.num_environments = config["env"]["numEnvs"]
        self.num_agents = config["env"].get("numAgents", 1)  # used for multi-agent environments
        self.num_observations = config["env"]["numObservations"]
        self.num_student_observations = config["env"]["numStudentObservations"]
        self.num_states = config["env"].get("numStates", 0)
        self.num_actions = config["env"]["numActions"]

        self.control_freq_inv = config["env"].get("controlFrequencyInv", 1)

        self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf)

        self.act_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)

        self.clip_obs = config["env"].get("clipObservations", np.Inf)
        self.clip_actions = config["env"].get("clipActions", np.Inf)

    @abc.abstractmethod 
    def allocate_buffers(self):
        """Create torch buffers for observations, rewards, actions dones and any additional data."""

    @abc.abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

    @abc.abstractmethod
    def reset(self)-> Dict[str, torch.Tensor]:
        """Reset the environment.
        Returns:
            Observation dictionary
        """

    @property
    def observation_space(self) -> gym.Space:
        """Get the environment's observation space."""
        return self.obs_space

    @property
    def action_space(self) -> gym.Space:
        """Get the environment's action space."""
        return self.act_space

    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self.num_environments

    @property
    def num_acts(self) -> int:
        """Get the number of actions in the environment."""
        return self.num_actions

    @property
    def num_obs(self) -> int:
        """Get the number of observations in the environment."""
        return self.num_observations

    @property
    def num_student_obs(self) -> int:
        """Get the number of observations in the environment."""
        return self.num_student_observations


class VecTask(Env):
    def __init__(self, config, sim_device, graphics_device_id, headless, gym=None, record=False):
        """Initialise the `VecTask`.

        Args:
            config: config dictionary for the environment.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
        """
        super().__init__(config, sim_device, graphics_device_id, headless)

        self.sim_params = self.__parse_sim_params(self.cfg["physics_engine"], self.cfg["sim"])
        if self.cfg["physics_engine"] == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
        elif self.cfg["physics_engine"] == "flex":
            self.physics_engine = gymapi.SIM_FLEX
        else:
            msg = f"Invalid physics engine backend: {self.cfg['physics_engine']}"
            raise ValueError(msg)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        if gym is None:
            self.gym = gymapi.acquire_gym()
        else:
            self.gym = gym

        self.first_randomization = True
        self.original_props = {}
        self.dr_randomizations = {}
        self.last_step = -1
        self.last_rand_step = -1

        # create envs, sim and viewer
        self.sim_initialized = False
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self.sim_initialized = True

        self.set_viewer()
        self.allocate_buffers()

        self.obs_dict = {}
        # self._image_randomizer = ImageRandomizer()

        # Simulating real OSC controller
        self.joint_torque_error_distributions: List[Distribution] = [
            Cauchy(torch.tensor([-0.05775765567736821]), torch.tensor([0.40806475094086614])),
            Cauchy(torch.tensor([-0.2907139508683057]), torch.tensor([1.0133192275983816])),
            Cauchy(torch.tensor([-0.07208627988273084]), torch.tensor([0.48509448198632255])),
            Cauchy(torch.tensor([0.0498790796468936]), torch.tensor([0.13319868553936284])),
            Laplace(torch.tensor([0.006411595749566054]), torch.tensor([0.09745708385954517])),
            Laplace(torch.tensor([0.013287506172192265]), torch.tensor([0.041132312519425555])),
            Cauchy(torch.tensor([0.008020268483317761]), torch.tensor([0.01759514632535382])),
        ]
        # self.torque_error_sample = torch.zeros((self.num_environments, 7), dtype=torch.float, device=self.rl_device)
        self.torque_error_sample = torch.zeros((self.num_environments, 6), dtype=torch.float, device=self.rl_device)

        # NOTE(dlee): these attributes must be initialized in the child class (Domain)
        # self._robot_dof_indices = None
        
        # Record
        self.record = record
        self.record_data: Dict[str, List] = dict()

    def set_viewer(self):
        """Create the viewer."""

        # TODO: find a better way to set this
        self.enable_viewer_sync = True
        if self.cfg["env"]["enableCameraSensors"]:
            self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

    def allocate_buffers(self):
        """Allocate the observation, states, etc. buffers.

        These are what is used to set observations and states in the environment classes which
        inherit from this one, and are read in `step` and other related functions.

        """

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
             self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

    #
    def set_sim_params_up_axis(self, sim_params: gymapi.SimParams, axis: str) -> int:
        """Set gravity based on up axis and return axis index.

        Args:
            sim_params: sim params to modify the axis for.
            axis: axis to set sim params for.
        Returns:
            axis index for up axis.
        """
        if axis == 'z':
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = -9.81
            return 2
        return 1

    def create_sim(self, compute_device: int, graphics_device: int, physics_engine, sim_params: gymapi.SimParams):
        """Create an Isaac Gym sim object.

        Args:
            compute_device: ID of compute device to use.
            graphics_device: ID of graphics device to use.
            physics_engine: physics engine to use (`gymapi.SIM_PHYSX` or `gymapi.SIM_FLEX`)
            sim_params: sim params to use.
        Returns:
            the Isaac Gym sim object.
        """
        sim = self.gym.create_sim(compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        return sim

    def get_state(self):
        """Returns the state buffer of the environment (the priviledged observations for asymmetric training)."""
        return torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    @abc.abstractmethod
    def pre_physics_step(self, actions: torch.Tensor):
        """Apply the actions to the environment (eg by setting torques, position targets).

        Args:
            actions: the actions to apply
        """

    @abc.abstractmethod
    def post_physics_step(self):
        """Compute reward and observations, reset any environments that require it."""

    @abc.abstractmethod
    def step_controller(self):
        """Step controller one step."""

    @abc.abstractmethod
    def refresh_buffer(self):
        """refresh data buffer from gym"""

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # randomize actions
        # if self.dr_randomizations.get('actions', None):
        #     actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)
       
        # step physics and render each frame
        for i in range(self.control_freq_inv):
            self.step_controller()
            self.gym.simulate(self.sim)
            self.refresh_buffer()
        self.render()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # fill time out buffer
        self.timeout_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.timeout_buf), torch.zeros_like(self.timeout_buf))

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        if self.record and self.reset_buf.all():
            import pickle
            with open('simulation_record.pkl', 'wb') as f:
                pickle.dump(self.record_data, f)
            exit()

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def zero_actions(self) -> torch.Tensor:
        """Returns a buffer with zero actions.

        Returns:
            A buffer of zero torch actions
        """
        actions = torch.zeros([self.num_envs, self.num_actions], dtype=torch.float32, device=self.rl_device)

        return actions

    def reset(self) -> torch.Tensor:
        """Reset the environment.
        Returns:
            Observation dictionary
        """
        zero_actions = self.zero_actions()

        # step the simulator
        self.step(zero_actions)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict
    
    def reset2(self) -> torch.Tensor:
        """Reset the environment.
        Returns:
            Observation dictionary
        """
        self.reset_buf[:]=1
        zero_actions = self.zero_actions()

        # step the simulator
        self.step(zero_actions)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, self.reset_buf.to(self.rl_device)

    def render(self):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

            else:
                self.gym.poll_viewer_events(self.viewer)

    def __parse_sim_params(self, physics_engine: str, config_sim: Dict[str, Any]) -> gymapi.SimParams:
        """Parse the config dictionary for physics stepping settings.

        Args:
            physics_engine: which physics engine to use. "physx" or "flex"
            config_sim: dict of sim configuration parameters
        Returns
            IsaacGym SimParams object with updated settings.
        """
        sim_params = gymapi.SimParams()

        # check correct up-axis
        if config_sim["up_axis"] not in ["z", "y"]:
            msg = f"Invalid physics up-axis: {config_sim['up_axis']}"
            print(msg)
            raise ValueError(msg)

        # assign general sim parameters
        sim_params.dt = config_sim["dt"]
        sim_params.num_client_threads = config_sim.get("num_client_threads", 0)
        sim_params.use_gpu_pipeline = config_sim["use_gpu_pipeline"]
        sim_params.substeps = config_sim.get("substeps", 2)

        # assign up-axis
        if config_sim["up_axis"] == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Y

        # assign gravity
        sim_params.gravity = gymapi.Vec3(*config_sim["gravity"])

        # configure physics parameters
        if physics_engine == "physx":
            # set the parameters
            if "physx" in config_sim:
                for opt in config_sim["physx"].keys():
                    if opt == "contact_collection":
                        setattr(sim_params.physx, opt, gymapi.ContactCollection(config_sim["physx"][opt]))
                    else:
                        setattr(sim_params.physx, opt, config_sim["physx"][opt])
        else:
            # set the parameters
            if "flex" in config_sim:
                for opt in config_sim["flex"].keys():
                    setattr(sim_params.flex, opt, config_sim["flex"][opt])

        # return the configured params
        return sim_params

    def destroy_sim(self):
        self.gym.destroy_viewer(self.viewer)
        for env in self.envs:
            self.gym.destroy_env(env)
        self.gym.destroy_sim(self.sim)
        return self.gym

    """
    Domain Randomization methods
    """

    def original_value_copy(self, dr_params: Dict[str, Any], env_ptr):

        if self.first_randomization:
            self.check_buckets(self.gym, self.envs, dr_params)
        
        # NOTE(dlee): Let this be determined by the child classes(Domain)
        # robot_dof_names = list()
        # for i in range(1, 7):
        #     robot_dof_names += [f'joint{i}'] # CAUTION: left/right different.
        # self._robot_dof_indices = OrderedDict.fromkeys(robot_dof_names, None)
        assert self._robot_dof_indices is not None or len(self._robot_dof_indices) > 0, '_robot_dof_indices must be initialized in the child class of VecTask'

        for actor, actor_properties in dr_params["actor_params"].items():
            for prop_name, prop_attrs in actor_properties.items():
                if prop_name == 'rigid_shape_properties':
                    if actor == 'robot':
                        handle = self.gym.find_actor_handle(env_ptr, actor)
                        prop = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
                        self.original_props[actor+'_ee_friction'] = deepcopy(prop[self.EE_handle].friction) # CAUTION: left/right different.
                    else:
                        handle = self.gym.find_actor_handle(env_ptr, actor)
                        prop = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
                        for p in prop:
                            self.original_props[actor+'friction'] = deepcopy(p.friction)
                if prop_name == 'dof_properties':
                    handle = self.gym.find_actor_handle(env_ptr, actor)
                    robot_dof_props = self.gym.get_actor_dof_properties(env_ptr, handle)

                    for k, dof_index in enumerate(self._robot_dof_indices.values()):
                        self.original_props[f'{k}_friction'] = deepcopy(robot_dof_props['friction'][dof_index])
                        self.original_props[f'{k}_damping'] = deepcopy(robot_dof_props['damping'][dof_index])
    
    # called in every post_physics_step
    def observation_and_scene_randomizer(self, dr_params: Dict[str, Any], obs_buf: torch.Tensor):
        return self.observation_randomizer(dr_params, obs_buf)
    
    def observation_randomizer(self, target_params, target_buf: torch.Tensor):
        dist = target_params["distribution"]
        op_type = target_params["operation"]
        op = operator.add if op_type == 'additive' else operator.mul

        if dist == 'gaussian':
            mu, var = target_params["range"]
            return op(target_buf, torch.randn_like(target_buf) * var + mu)

        elif dist == 'uniform':
            lo, hi = target_params["range"]
            return op(target_buf, torch.rand_like(target_buf) * (hi-lo) + lo)

    def image_randomizer(self, dr_params: Dict[str, Any], image: torch.Tensor, segmentation: torch.Tensor) -> torch.Tensor:
        image = self._image_randomizer.randomize_image(dr_params, image, segmentation)
        return image

    def camera_randomizer(self, dr_params: Dict[str, Any], camera_props, camera_position, camera_angle):
        if dr_params["camera"]["fov"]:
            camera_props = self._camera_fov_randomizer(camera_props)
        if dr_params["camera"]["transform"]:
            camera_position, camera_angle = self._camera_transformation_randomizer(camera_position, camera_angle)
        camera_transform = self._get_camera_transform(camera_position, camera_angle)
        return camera_props, camera_transform

    def _camera_fov_randomizer(self, camera_props):
        fov: float = camera_props.horizontal_fov
        fov += (-0.1 + torch.rand(1).item() * 0.2)
        camera_props.horizontal_fov = fov
        return camera_props

    def _camera_transformation_randomizer(self, camera_position, camera_angle):
        camera_position += (-0.005 + 0.01 * torch.rand((3,)))
        camera_angle += (-0.5 + torch.rand((1)).item())
        return camera_position, camera_angle

    def _get_camera_transform(self, camera_position, camera_angle):
        camera_transform = gymapi.Transform()
        camera_transform.p = gymapi.Vec3(*camera_position)
        camera_transform.r = (
            gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(180 - camera_angle)) 
            # * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.radians(180))
        )
        return camera_transform

    # called in env reset func
    def env_randomizer(self, dr_params: Dict[str, Any], env_ids: torch.Tensor):
        """Apply domain randomizations to the environment.

        Note that currently we can only apply randomizations only on resets, due to current PhysX limitations

        Args:
            dr_params: parameters for domain randomization to use.
        """
        
        for env_id in env_ids:
            for actor, actor_properties in dr_params["actor_params"].items():
                for prop_name, prop_attrs in actor_properties.items():
                    if prop_name == 'scale':
                        self.scale_randomizer(dr_params, env_id, actor, prop_attrs)
                    elif prop_name == 'color':
                        self.color_randomizer(dr_params, env_id, actor, prop_attrs)
                    elif prop_name == 'rigid_shape_properties':
                        self.friction_randomizer(env_id, actor, prop_attrs)
                    elif prop_name == 'dof_properties':
                        self.joint_randomizer(env_id, actor, prop_attrs)
                    else:
                        self.mass_randomizer(dr_params, env_id, actor, prop_name, prop_attrs)
                    
        self.first_randomization = False

    def scale_randomizer(self, dr_params: Dict[str, Any], env_id: torch.Tensor, actor, prop_attrs):
        env = self.envs[env_id]
        handle = self.gym.find_actor_handle(env, actor)
        setup_only = prop_attrs.get('setup_only', False)
        if (setup_only and not self.sim_initialized) or not setup_only:
            sample = generate_random_samples(prop_attrs, 1, self.last_step, None)
            og_scale = 1
            if prop_attrs['operation'] == 'scaling':
                new_scale = og_scale * sample[0]
            elif prop_attrs['operation'] == 'additive':
                new_scale = og_scale + sample[0]
            self.gym.set_actor_scale(env, handle, new_scale)

    def color_randomizer(self, dr_params: Dict[str, Any], env_id: torch.Tensor, actor, prop_attrs):
        env = self.envs[env_id]
        handle = self.gym.find_actor_handle(env, actor)
        num_bodies = self.gym.get_actor_rigid_body_count(env, handle)
        for n in range(num_bodies):
            self.gym.set_rigid_body_color(
                env, handle, n, gymapi.MESH_VISUAL, gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
            )

    def friction_randomizer(self, env_id: torch.Tensor, actor, prop_attrs):
        env = self.envs[env_id]
        handle = self.gym.find_actor_handle(env, actor)
        
        if actor=='robot':

            prop = self.gym.get_actor_rigid_shape_properties(env, handle)

            og_friction = self.original_props[actor+'_ee_friction'] # CAUTION
            sample = generate_random_samples(prop_attrs['friction'], 1, self.last_step, None)
            if prop_attrs['friction']['operation'] == 'scaling':
                sample[0] = self.get_bucketed_val(sample[0], prop_attrs['friction'])
                new_friction = og_friction * sample[0]
            elif prop_attrs['friction']['operation'] == 'additive':
                sample[0] = self.get_bucketed_val(sample[0], prop_attrs['friction'])
                new_friction = og_friction + sample[0]
            prop[self.EE_handle].friction = new_friction

            self.gym.set_actor_rigid_shape_properties(env, handle, prop)
        
        else:
            prop = self.gym.get_actor_rigid_shape_properties(env, handle)
            for p in prop:
                sample = generate_random_samples(prop_attrs['friction'], 1, self.last_step, None)
                og_friction = self.original_props[actor+"friction"]
                if prop_attrs['friction']['operation'] == 'scaling':
                    sample[0] = self.get_bucketed_val(sample[0], prop_attrs['friction'])
                    new_friction = og_friction * sample[0]
                elif prop_attrs['friction']['operation'] == 'additive':
                    sample[0] = self.get_bucketed_val(sample[0], prop_attrs['friction'])
                    new_friction = og_friction + sample[0]
                p.friction = new_friction
                
            self.gym.set_actor_rigid_shape_properties(env, handle, prop)
        
    
    def joint_randomizer(self, env_id: torch.Tensor, actor, prop_attrs):
        env = self.envs[env_id]
        handle = self.gym.find_actor_handle(env, actor)
        robot_dof_props = self.gym.get_actor_dof_properties(env, handle)
        
        for k, dof_index in enumerate(self._robot_dof_indices.values()):
            if k>5: break
            sample = generate_random_samples(prop_attrs['friction'], 1, self.last_step, None)
            og_friction = self.original_props[f'{k}_friction']
            if prop_attrs['friction']['operation'] == 'scaling':
                new_friction = og_friction * sample[0]
            elif prop_attrs['friction']['operation'] == 'additive':
                new_friction = og_friction + sample[0]

            robot_dof_props['friction'][dof_index] = new_friction

            sample = generate_random_samples(prop_attrs['damping'], 1, self.last_step, None)
            og_damping = self.original_props[f'{k}_damping']
            if prop_attrs['damping']['operation'] == 'scaling':
                new_damping = og_damping * sample[0]
            elif prop_attrs['damping']['operation'] == 'additive':
                new_damping = og_damping + sample[0]

            robot_dof_props['damping'][dof_index] = new_damping


        self.gym.set_actor_dof_properties(env, handle, robot_dof_props)

    def mass_randomizer(self, dr_params: Dict[str, Any], env_id: torch.Tensor, actor, prop_name, prop_attrs):
        
        param_setters_map = get_property_setter_map(self.gym)
        param_setter_defaults_map = get_default_setter_args(self.gym)
        param_getters_map = get_property_getter_map(self.gym)

        env = self.envs[env_id]
        handle = self.gym.find_actor_handle(env, actor)
        prop = param_getters_map[prop_name](env, handle)
        set_random_properties = True
        if isinstance(prop, list):
            if self.first_randomization:
                self.original_props[prop_name] = [
                    {attr: getattr(p, attr) for attr in dir(p)} for p in prop]
            for p, og_p in zip(prop, self.original_props[prop_name]):
                for attr, attr_randomization_params in prop_attrs.items():
                    setup_only = attr_randomization_params.get('setup_only', False)
                    if (setup_only and not self.sim_initialized) or not setup_only:
                        smpl = None
                        apply_random_samples(
                            p, og_p, attr, attr_randomization_params,
                            self.last_step, smpl
                        )
                    else:
                        set_random_properties = False
        else:
            if self.first_randomization:
                self.original_props[prop_name] = deepcopy(prop)
            for attr, attr_randomization_params in prop_attrs.items():
                setup_only = attr_randomization_params.get('setup_only', False)
        
                if (setup_only and not self.sim_initialized) or not setup_only:
                    smpl = None
                    apply_random_samples(
                        prop, self.original_props[prop_name], attr,
                        attr_randomization_params, self.last_step, smpl
                    )
                else:
                    set_random_properties = False

        if set_random_properties:
            setter = param_setters_map[prop_name]
            default_args = param_setter_defaults_map[prop_name]
            setter(env, handle, prop, *default_args)

    # called in every reset_idx function
    def table_z_randomizer(self, dr_params: Dict[str, Any], env_id: torch.Tensor):
        table_z = self.cfg["env"]["geometry"]["table"]["z"]
        dr_range=dr_params["table_height"]["range"]

        sample = generate_random_samples(dr_params['table_height'], env_id.shape[0], self.last_step, None)
        dr_sample = np.zeros(env_id.shape[0])
        if dr_params['table_height']['operation'] == 'scaling':
            dr_sample = table_z * sample
        elif dr_params['table_height']['operation'] == 'additive':
            dr_sample = table_z + sample
        return dr_sample
        



    # called in every controller step
    def torque_randomizer(self, dr_params: Dict[str, Any], computed_torque: torch.Tensor, restrict_gripper):
        prop_attrs = dr_params["torque"]

        JointTorqueDim = 6

        if (prop_attrs["operation"] == "scaling"):
            sample = generate_random_samples(prop_attrs, JointTorqueDim, self.last_step, None)
            self.computed_torque = torch.mul(self.computed_torque, torch.Tensor(sample).to(self.rl_device))

        elif (prop_attrs["operation"] == "additive"):
            sample = generate_random_samples(prop_attrs, JointTorqueDim, self.last_step, None)
            self.computed_torque[:, :] = torch.add(self.computed_torque[:, :], torch.Tensor(sample).to(self.rl_device))

        elif (prop_attrs["operation"] == "simulate_real"):
            if restrict_gripper == True:
                for i in range(6):
                    self.torque_error_sample[:, i] = self.joint_torque_error_distributions[i].sample((self.num_environments,)).squeeze(dim=-1)
                self.computed_torque[:,:6] += self.torque_error_sample

    def check_buckets(self, gym, envs, dr_params):

        total_num_buckets = 0
        for actor, actor_properties in dr_params["actor_params"].items():
            cur_num_buckets = 0

            if 'rigid_shape_properties' in actor_properties.keys():
                prop_attrs = actor_properties['rigid_shape_properties']
                if 'restitution' in prop_attrs and 'num_buckets' in prop_attrs['restitution']:
                    cur_num_buckets = prop_attrs['restitution']['num_buckets']
                if 'friction' in prop_attrs and 'num_buckets' in prop_attrs['friction']:
                    if cur_num_buckets > 0:
                        cur_num_buckets *= prop_attrs['friction']['num_buckets']
                    else:
                        cur_num_buckets = prop_attrs['friction']['num_buckets']
                total_num_buckets += cur_num_buckets

        assert total_num_buckets <= 64000, 'Explicit material bucketing has been specified, but the provided total bucket count exceeds 64K: {} specified buckets'.format(
            total_num_buckets)

        shape_ct = 0

        # Separate loop because we should not assume that each actor is present in each env
        for env in envs:
            for i in range(gym.get_actor_count(env)):
                actor_handle = gym.get_actor_handle(env, i)
                actor_name = gym.get_actor_name(env, actor_handle)
                if actor_name in dr_params["actor_params"] and 'rigid_shape_properties' in dr_params["actor_params"][actor_name]:
                    shape_ct += gym.get_actor_rigid_shape_count(env, actor_handle)

        assert shape_ct <= 64000 or total_num_buckets > 0, 'Explicit material bucketing is not used but the total number of shapes exceeds material limit. Please specify bucketing to limit material count.'

    def get_bucketed_val(self, new_prop_val, attr_randomization_params):
        if attr_randomization_params['distribution'] == 'uniform':
            # range of buckets defined by uniform distribution
            lo, hi = attr_randomization_params['range'][0], attr_randomization_params['range'][1]
        else:
            # for gaussian, set range of buckets to be 2 stddev away from mean
            lo = attr_randomization_params['range'][0] - 2 * np.sqrt(attr_randomization_params['range'][1])
            hi = attr_randomization_params['range'][0] + 2 * np.sqrt(attr_randomization_params['range'][1])
        num_buckets = attr_randomization_params['num_buckets']
        buckets = [(hi - lo) * i / num_buckets + lo for i in range(num_buckets)]
        return buckets[bisect(buckets, new_prop_val) - 1]