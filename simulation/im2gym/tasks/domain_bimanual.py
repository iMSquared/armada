import sys, os
import math
import numpy as np
import torch
import time

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from collections import OrderedDict, deque
from types import SimpleNamespace
from omegaconf import OmegaConf
from typing import Deque, Dict, Tuple, List, Union
from scipy.spatial.transform import Rotation as R

from simulation.utils.torch_jit_utils import *
from simulation.im2gym import *
from simulation.im2gym.tasks.base.vec_task import VecTask
from simulation.im2gym.algos.policy import Model
from simulation.im2gym.algos.utils import load_checkpoint
from control import DefaultControllerValues as RobotProp
from simulation.im2gym.tasks.utils.control import RobotControl

from pathlib import Path

BASEDIR = Path(__file__).parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))

class DomainBimanual(VecTask):

    # constants
    # directory where assets for the simulator are present
    _assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
    
    # dimensions of the system
    _dims = Dimensions
    _state_history_len = 2

    _ee_limits: dict = {
        "ee_position": SimpleNamespace(
            low=np.array([-1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32)
        ),
        "ee_orientation": SimpleNamespace(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32)
        ),
        "ee_velocity": SimpleNamespace(
            low=np.full(_dims.VelocityDim.value, -3, dtype=np.float32),
            high=np.full(_dims.VelocityDim.value, 3, dtype=np.float32)
        )
    }

    _object_limits: dict = {
        "position": SimpleNamespace(
            low=np.array([-1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1.5], dtype=np.float32) # maximum height fixed to 1.5m
        ),
        "orientation": SimpleNamespace(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            default=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        ),
        "velocity": SimpleNamespace(
            low=np.full(_dims.VelocityDim.value, -2, dtype=np.float32),
            high=np.full(_dims.VelocityDim.value, 2, dtype=np.float32),
            default=np.zeros(_dims.VelocityDim.value, dtype=np.float32)
        ),
        # 2D keypoint - deprecated
        # "2Dkeypoint": SimpleNamespace(
        #     low=np.array([0, 0], dtype=np.float32),
        #     high=np.array([320, 240], dtype=np.float32) #TODO: make this to be changed by the config file
        # )
    }
    # gripper links state list([num. of instances, num. of fingers, 13]) where 13: (x, y, z, quat, v, omega)
    # The length of list is the history of the state: 0: t, 1: t-1, 2: t-2, ... step.
    _EE_frames_state_history: Deque[torch.Tensor] = deque(maxlen=_state_history_len)
    
    # Object prim state [num. of instances, 13] where 13: (x, y, z, quat, v, omega)
    # The length of list is the history of the state: 0: t, 1: t-1, 2: t-2, ... step.
    _object_state_history: Deque[torch.Tensor] = deque(maxlen=_state_history_len)
   
    def __init__(self, cfg, sim_device, graphics_device_id, headless, use_state=False, gym=None, record=False, reset_policy=None):
        """Initializes the card environment configure the buffers.

        Args:
            cfg: Dictionory containing the configuration.
            sim_device: Torch device to store created buffers at (cpu/gpu).
            graphics_device_id: device to render the objects
            headless: if True, it will open the GUI, otherwise, it will just run the server.
        """
        # load default config
        self.cfg = cfg

        self._object_dims: CuboidalObject = self._set_object_dimension(self.cfg["env"]["geometry"]["object"])

        self.action_dim = RobotProp.DOF*3*2 # res kp kd 18 * 2 bimanual

        self.keypoints_num = int(self.cfg['env']['keypoint']['num'])
        

        self.obs_spec = {
            # robot joint
            "robot_q": self._dims.GeneralizedCoordinatesDim.value,
            # robot joint velocity
            "robot_u": self._dims.GeneralizedVelocityDim.value,
            # 2D keypoint - deprecated
            # # object position represented as 2D kepoints
            # "object_q": (self._dims.TwoDimensionKeypointDim.value * self.keypoints_num),
            # # object goal position represented as 2D kepoints
            # "object_q_des": (self._dims.TwoDimensionKeypointDim.value * self.keypoints_num),
            # object pose
            "object_pose": self._dims.ObjectPoseDim.value,
            # hand pose
            "hand_state": self._dims.ObjectPoseDim.value*2,
            # previous action
            "command": self.action_dim
        }

        # actions
        self.action_spec = {
            "command": self.action_dim
        }

        # student observations
        self.stud_obs_spec = {
            # end-effector pose
            'end-effector_pose': self._dims.PoseDim.value,
            # width
            'width': self._dims.WidthDim.value,
            # joint position (without gripper)
            'joint_position_wo_gripper': self._dims.GeneralizedVelocityDim.value,
            # joint velocity (without gripper)
            'joint_velocity_wo_gripper': self._dims.GeneralizedVelocityDim.value,
            # previous action
            "command": self.action_dim
        }
        self.cfg["env"]["numObservations"] = sum(self.obs_spec.values())
        self.cfg["env"]["numActions"] = sum(self.action_spec.values())
        self.cfg["env"]["numStudentObservations"] = sum(self.stud_obs_spec.values())
        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.random_external_force = self.cfg['task']['random_external_force']
        self.observation_randomize = self.cfg["task"]["observation_randomize"]
        self.image_randomize = self.cfg["task"]["image_randomize"]
        self.env_randomize = self.cfg["task"]["env_randomize"]
        self.torque_randomize = self.cfg["task"]["torque_randomize"]
        self.camera_randomize = self.cfg["task"]["camera_randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.uniform_random_contact = False

        self.dof_pos_offset = self.cfg["env"]["initial_dof_pos_limit"]
        self.dof_vel_offset = self.cfg["env"]["initial_dof_vel_limit"]

        if self.cfg["env"]["adaptive_dof_pos_limit"]["activate"]:
            self.dof_pos_limit_maximum = self.cfg["env"]["adaptive_dof_pos_limit"]["maximum"]
        
        if self.cfg["env"]["adaptive_dof_vel_limit"]["activate"]:
            self.dof_vel_limit_maximum = self.cfg["env"]["adaptive_dof_vel_limit"]["maximum"]

        # define prims present in the scene
        prim_names = ["robot"]
        prim_names += ["object", "goal_object"]
        if self.cfg['env']['scene_randomization']['background']:
            prim_names += ["floor", "back"]
        # mapping from name to asset instance
        self.asset_handles = dict.fromkeys(prim_names)

        # mapping from name to gym rigid body handles
        # name of finger tips links i.e. end-effector frames
        self.hand = self.cfg['env']['hand'] # should be BIMANUAL='both'
        ee_frames = list()
        robot_dof_names = list()
        for hand in [LEFT, RIGHT]:
            ee_frames.append(EE_NAMES[hand])
            for i in range(JOINT_FIRST[hand], JOINT_LAST[hand]+1):
                robot_dof_names.append(f'joint{i}')

        assert len(robot_dof_names) > 0, 'Hand name might be wrong (left/right/both)'
        self._ee_handles = OrderedDict.fromkeys(ee_frames, None)
        self._robot_dof_indices = OrderedDict.fromkeys(robot_dof_names, None)

        # Inductive reward. This is used for the baseline for claim 1 only.
        self.inductive_reward = self.cfg["env"]["reward_terms"]["inductive_reward"]["activate"]
        self.energy_reward = self.cfg["env"]["reward_terms"]["energy_reward"]["activate"]

        # Camera sensor
        if self.cfg["env"]["enableCameraSensors"]:
            self.camera_handles = list()
            self._torch_camera_rgba_images: List[torch.Tensor] = list()
            if self.cfg['env']["camera"]["segmentation"]:
                self._torch_camera_segmentation: List[torch.Tensor] = list()
        
        # During initialization its parent create_sim is called
        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, gym=gym, record=record)

        # Initialize the buffers
        self.__initialize()

        # Controller
        self.robot_control = RobotControl(self.cfg, self.device, self.graphics_device_id, self.j_eef)
        self.controllers={"position": self.step_jp, "position_2": self.step_jp}
        self.controller = self.controllers[self.cfg["env"]["controller"]]
        self.compute_target = self.compute_joint_target
        self.hold_policy: Model = None
        if reset_policy is not None:
            self.load_reset_policy(reset_policy)

        if self.viewer != None:
            # cam_pos = gymapi.Vec3(0.8, 0.0, 0.8)
            cam_pos = gymapi.Vec3(2, -1.0, 1)
            cam_target = gymapi.Vec3(0.5, 0.0, 0.4)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
        for limit_name in self._object_limits:
            # extract limit simple-namespace
            limit_dict = self._object_limits[limit_name].__dict__
            # iterate over namespace attributes
            for prop, value in limit_dict.items():
                limit_dict[prop] = torch.tensor(value, dtype=torch.float, device=self.device)
        for limit_name in self._ee_limits:
            # extract limit simple-namespace
            limit_dict = self._ee_limits[limit_name].__dict__
            # iterate over namespace attributes
            for prop, value in limit_dict.items():
                limit_dict[prop] = torch.tensor(value, dtype=torch.float, device=self.device)

        _camera_position = torch.tensor(OmegaConf.to_object(self.cfg["env"]["camera"]["position"]), device=self.device).unsqueeze(-1)
        _camera_angle = float(self.cfg["env"]["camera"]["angle"])
        rotation_matrix = torch.tensor((R.from_rotvec(np.array([0.,1.,0.])*np.radians(-90-_camera_angle))\
                                        *R.from_rotvec(np.array([0.,0.,1.,])*np.radians(90))\
                                        *R.from_rotvec(np.array([0.,0.,1.,])*np.radians(180))).inv().as_matrix(),dtype=torch.float).to(self.device)
        self.translation_from_camera_to_object = torch.zeros((3, 4), device=self.device)
        self.translation_from_camera_to_object[:3, :3] = rotation_matrix
        self.translation_from_camera_to_object[:3, 3] = -rotation_matrix.mm(_camera_position)[:, 0]
        self.camera_matrix = self.compute_camera_intrinsics_matrix(int(self.cfg["env"]["camera"]["size"][1]), int(self.cfg["env"]["camera"]["size"][0]), 79.3706, self.device) # 55.368

        # Set the mdp spaces
        self.__configure_mdp_spaces()

        # Reward setting
        self.get_reward_weight()

        # Initialize for photorealistic rendering
        if self.cfg["env"]["nvisii"]["photorealistic_rendering"]:
            self.__init_photorealistic_rendering(headless)
        
        # Save previous smoothed action for interpolation
        # dim=19 because quaternion of the residual is saved
        self.smoothing_coefficient = self.cfg["env"]["smoothing_coefficient"]
        self.previous_smoothed_action = torch.zeros((self.num_envs, 18), dtype=torch.float, device=self.device)

        # Save previous keypoints
        # self.prev_keypoints = torch.ones(self.num_envs, 8, 3, dtype=torch.float32, device=self.device)

        self.sin_t = 0
        # self.start_time_ = time.time()
        # self.end_time_ = 0
        # self.elapsed_time = 0

        # self.itr = 0
        # self.flag_jiho = False
        # self.init_pos = None
        # self.init_rot = None
        # self.itr_count = 0

        # self.itrs = []
        # self.desired_EE = []
        # self.cur_EE = []
        # self.init_axis_rot = None


        self._obs__ = []
        self.__velocity__ = []

        if self.record:
            self.record_data['torque'] = []
            self.record_data['obs'] = []
            self.record_data['joint_pos'] = []
            self.record_data['act'] = []

    def load_reset_policy(self, model: Model):
        state: Dict = load_checkpoint(self.cfg['env']['geometry']['hold_policy'])
        post_state = state.pop('post')
        model.policy.load_state_dict(post_state)

        print(f"restored running mean for value(count, mean, std):\
         {model.policy.value_mean_std.count} {model.policy.value_mean_std.running_mean} {model.policy.value_mean_std.running_var}")
        print(f"restored running mean for obs(count, mean, std):\
         {model.policy.running_mean_std.count} {model.policy.running_mean_std.running_mean} {model.policy.running_mean_std.running_var}")
        
        model.policy.eval()

        self.hold_policy = model

    def allocate_buffers(self):
        super().allocate_buffers()
        self.reward_terms = dict()
        for reward_name in self.cfg['reward']['weight'].keys():
            self.reward_terms[reward_name] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)


    def get_reward_weight(self):
        # Read weight config
        weight_cfg = self.cfg['reward']['weight']
        
        # Weights
        self.inductive_weight = weight_cfg['inductive']
        self.fall_weight = weight_cfg['fall']
        self.z_weight = weight_cfg['z']
        self.distance_weight = weight_cfg['distance']
        self.velocity_weight = weight_cfg['velocity']


    def use_uniform_random_contact(self):
        self.uniform_random_contact = True

    def _set_object_dimension(self, object_dims) -> CuboidalObject:
        return CuboidalObject((object_dims["width"], object_dims["length"], object_dims["height"]))


    """
    Protected members - Implementation specifics.
    """

    def create_sim(self):
        """
        Setup environment and the simulation scene.
        """
        # define ground plane for simulation
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.0
        # create ground
        self.gym.add_ground(self.sim, plane_params)
        # define scene assets
        self.__create_scene_assets()
        # create environments
        self.__create_envs()

        env_ptr = self.envs[0]
        
        # If randomizing, apply once immediately on startup before the fist sim step
        if self.env_randomize:
            self.original_value_copy(self.randomization_params, env_ptr)
            env_ids = list(range(self.num_envs))
            self.env_randomizer(self.randomization_params, env_ids)

    def pre_physics_step(self, actions):
        """
        Setting of input actions into simulator before performing the physics simulation step.
        """
        
        # Light randomization
        if self.cfg['env']['scene_randomization']['light']:
            self.randomize_light()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.env_ids = env_ids
        self.actions = actions.clone().to(self.device)
        self.actions[env_ids, :] = 0.0
        
        # If normalized_action is true, then denormalize them.
        if self.cfg['env']["normalize_action"]:
            action_transformed = unscale_transform(
                self.actions,
                lower=self._action_scale.low,
                upper=self._action_scale.high
            )
        else:
            action_transformed = self.actions
        self.action_transformed = action_transformed

        # Compute target ee pose or joint pose
        self.compute_target()
        self.mean_energy[:] = 0
        self.max_torque[:] = 0

        if self.record:
            self.record_data['act'].append(actions.detach().cpu().numpy())

    def compute_joint_target(self): 
        """compute the joint target to run the controller"""

        l_desired = self.robot_control.compute_joint_target(LEFT,
                                                            self.action_transformed,
                                                            self.env_ids,
                                                            self.dof_position,
                                                            self.franka_dof_lower_limits,
                                                            self.franka_dof_upper_limits)
        r_desired = self.robot_control.compute_joint_target(RIGHT,
                                                            self.action_transformed,
                                                            self.env_ids,
                                                            self.dof_position,
                                                            self.franka_dof_lower_limits,
                                                            self.franka_dof_upper_limits)

        self.desired_joint_position = torch.cat((l_desired, r_desired), axis=1)

    def step_jp(self):

        # Obtain gains
        ct = self.robot_control.compute_torque([LEFT, RIGHT],
                                               self.desired_joint_position,
                                               self.dof_position,
                                               self.dof_velocity,
                                               self.action_transformed)

        return ct
    
    
    def step_controller(self):
        computed_torque = torch.zeros(
            self.num_envs, self._dims.JointTorqueDim.value, dtype=torch.float32, device=self.device
        )
        ct = self.controller()
        computed_torque[:, :DOF*2] = ct
        
        if self.cfg["env"]["deadzone"]["activate"]:
            deadzone_th = self.cfg["env"]["deadzone"]["threshold_torque"] # dim 6 list
            # if any element in abs(computed_torque) (dimension N_env, 6) is smaller than deadzone_th, make it zero 
            abs_computed_torque = torch.abs(computed_torque)
            deadzone_th_tensor = torch.tensor(deadzone_th, device=self.rl_device)
            random_scalars = 1. + torch.rand(deadzone_th_tensor.size(), device=self.rl_device)

            # Multiply each element in deadzone_th by a random scalar between 1 and 2
            deadzone_th_scaled = deadzone_th_tensor * random_scalars

            # Create a mask where values less than the threshold are set to zero
            mask = abs_computed_torque >= deadzone_th_scaled  # shape (N_env, 6)

            # Apply the mask to computed_torque, setting values below the threshold to zero
            computed_torque = computed_torque * mask.float() 

        self.computed_torque = computed_torque
        
        if self.record:
            self.record_data['torque'].append(computed_torque.detach().cpu().numpy())

        # apply Domain Randomization before saturating torque
        if self.torque_randomize:
            self.torque_randomizer(self.randomization_params, self.computed_torque, self.cfg["env"]["restrict_gripper"])
        applied_torque = saturate(
            self.computed_torque,
            lower=-self.franka_dof_effort_scales,
            upper=self.franka_dof_effort_scales
        )
        applied_torque[self.env_ids, :] = 0
        self.max_torque = torch.maximum(self.max_torque, torch.norm(applied_torque[:,:12],dim=-1))
        self.applied_torque = applied_torque

        # set computed torques to simulator buffer.
        zero_torque = torch.zeros_like(self.applied_torque)
        # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(zero_torque))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(applied_torque))

    def refresh_buffer(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.mean_energy = torch.add(self.mean_energy, torch.norm(self.dof_velocity[:, :6] * self.applied_torque[:, :6], dim=-1))

    def post_physics_step(self):
        """
        Setting of buffers after performing the physics simulation step.

        @note Also need to update the reset buffer for the instances that have terminated.
              The termination conditions to check are besides the episode timeout.
        """
        # count step for each environment
        self.progress_buf += 1
        self.randomize_buf += 1
        # fill observations buffer
        self.compute_observations()
        # compute rewards
        self.compute_reward(self.actions)
        
        # check termination
        self._check_termination()

        if self.observation_randomize: # TODO: change self.randomize option separately
            # vel
            # self.obs_buf[:, 6:12] = self.keypoints_randomizer(self.randomization_params, self.obs_buf[:, 6:12])
            # obs_buf, torch.randn_like(obs_buf) * var + mu

            # DEBUG: should re-check the observation dimension
            breakpoint()

            # self.obs_buf[:, 12:28] = self.observation_randomizer(self.randomization_params["keypoints"], self.obs_buf[:, 12:28]) # cur keypoint
            self.obs_buf[:, :6] = self.observation_randomizer(self.randomization_params["joint_pos"], self.obs_buf[:, :6]) # joint pos
            self.obs_buf[:, 6:12] = self.observation_randomizer(self.randomization_params["joint_vel"], self.obs_buf[:, 6:12]) # joint vel
            self.obs_buf[:, 44:51] = self.observation_randomizer(self.randomization_params["hand_pos"], self.obs_buf[:, 44:51]) # hand pose
            # self.extras["regularization"]=self.regularization
        
    def compute_reward(self, actions):
        self.rew_buf[:] = 0
        self.reset_buf[:] = 0

        # Get values from configs
        object_info = self.cfg['env']['geometry']['object']
        table_info = self.cfg['env']['geometry']['table']
        
        object_dims = (object_info['width'], object_info['length'], object_info['height'])
        table_dims = (table_info['width'], table_info['length'], table_info['height'])
        table_position = (table_info['x'], table_info['y'], table_info['z'])

        robot_pos = self.cfg['env']['robot']['p']

        # Compute reset
        self.reset_buf[:] = compute_reset(self.reset_buf, self.progress_buf, self.max_episode_length)

        # Compute each reward term
        distance = compute_flight_distance_reward(self._object_state_history[0],
                                                  torch.tensor(robot_pos, device=self.device))

        fall = compute_fall_penalty(self._object_state_history[0],
                                    object_dims,
                                    table_dims,
                                    table_position)
        
        inductive_reward = compute_inductive_reward(self.cfg["env"]["reward_terms"]["object_dist"]["epsilon"],
                                                    self._object_state_history[0],
                                                    self._EE_frames_state_history[0])
        
        # height_reward = compute_height_reward(self._object_state_history[0],
        #                                       object_dims,
        #                                       table_dims,
        #                                       table_position)

        height_reward = compute_apogee_reward(self.apogee, self.reset_buf)
        
        velocity = compute_velocity_reward(self._object_state_history[0])
        velocity_reward = compute_max_speed_reward(self.max_speed, self.reset_buf)
        
        below_table = check_object_below_table(self._object_state_history[0],
                                              object_dims,
                                              table_dims,
                                              table_position)
        
        # Update dropped
        self.dropped = torch.logical_or(self.dropped, fall)

        # Update apogee
        indices = self.apogee < self._object_state_history[0][:,2]
        self.apogee[indices] = self._object_state_history[0][:,2][indices]

        # Update speed
        indices = self.max_speed < velocity
        self.max_speed[indices] = velocity[indices]

        # Compute total reward
        weighted_inductive_reward = self.inductive_weight*inductive_reward*fall + 0.01*self.inductive_weight*inductive_reward*(~fall)
        weighted_fall_reward = self.fall_weight*fall
        weighted_z_reward = torch.clamp(self.z_weight*height_reward, min=0)*self.reset_buf
        weighted_distance_reward = self.distance_weight*distance*distance*(~fall)*self.reset_buf
        weighted_velocity_reward = self.velocity_weight*velocity_reward

        self.rew_buf[:] = weighted_inductive_reward + weighted_fall_reward + weighted_z_reward + weighted_distance_reward + weighted_velocity_reward
        
        # Log each rewards
        self.reward_terms['inductive'][:] = weighted_inductive_reward
        self.reward_terms['fall'][:] = weighted_fall_reward
        self.reward_terms['z'][:] = weighted_z_reward
        self.reward_terms['distance'][:] = weighted_distance_reward
        self.reward_terms['velocity'][:] = weighted_velocity_reward

        
        
    """
    Private functions
    """

    def __initialize(self):
        """Allocate memory to various buffers.
        """
        if self.cfg['env']["enableCameraSensors"]:
            rgb_image_shape = [self.num_envs] + OmegaConf.to_object(self.cfg["env"]["camera"]["size"]) + [3]
            self._camera_image = torch.zeros(rgb_image_shape, dtype=torch.uint8, device=self.rl_device)
            segmentation_shape = [self.num_envs] + OmegaConf.to_object(self.cfg["env"]["camera"]["size"]) + [1]
            self._segmentation = torch.zeros(segmentation_shape, dtype=torch.int32, device=self.rl_device)
        if self.cfg['env']["nvisii"]["photorealistic_rendering"]:
            rgb_image_shape = [1] + OmegaConf.to_object(self.cfg["env"]["camera"]["size"]) + [3]
            self._photorealistic_image = torch.zeros(rgb_image_shape, dtype=torch.uint8, device=self.rl_device)
        # store the sampled goal poses for the object: [num. of instances, 7]
        self._object_goal_poses_buf = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float)
        
        # get gym GPU state tensors
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "robot")
        _mm = self.gym.acquire_mass_matrix_tensor(self.sim, "robot")
        _dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        
        # refresh the buffer
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)
        
        # create wrapper tensors for reference (consider everything as pointer to actual memory)
        # DOF
        self.dof_state: torch.Tensor = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self.dof_position = self.dof_state[..., 0]
        self.dof_velocity = self.dof_state[..., 1]
        self.desired_joint_position = self.dof_position.clone()
        self.computed_torque = torch.zeros_like(self.dof_position)
        self.applied_torque = torch.zeros_like(self.dof_position)

        # Rigid body
        self._rigid_body_state: torch.Tensor = gymtorch.wrap_tensor(_rigid_body_tensor).view(self.num_envs, -1, 13)
        
        # Root actors
        self._actors_root_state: torch.Tensor = gymtorch.wrap_tensor(_actor_root_state_tensor).view(-1, 13)
        self.robot_link_dict: Dict[str, int] = self.gym.get_asset_rigid_body_dict(self.asset_handles["robot"])
        hand_index: Union[int, List[int]] = self.robot_link_dict[EE_NAMES[self.hand]] if self.hand in [LEFT, RIGHT] else [self.robot_link_dict[EE_NAMES[hand]] for hand in [LEFT, RIGHT]]      # TODO(L/R): Account for bimanual hand
        
        # Should be fixed to use jacobian and mass matrix in bimanual setting
        # jacobian
        self.jacobian: torch.Tensor = gymtorch.wrap_tensor(_jacobian)
        self.j_eef: Dict[str, torch.Tensor] = dict()
        for arm, idx in zip([LEFT, RIGHT], hand_index):
            self.j_eef[arm] = self.jacobian[:, (idx - 1), :, :7]
        # # mass matrix
        # self.mm: torch.Tensor = gymtorch.wrap_tensor(_mm)
        # self.mm = self.mm[:, :(hand_index - 1), :(hand_index - 1)]

        # joint torques
        self.dof_torque: torch.Tensor = gymtorch.wrap_tensor(_dof_force_tensor).view(self.num_envs, self._dims.JointTorqueDim.value)
        self.net_cf: torch.Tensor = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)

        if self.cfg["env"]["hand_force_limit"]:
            _force_torque_sensor_data = self.gym.acquire_force_sensor_tensor(self.sim)
            self.force_torque_sensor_data = gymtorch.wrap_tensor(_force_torque_sensor_data)


        # frames history
        EE_handles_indices = list(self._ee_handles.values())
        object_indices = self.actor_indices["object"]
        # timestep 0 is current tensor
        curr_history_length = 0
        while curr_history_length < self._state_history_len:
            # add tensors to history list
            self._EE_frames_state_history.append(self._rigid_body_state[:, EE_handles_indices, :7])
            self._object_state_history.append(self._actors_root_state[object_indices])
            # update current history length
            curr_history_length += 1
        print(f'ee frames history shape: {self._EE_frames_state_history[0].shape}')
        
        self._observations_scale = SimpleNamespace(low=None, high=None)
        self._action_scale = SimpleNamespace(low=None, high=None)

        self._successes = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        if self.cfg["env"]["extract_successes"]:
            self.extract = False
            self._successes_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        self.max_torque = torch.zeros(self.num_envs,dtype=torch.float, device=self.device)
        self.mean_energy = torch.zeros(self.num_envs,dtype=torch.float, device=self.device)
        self.apogee = torch.zeros(self.num_envs,dtype=torch.float, device=self.device)
        self.max_speed = torch.zeros(self.num_envs,dtype=torch.float, device=self.device)
        self.dropped = torch.zeros(self.num_envs,dtype=torch.float, device=self.device)

    def __configure_mdp_spaces(self):
        """
        Configures the observations, action spaces.
        """
        # Action scale for the MDP
        # Note: This is order sensitive.

        # action space is residual pose of eef and gripper position
        self.minimum_residual_scale = self.cfg["env"]["adaptive_residual_scale"]["minimum"]
        initial_residual_scale = self.cfg["env"]["initial_residual_scale"]
        self.position_scale = initial_residual_scale[0]
        self.orientation_scale = initial_residual_scale[1]

        # residual_scale_low = [-self.position_scale]*3+[-self.orientation_scale]*3
        # residual_scale_high = [self.position_scale]*3+[self.orientation_scale]*3
        residual_scale_low = [-0.0]*3 # Dump
        residual_scale_high = [0.0]*3 # Dump

        # Action: Left 18 + Right 18

        if self.cfg["env"]["adaptive_residual_scale"]['activate'] == True:      
            self.position_scale = initial_residual_scale[:6]
            self.KP_scale = initial_residual_scale[6:12]
            self.KD_scale = initial_residual_scale[12:]
            position_residual_scale_low = [-self.position_scale[0], -self.position_scale[1], -self.position_scale[2], 
                                        -self.position_scale[3], -self.position_scale[4], -self.position_scale[5]]
            position_residual_scale_high = [self.position_scale[0], self.position_scale[1], self.position_scale[2], 
                                        self.position_scale[3], self.position_scale[4], self.position_scale[5]]
            p_gain_scale = [self.KP_scale[0], self.KP_scale[1], self.KP_scale[2],
                            self.KP_scale[3], self.KP_scale[4], self.KP_scale[5]]
            d_gain_scale = [self.KD_scale[0], self.KD_scale[1], self.KD_scale[2],
                            self.KD_scale[3], self.KD_scale[4], self.KD_scale[5]]
            p_gain_scale_low = (0.5*np.array(p_gain_scale)).tolist()
        else:
            position_residual_scale_low = [-0.30, -0.35, -0.45, -0.40, -0.30, -0.35]
            position_residual_scale_high = [0.30, 0.35, 0.45, 0.40, 0.30, 0.35]
            p_gain_scale = RobotProp.DEFAULT_P_GAIN.tolist()
            d_gain_scale = RobotProp.DEFAULT_D_GAIN.tolist()


        if self.cfg["env"]["controller"]=="position":
            self._action_scale.low = to_torch(
                position_residual_scale_low +[0.9, 1.2, 0.95, 0.9, 0.3, 0.35]+[0.0]*6+position_residual_scale_low +[0.9, 1.2, 0.95, 0.9, 0.3, 0.35]+[0.0]*6,
                device=self.device
            )
            self._action_scale.high = to_torch(
                position_residual_scale_high + p_gain_scale + d_gain_scale\
                +position_residual_scale_high + p_gain_scale + d_gain_scale,
                device=self.device)
            
        elif self.cfg["env"]["controller"]=="position_2":
            self._action_scale.low = to_torch(
                [-0.495, -0.726, -0.792, -0.704, -0.294, -0.2]+[0.9, 1.1, 0.8, 0.8, 0.358, 0.4]+[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\
                +[-0.495, -0.726, -0.792, -0.704, -0.294, -0.2]+[0.9, 1.1, 0.8, 0.8, 0.358, 0.4]+[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                device=self.device
            )
            self._action_scale.high = to_torch(
                [0.495, 0.726, 0.792, 0.704, 0.294, 0.2]+[1.8, 2.2, 1.6, 1.6, 0.862, 0.8] + [0.06, 0.08, 0.1, 0.1, 0.03, 0.03]\
                +[0.495, 0.726, 0.792, 0.704, 0.294, 0.2]+[1.8, 2.2, 1.6, 1.6, 0.862, 0.8] + [0.06, 0.08, 0.1, 0.1, 0.03, 0.03], 
                device=self.device)

        # Observations scale for the MDP
        # check if policy outputs normalized action [-1, 1] or not.
        if self.cfg['env']["normalize_action"]:
            obs_action_scale = SimpleNamespace(
                low=torch.full((self.action_dim,), -1, dtype=torch.float, device=self.device),
                high=torch.full((self.action_dim,), 1, dtype=torch.float, device=self.device)
            )
        else:
            obs_action_scale = self._action_scale
        # Note: This is order sensitive.
        # 14D joint pos 14D joint vel keypoint
        if self.cfg["env"]["keypoint"]["activate"]:
            self._observations_scale.low = torch.cat([
                self.franka_dof_lower_limits,
                -self.franka_dof_speed_scales,
                # 2D keypoint - deprecated
                # self._object_limits["2Dkeypoint"].low.repeat(8),
                # self._object_limits["2Dkeypoint"].low.repeat(8),
                self._object_limits["position"].low,
                self._object_limits["orientation"].low,
                self._ee_limits["ee_position"].low,
                self._ee_limits["ee_orientation"].low,
                self._ee_limits["ee_position"].low,
                self._ee_limits["ee_orientation"].low,
                obs_action_scale.low
            ])
            self._observations_scale.high = torch.cat([
                self.franka_dof_upper_limits,
                self.franka_dof_speed_scales,
                # 2D keypoint - deprecated
                # self._object_limits["2Dkeypoint"].high.repeat(8),
                # self._object_limits["2Dkeypoint"].high.repeat(8),
                self._object_limits["position"].high,
                self._object_limits["orientation"].high,
                self._ee_limits["ee_position"].high,
                self._ee_limits["ee_orientation"].high,
                 self._ee_limits["ee_position"].high,
                self._ee_limits["ee_orientation"].high,
                obs_action_scale.high
            ])
        else:
            # keypoints are always activated. 
            pass

        # State scale for the MDP
        obs_dim = sum(self.obs_spec.values())
        action_dim = sum(self.action_spec.values())
        # check that dimensions match
        # observations
        if self._observations_scale.low.shape[0] != obs_dim or self._observations_scale.high.shape[0] != obs_dim:
            msg = f"Observation scaling dimensions mismatch. " \
                  f"\tLow: {self._observations_scale.low.shape[0]}, " \
                  f"\tHigh: {self._observations_scale.high.shape[0]}, " \
                  f"\tExpected: {obs_dim}."
            raise AssertionError(msg)
        # actions
        if self._action_scale.low.shape[0] != action_dim or self._action_scale.high.shape[0] != action_dim:
            msg = f"Actions scaling dimensions mismatch. " \
                  f"\tLow: {self._action_scale.low.shape[0]}, " \
                  f"\tHigh: {self._action_scale.high.shape[0]}, " \
                  f"\tExpected: {action_dim}."
            raise AssertionError(msg)
    
    def update_residual_scale(self, position_ratio, kp_ratio, kd_ratio):

        # DEBUG: Should be fixed with bimanual setting
        import pdb
        pdb.set_trace()

        if self.position_scale[0] <= self.minimum_residual_scale[0]: return False

        self.position_scale = (np.array(self.position_scale) * position_ratio.T).tolist()
        self.KP_scale = (np.array(self.KP_scale) * kp_ratio.T).tolist()
        self.KD_scale = (np.array(self.KD_scale) * kd_ratio.T).tolist()

        if self.position_scale[0] <= self.minimum_residual_scale[0]:
            self.position_scale = self.minimum_residual_scale[:6]
            self.KP_scale = self.minimum_residual_scale[6:12]
            self.KD_scale = self.minimum_residual_scale[12:]

        residual_scale_low = [-self.position_scale[0]]*3
        residual_scale_high = [self.position_scale[0]]*3

        position_residual_scale_low = [-self.position_scale[0], -self.position_scale[1], -self.position_scale[2], 
                                       -self.position_scale[3], -self.position_scale[4], -self.position_scale[5]]
        position_residual_scale_high = self.position_scale
        p_gain_scale = self.KP_scale
        d_gain_scale = self.KD_scale
        p_gain_scale_low = (0.5*np.array(p_gain_scale)).tolist()

        print(f"Residual scale is reduced to {position_residual_scale_high}")
        print(f"KP scale is reduced to {p_gain_scale}")
        print(f"KD scale is reduced to {d_gain_scale}")

        if self.cfg["env"]["restrict_gripper"]:
            if self.cfg["env"]["controller"]=="JP":
                self._action_scale.low = to_torch(
                    residual_scale_low+[10.0]*7+[0.0]*7, 
                    device=self.device
                )  # TODO: add explanations to this values
                self._action_scale.high = to_torch(
                    residual_scale_high+[200.0]*4+[100.0]*3+[2.0]*7, 
                    device=self.device
                )
            elif self.cfg["env"]["controller"]=="position":
                self._action_scale.low = to_torch(
                    position_residual_scale_low +[0.9, 1.2, 0.95, 0.9, 0.3, 0.35]+[0.0]*6,
                    device=self.device
                )  # TODO: add explanations to this values
                self._action_scale.high = to_torch(
                    position_residual_scale_high + p_gain_scale + d_gain_scale, 
                    device=self.device
                )
            elif self.cfg["env"]["controller"]=="position_2":
                self._action_scale.low = to_torch(
                    position_residual_scale_low + p_gain_scale_low +[0.0]*6,
                    device=self.device
                )  # TODO: add explanations to this values
                self._action_scale.high = to_torch(
                    position_residual_scale_high + p_gain_scale + [0.03, 0.045, 0.45, 0.45, 0.01, 0.01], 
                    device=self.device
                )
                print(f"Actually it was kp*Residual: {position_residual_scale_high}")

        else:
            # plus gain
            self._action_scale.low = to_torch(residual_scale_low+[0.0]+[10.0]*6+[0.0]*6, device=self.device)
            self._action_scale.high = to_torch(residual_scale_high+[0.04]+[300.0]*6+[2.0]*6, device=self.device)
        
        return True

    def __create_scene_assets(self):
        """ Define Gym assets for robot and object.
        """
        # Define assets
        self.asset_handles["robot"] = self.__define_robot_asset()
        if self.cfg['env']['scene_randomization']['background']:
            self.asset_handles["floor"] = self.__define_floor_asset()
            self.asset_handles["back"] = self.__define_back_asset()
        self.asset_handles["table"] = self._define_table_asset()
        self.asset_handles["object"] = self._define_object_asset()
        self.asset_handles["goal_object"] = self.__define_goal_object_asset()
        
        # Display the properties (only for debugging)
        # Robot
        print("Franka Robot Asset: ")
        print(f'\t Number of bodies: {self.gym.get_asset_rigid_body_count(self.asset_handles["robot"])}')
        print(f'\t Number of shapes: {self.gym.get_asset_rigid_shape_count(self.asset_handles["robot"])}')
        print(f'\t Number of dofs: {self.gym.get_asset_dof_count(self.asset_handles["robot"])}')
        print(f'\t Number of actuated dofs: {self._dims.JointTorqueDim.value}')

    def __create_envs(self):
        """Create various instances for the environment.
        """
        robot_dof_props = self.gym.get_asset_dof_properties(self.asset_handles["robot"])

        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self.franka_dof_speed_scales = []
        self.franka_dof_effort_scales = []

        sysid_params = np.load('simulation/im2gym/sysid_result.npy').reshape(6, -1)

        # sysid_friction = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        # sysid_damping  = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        # sysid_armature = [0., 0., 0., 0., 0., 0.]
            
        for k, dof_index in enumerate(self._robot_dof_indices.values()):
            
            robot_dof_props['driveMode'][dof_index] = gymapi.DOF_MODE_EFFORT

            robot_dof_props['friction'][k]  = sysid_params[k%6, 0]
            robot_dof_props['damping'][k]   = sysid_params[k%6, 1]
            robot_dof_props['armature'][k]  = sysid_params[k%6, 2]

            self.franka_dof_lower_limits.append(robot_dof_props['lower'][k])
            self.franka_dof_upper_limits.append(robot_dof_props['upper'][k])
            self.franka_dof_speed_scales.append(robot_dof_props['velocity'][k])
            self.franka_dof_effort_scales.append(robot_dof_props['effort'][k])
        
        if self.cfg["env"]["restrict_gripper"]:
            _effort = 1000
        else:
            _effort = 200
        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self.franka_dof_speed_scales = to_torch(self.franka_dof_speed_scales, device=self.device)
        # self.franka_dof_speed_scales[[7, 8]] = 0.1
        # robot_dof_props['effort'][7] = _effort
        # robot_dof_props['effort'][8] = _effort
        self.franka_dof_effort_scales = to_torch(self.franka_dof_effort_scales, device=self.device)
        # self.franka_dof_effort_scales[[7, 8]] = _effort

        # define lower and upper region bound for each environment
        env_lower_bound = gymapi.Vec3(-self.cfg['env']["envSpacing"], -self.cfg['env']["envSpacing"], 0.0)
        env_upper_bound = gymapi.Vec3(self.cfg['env']["envSpacing"], self.cfg['env']["envSpacing"], self.cfg['env']["envSpacing"])
        num_instances_per_row = int(np.sqrt(self.num_envs))

        # mapping from name to gym actor indices
        # note: later the list is converted to torch tensor for ease in interfacing with IsaacGym.
        actor_indices: Dict[str, List[int]] = dict()
        for asset_name in self.asset_handles.keys():
            actor_indices[asset_name] = list()
        
        # count number of shapes and bodies
        max_agg_bodies = 0
        max_agg_shapes = 0
        for asset_handle in self.asset_handles.values():
            max_agg_bodies += self.gym.get_asset_rigid_body_count(asset_handle)
            max_agg_shapes += self.gym.get_asset_rigid_shape_count(asset_handle)

        if self.cfg['env']["enableCameraSensors"]:
            camera_position = torch.tensor(OmegaConf.to_object(self.cfg["env"]["camera"]["position"]))
            camera_angle = float(self.cfg["env"]["camera"]["angle"])
        
        self.envs = []
        # iterate and create environment instances
        for env_index in range(self.num_envs):
            # create environment
            env_ptr = self.gym.create_env(self.sim, env_lower_bound, env_upper_bound, num_instances_per_row)
            self.envs.append(env_ptr)

            # Begin aggregration
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Add robot to environment
            robot_pose = gymapi.Transform()
            robot_pose.p = gymapi.Vec3(*self.cfg['env']['robot']['p'])
            robot_pose.r = gymapi.Quat(*self.cfg['env']['robot']['r'])

            # param6: bitwise filter for elements in the same collisionGroup to mask off collision
            # originally 0 but 0 makes self-collision. positive value will mask out self collision
            Franka_actor = self.gym.create_actor(env_ptr, self.asset_handles["robot"], robot_pose, "robot", env_index, 0, 3)

            self.gym.set_actor_dof_properties(env_ptr, Franka_actor, robot_dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, Franka_actor)
            robot_idx = self.gym.get_actor_index(env_ptr, Franka_actor, gymapi.DOMAIN_SIM)
            actor_indices["robot"].append(robot_idx)

            # add table to environment
            self._create_table(env_ptr, env_index, actor_indices)

            # add object to environment
            self._create_object(env_ptr, env_index, actor_indices)

            if self.cfg['env']['scene_randomization']['background']:
                back_pose = gymapi.Transform()
                back_pose.p = gymapi.Vec3(-0.5, 0, 0.5)
                back_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                back_handle = self.gym.create_actor(env_ptr, self.asset_handles["back"], back_pose, "back", (env_index + self.num_envs * 2), 0, 4)
                back_idx = self.gym.get_actor_index(env_ptr, back_handle, gymapi.DOMAIN_SIM)
                self.gym.set_rigid_body_color(env_ptr, back_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 0, 0))
                actor_indices["back"].append(back_idx)
                
                floor_pose = gymapi.Transform()
                floor_pose.p = gymapi.Vec3(0, 0, 0.001)
                floor_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                floor_handle = self.gym.create_actor(env_ptr, self.asset_handles["floor"], floor_pose, "floor", (env_index + self.num_envs * 3), 0, 5)
                floor_idx = self.gym.get_actor_index(env_ptr, floor_handle, gymapi.DOMAIN_SIM)
                self.gym.set_rigid_body_color(env_ptr, floor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 0, 0))
                actor_indices["floor"].append(floor_idx)
            
            
            if self.cfg['env']["enableCameraSensors"]:
                # add camera sensor to environment
                camera_props = gymapi.CameraProperties()
                camera_props.enable_tensors = True
                camera_props.horizontal_fov = 79.3706 # 55.368
                camera_props.height = self.cfg["env"]["camera"]["size"][0]
                camera_props.width = self.cfg["env"]["camera"]["size"][1]
                if self.camera_randomize:
                    camera_props, camera_transform = self.camera_randomizer(camera_props, camera_position, camera_angle)
                else:
                    camera_transform = self._get_camera_transform(camera_position, camera_angle)
                camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
                self.gym.set_camera_transform(camera_handle, env_ptr, camera_transform)
                self.camera_handles.append(camera_handle)
                camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR)
                self._torch_camera_rgba_images.append(gymtorch.wrap_tensor(camera_rgba_tensor))
                if self.cfg['env']["camera"]["segmentation"]:
                    camera_segmentation_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_SEGMENTATION)
                    self._torch_camera_segmentation.append(gymtorch.wrap_tensor(camera_segmentation_tensor))
            
            # end aggregation
            self.gym.end_aggregate(env_ptr)
        
        # light source
        intensity = [0.2, 0.2, 0.1, 0.]
        ambient = [0.8, 0.9, .9, .0]
        direction = torch.tensor([[1.0, -0.05, 1.6], [2.4, 2.0, 3.0], [0.6, 0, 0.6]], device=self.rl_device)
        if self.cfg['env']['scene_randomization']['light']:
            intensity[:3] =+ (-0.005 + 0.01 * torch.rand(1, device=self.rl_device))
            ambient[:3] =+ (-0.005 + 0.01 * torch.rand(1, device=self.rl_device))
            direction += (-0.005 + 0.01 * torch.rand((3, 3), device=self.rl_device))
        self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(intensity[0],intensity[0],intensity[0]),\
                gymapi.Vec3(ambient[0], ambient[0], ambient[0]), gymapi.Vec3(*direction[0]))
        self.gym.set_light_parameters(self.sim, 1, gymapi.Vec3(intensity[1],intensity[1],intensity[1]),\
                gymapi.Vec3(ambient[1], ambient[1], ambient[1]), gymapi.Vec3(*direction[1]))
        self.gym.set_light_parameters(self.sim, 2, gymapi.Vec3(intensity[2],intensity[2],intensity[2]),\
                gymapi.Vec3(ambient[2], ambient[2], ambient[2]), gymapi.Vec3(*direction[2]))
        self.gym.set_light_parameters(self.sim, 3, gymapi.Vec3(intensity[3],intensity[3],intensity[3]),\
                gymapi.Vec3(ambient[3], ambient[3], ambient[3]), gymapi.Vec3(0., -0.1, 0.5))
        # convert gym actor indices from list to tensor
        self.actor_indices: Dict[str, torch.Tensor] = dict()
        for asset_name, indices in actor_indices.items():
            self.actor_indices[asset_name] = torch.tensor(indices, dtype=torch.long, device=self.device)

    def _create_table(self, env_ptr, env_index: int, actor_indices: Dict[str, List[int]]):
        
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(self.cfg['env']['geometry']['table']['x'], self.cfg['env']['geometry']['table']['y'], self.cfg['env']['geometry']['table']['z'])

        table_handle = self.gym.create_actor(env_ptr, self.asset_handles["table"], table_pose, "table", env_index, 0, 1)
        table_color = gymapi.Vec3(0.54, 0.57, 0.59)
        self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, table_color)
        table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
        actor_indices["table"].append(table_idx)

    def _create_object(self, env_ptr, env_index: int, actor_indices: Dict[str, List[int]]):
        object_pose = gymapi.Transform()
        object_pose.p = gymapi.Vec3(*self.cfg['env']['geometry']['init_goal']['T_O'][:3])
        object_pose.r = gymapi.Quat(*self.cfg['env']['geometry']['init_goal']['T_O'][3:])
        
        object_handle = self.gym.create_actor(env_ptr, self.asset_handles["object"], object_pose, "object", env_index, 0, 2)
        object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
        actor_indices["object"].append(object_idx)

    def _check_termination(self):
        failed_reset = self._check_failure()

        if self.cfg["env"]["adaptive_dof_pos_limit"]["activate"]:
            dof_pos_limit_reset = self._check_dof_position_limit_reset(self.dof_pos_offset)
            failed_reset = torch.logical_or(failed_reset, dof_pos_limit_reset)

        if self.cfg["env"]["adaptive_dof_vel_limit"]["activate"]:
            dof_vel_limit_reset = self._check_dof_velocity_limit_reset(self.dof_vel_offset)
            failed_reset = torch.logical_or(failed_reset, dof_vel_limit_reset)
        
        if self.cfg["env"]["hand_force_limit"]:
            force_limit_reset = torch.norm(self.force_torque_sensor_data[:,:3], p=2, dim=-1) >40.
            #force_limit_reset = torch.max(torch.abs(self.force_torque_sensor_data[:,:3]), dim=-1) >40.
            failed_reset = torch.logical_or(failed_reset, force_limit_reset)

        # Success reward
        goal_reached = self._check_success()
        self.rew_buf[goal_reached] += self.cfg["env"]["reward_terms"]["object_dist"]["weight2"]

        # Reset
        self.reset_buf = torch.logical_or(self.reset_buf, failed_reset)
        if self.cfg["env"]["extract_successes"] and self.extract:
            self._successes_count[goal_reached] += 1
        self.reset_buf = torch.logical_or(self.reset_buf, goal_reached)
        self._successes[goal_reached] = 1
        self.reset_buf = self.reset_buf.float()
    
    def add_dof_position_limit_offset(self, offset):
        if self.dof_pos_offset >= self.dof_pos_limit_maximum: return False
        self.dof_pos_offset += offset
        if self.dof_pos_offset >= self.dof_pos_limit_maximum: 
            self.dof_pos_offset = self.dof_pos_limit_maximum
            print("dof_pos_limit is maximum " +str(self.dof_pos_offset))
        else: 
            print("dof_pos_limit is increased to " +str(self.dof_pos_offset))
        return True

    def add_dof_velocity_limit_offset(self, offset):
        if self.dof_vel_offset >= self.dof_vel_limit_maximum: return False
        self.dof_vel_offset += offset
        if self.dof_vel_offset > self.dof_vel_limit_maximum: 
            self.dof_vel_offset = self.dof_vel_limit_maximum
        else: print("dof_vel_limit is increased to " +str(self.dof_vel_offset))
        return True

    def _check_dof_position_limit_reset(self, offset):

        dof_lower_safety_limit = ((1.0-offset)*self.franka_dof_lower_limits[:6] + offset*self.franka_dof_upper_limits[:6])
        dof_upper_safety_limit = (offset*self.franka_dof_lower_limits[:6] + (1.0-offset)*self.franka_dof_upper_limits[:6])
        dof_pos_low_envs = torch.any(torch.le(self.dof_position[:, :6], dof_lower_safety_limit[None, :]), dim=1)
        dof_pos_high_envs = torch.any(torch.gt(self.dof_position[:, :6], dof_upper_safety_limit[None, :]), dim=1)

        dof_pos_limit_exceeded_envs = torch.logical_or(dof_pos_low_envs, dof_pos_high_envs)
        return dof_pos_limit_exceeded_envs

    def _check_dof_velocity_limit_reset(self, offset):
        dof_vel_upper_safety_limit = (1.0-offset)*self.franka_dof_speed_scales[:6]
        dof_vel_high_envs = torch.any(torch.gt(torch.abs(self.dof_velocity[:, :6]), dof_vel_upper_safety_limit[None, :]), dim=1)
        return dof_vel_high_envs

    def _check_failure(self) -> torch.Tensor:
        # TODO: to be fixed
        # failed_envs = torch.le(1.0, torch.norm(self._object_state_history[0][:, :3], dim=1))
        failed_envs = torch.zeros_like(self._object_state_history[0][:, 0])
        return failed_envs

    def _check_success(self) -> torch.Tensor:
        delta = self._object_state_history[0][:, 0:3] - self._object_goal_poses_buf[:, 0:3]
        dist = torch.norm(delta, p=2, dim=-1)
        goal_position_reached = torch.le(dist, self.cfg["env"]["reward_terms"]["object_dist"]["th"])
        quat_a = self._object_state_history[0][:, 3:7]
        quat_b = self._object_goal_poses_buf[:, 3:7]
        angles = quat_diff_rad(quat_a, quat_b)
        goal_rotation_reached = torch.le(torch.abs(angles), self.cfg["env"]["reward_terms"]["object_rot"]["th"])
        goal_reached = torch.logical_and(goal_rotation_reached, goal_position_reached)
        return goal_reached


    def reset_idx(self, env_ids: torch.Tensor):
        # randomization can happen only at reset time, since it can reset actor positions on GPU

        if self.env_randomize:
            self.env_randomizer(self.randomization_params, env_ids)

        # A) Reset episode stats buffers
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.randomize_buf[env_ids] = 0
        self._successes[env_ids] = 0
        self.previous_smoothed_action[env_ids] = 0


        # B) Various randomizations at the start of the episode:
        # -- Robot base position.
        # -- Stage position.
        # -- Coefficient of restituion and friction for robot, object, stage.
        # -- Mass and size of the object
        # -- Mass of robot links
        
        object_initial_state_config = self.cfg["env"]["reset_distribution"]["object_initial_state"]
        # If joint training
        # we must save which data is used to reset environment
        if object_initial_state_config["type"] == "pre_contact_policy":
            indices = torch.arange(
                self._init_data_use_count, (self._init_data_use_count + env_ids.shape[0]), dtype=torch.long, device=self.device
            )
            indices %= self._robot_buffer.shape[0]
            self._init_data_use_count += env_ids.shape[0]
            self._init_data_indices[env_ids] = indices

        self._sample_table_poses(env_ids)
        # -- Sampling of initial pose of the object
        self._sample_object_poses(env_ids)

        # -- Sampling of goal pose of the object
        # self._sample_object_goal_poses(env_ids)

        # -- Robot joint state
        self._sample_robot_state(env_ids)


        # C) Extract franka indices to reset
        robot_indices = self.actor_indices["robot"][env_ids].to(torch.int32)
        object_indices = self.actor_indices["object"][env_ids].to(torch.int32)
        table_indices = self.actor_indices["table"][env_ids].to(torch.int32)

        all_indices = torch.unique(torch.cat([robot_indices, object_indices, table_indices]))

        # D) Set values into simulator
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(robot_indices), len(robot_indices)
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._actors_root_state), gymtorch.unwrap_tensor(all_indices), len(all_indices)
        )

        # TODO(dlee): add a code that grasps the object with two hands -> it is impossible to simulate only reset_idx envs. 
        # we should use different way to enforce it such as ignoring the initial 10 timestep action values
        if self.cfg['task']['throw_only']:
            self._hold_object()
        
    
    """
    Helper functions - define assets
    """

    def __define_robot_asset(self):
        """ Define Gym asset for robot.
        """
        # define Franka asset
        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options.flip_visual_attachments = False
        robot_asset_options.fix_base_link = True
        robot_asset_options.collapse_fixed_joints = False
        robot_asset_options.disable_gravity = True
        robot_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        robot_asset_options.thickness = 0.001
        robot_asset_options.convex_decomposition_from_submeshes = True
        # robot_asset_options.vhacd_enabled = True
        # robot_asset_options.vhacd_params = gymapi.VhacdParams()
        # robot_asset_options.vhacd_params.resolution = 1000000

        robot_asset = self.gym.load_asset(self.sim, self._assets_dir, "urdf/RobotBimanualV5/urdf/Simplify_Robot_plate_gripper.urdf", robot_asset_options)

        robot_props = self.gym.get_asset_rigid_shape_properties(robot_asset)
        self.EE_handle: List[int] = [self.gym.find_asset_rigid_body_index(robot_asset, EE_LINK[hand]) for hand in [LEFT, RIGHT]]

        cnt=0
        ee_start = [(self.gym.get_asset_rigid_body_shape_indices(robot_asset)[ee_handle].start) for ee_handle in self.EE_handle]
        ee_count = [(self.gym.get_asset_rigid_body_shape_indices(robot_asset)[ee_handle].count) for ee_handle in self.EE_handle]

        for i, p in enumerate(robot_props):
            if ee_start[0] <= i and i < ee_start[0]+ee_count[0]:
                p.friction = 2.0
            elif ee_start[1] <= i and i < ee_start[1]+ee_count[1]:
                p.friction = 2.0
            else:
                p.friction = 0.2
            # p.restitution = 0.2
        self.gym.set_asset_rigid_shape_properties(robot_asset, robot_props)
        after_update_props = self.gym.get_asset_rigid_shape_properties(robot_asset)
        

        for frame_name in self._ee_handles.keys():
            self._ee_handles[frame_name] = self.gym.find_asset_rigid_body_index(robot_asset, frame_name)
            # check valid handle
            if self._ee_handles[frame_name] == gymapi.INVALID_HANDLE:
                msg = f"Invalid handle received for frame: `{frame_name}`."
                print(msg)
        
        self._tool_handle: Union[int, List[int]] = [self.gym.find_asset_rigid_body_index(robot_asset, EE_NAMES[hand]) for hand in [LEFT, RIGHT]]

        for dof_name in self._robot_dof_indices.keys():
            self._robot_dof_indices[dof_name] = self.gym.find_asset_dof_index(robot_asset, dof_name)
            # check valid handle
            if self._robot_dof_indices[dof_name] == gymapi.INVALID_HANDLE:
                msg = f"Invalid index received for DOF: `{dof_name}`."
                print(msg)
        # return the asset
        return robot_asset    

    def __define_floor_asset(self):
        """ Define Gym asset for a floor.
        """

        floor_asset_options = gymapi.AssetOptions()
        floor_asset_options.disable_gravity = True
        floor_asset_options.fix_base_link = True
        floor_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        floor_asset_options.thickness = 0.001

        floor_asset = self.gym.create_box(self.sim, 2, 5, 0.002, floor_asset_options)

        return floor_asset

    def __define_back_asset(self):
        """ Define Gym asset for the background.
        """

        back_asset_options = gymapi.AssetOptions()
        back_asset_options.disable_gravity = True
        back_asset_options.fix_base_link = True
        back_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        back_asset_options.thickness = 0.001

        back_asset = self.gym.create_box(self.sim, 0.002, 2.5, 1, back_asset_options)

        return back_asset
    
    def _define_table_asset(self):
        """ Define Gym asset for table.
        """
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        table_asset_options.thickness = 0.001
        table_dims = self.cfg["env"]["geometry"]["table"]
        table_asset = self.gym.create_box(self.sim, table_dims['width'], table_dims['length'], table_dims['height'],  table_asset_options)
        table_props = self.gym.get_asset_rigid_shape_properties(table_asset)
        for p in table_props:
            p.friction = 0.9
        self.gym.set_asset_rigid_shape_properties(table_asset, table_props)
        self.asset_handles["table"] = table_asset
        return table_asset

    def _define_object_asset(self):
        """ Define Gym asset for object.
        """
        # define object asset
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = False
        object_asset_options.thickness = 0.001
        object_asset_options.density = self.cfg["env"]["geometry"]["object"]["density"]
        object_dims = self.cfg["env"]["geometry"]["object"]
        # object_asset = self.gym.load_asset(self.sim, self._assets_dir, 'urdf/Cube/im2_cube.urdf', object_asset_options)
        object_asset = self.gym.create_box(self.sim, object_dims['width'], object_dims['length'], object_dims['height'],  object_asset_options)
        object_props = self.gym.get_asset_rigid_shape_properties(object_asset)
        # sensor_pose = gymapi.Transform()
        # _=self.gym.create_asset_force_sensor(object_asset, 0, sensor_pose)
        for p in object_props:
            p.friction = 0.6
            # p.restitution = 0.2
        self.gym.set_asset_rigid_shape_properties(object_asset, object_props)

        return object_asset
    def __define_goal_object_asset(self):
        """ Define Gym asset for goal object.
        """
        if self.cfg['env']["enableCameraSensors"]: return None # Goal object should not be visible while training student policies
        # define object asset
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = True
        object_asset_options.fix_base_link = True
        object_asset_options.thickness = 0.001
        object_asset_options.flip_visual_attachments = True

        size = self._object_dims.size
        goal_object_asset = self.gym.create_box(self.sim, size[0], size[1], size[2], object_asset_options)
        
        return goal_object_asset

    """
    Helper functions - MDP
    """
    def compute_observations(self):
        """
        Fills observation and state buffer with the current state of the system.
        """
        self.mean_energy = 1/self.control_freq_inv * self.mean_energy
        # Extract frame handles
        EE_indices = list(self._ee_handles.values())
        object_indices = self.actor_indices["object"]
        # Update state histories
        self._EE_frames_state_history.appendleft(self._rigid_body_state[:, EE_indices, :7])
        self._object_state_history.appendleft(self._actors_root_state[object_indices])

        # fill the observations and states buffer
        self.__compute_teacher_observations()
        if self.cfg["env"]["enableCameraSensors"]:
            self.__compute_camera_observations()
        
        # normalize observations if flag is enabled
        if self.cfg['env']["normalize_obs"]:
            # for normal obs
            self.obs_buf = scale_transform(
                self.obs_buf,
                lower=self._observations_scale.low,
                upper=self._observations_scale.high
            )

        if self.record:
            self.record_data['obs'].append(self.obs_buf.detach().cpu().numpy())
            obs = unscale_transform(self.obs_buf, lower=self._observations_scale.low, upper=self._observations_scale.high)
            self.record_data['joint_pos'].append(obs.detach().cpu().numpy()[:,:self._dims.GeneralizedCoordinatesDim.value])


    def __compute_teacher_observations(self):

        # generalized coordinates
        start_offset = 0
        end_offset = start_offset + self._dims.GeneralizedCoordinatesDim.value # bimanual
        self.obs_buf[:, start_offset:end_offset] = self.dof_position

        # generalized velocities
        start_offset = end_offset
        end_offset = start_offset + self._dims.GeneralizedVelocityDim.value # bimanual
        self.obs_buf[:, start_offset:end_offset] = self.dof_velocity

        # 2D keypoint - deprecated
         
        # # 2D projected object keypoints
        # start_offset = end_offset
        # end_offset = start_offset + self._dims.TwoDimensionKeypointDim.value * self.keypoints_num
        # current_keypoints = gen_keypoints(pose=self._object_state_history[0][:, 0:7], size=self._object_dims.size)
        # self.obs_buf[:, start_offset:end_offset] = compute_projected_points(self.translation_from_camera_to_object, current_keypoints, self.camera_matrix, self.device).reshape(self.num_envs, 16)[:]

        # # 2D projected goal keypoints
        # start_offset = end_offset
        # end_offset = start_offset + self._dims.TwoDimensionKeypointDim.value * self.keypoints_num
        # goal_keypoints = gen_keypoints(pose=self._object_goal_poses_buf, size=self._object_dims.size)
        # self.obs_buf[:, start_offset:end_offset] = compute_projected_points(self.translation_from_camera_to_object, goal_keypoints, self.camera_matrix, self.device).reshape(self.num_envs, 16)[:]

        # object pose
        start_offset = end_offset
        end_offset = start_offset + self._dims.ObjectPoseDim.value
        obj_poses = self._object_state_history[0][:, 0:7].clone()

        mask = obj_poses[:, -1] < 0
        obj_poses[mask, 3:7] = -obj_poses[mask, 3:7]
        self.obs_buf[:, start_offset:end_offset] = obj_poses

        # hand pose: LEFT
        start_offset = end_offset
        end_offset = start_offset + self._dims.ObjectPoseDim.value
        ee_poses = self._rigid_body_state[:, self._tool_handle[0], :7].clone()

        mask = ee_poses[:, -1] < 0
        ee_poses[mask, 3:7] = -ee_poses[mask, 3:7]
        self.obs_buf[:, start_offset:end_offset] = ee_poses

        # hand pose: RIGHT
        start_offset = end_offset
        end_offset = start_offset + self._dims.ObjectPoseDim.value
        ee_poses = self._rigid_body_state[:, self._tool_handle[1], :7].clone()

        mask = ee_poses[:, -1] < 0
        ee_poses[mask, 3:7] = -ee_poses[mask, 3:7]
        self.obs_buf[:, start_offset:end_offset] = ee_poses

        # previous action
        start_offset = end_offset
        end_offset = start_offset + self.action_dim 
        self.obs_buf[:, start_offset:end_offset] = self.actions

    def __compute_camera_observations(self):
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        for i in range(self.num_envs):
            if self.cfg['env']["camera"]["segmentation"]:
                segmentation = self._torch_camera_segmentation[i].unsqueeze(-1)
                img = self._torch_camera_rgba_images[i][..., :3]
                # background and floor are maksed as 4 and 5 respectively. Mask them out.
                img[segmentation.repeat((1, 1, 3)) > 3] = 0
                self._segmentation[i] = segmentation
                self._camera_image[i, ..., :3] = img
            else:
                self._camera_image[i, ..., :3] = self._torch_camera_rgba_images[i][..., :3]
        self.gym.end_access_image_tensors(self.sim)
        
    def convert_linear_to_RGB(self, linear: np.array):
            return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * (pow(linear, (1.0 / 2.4))) - 0.055)

    def _sample_robot_state(self, reset_env_indices: torch.Tensor):
        """Samples the robot DOF state based on the settings.

        Type of robot initial state distribution: ["default", "random"]
             - "default" means that robot is in default configuration.
             - "pre_contact_policy" means that robot pose is made by pi_pre

        Args:
            reset_env_indices: A tensor contraining indices of environments to reset.
            distribution: Name of distribution to sample initial state from: ['default', 'random']
        """
        self.dof_position[reset_env_indices] = self._robot_buffer[self._init_data_indices[reset_env_indices]]

        self.dof_velocity[reset_env_indices] = torch.zeros(
            (reset_env_indices.shape[0], self._dims.GeneralizedVelocityDim.value), device=self.device
        )
        # reset robot grippers state history
        for idx in range(1, self._state_history_len):
            self._EE_frames_state_history[idx][reset_env_indices] = 0.0

    def _sample_table_poses(self, reset_env_indices: torch.Tensor):
        """Sample poses for the table.

        Args:
            reset_env_indices: A tensor contraining indices of environment instances to reset.
        """
        table_params = self.cfg["env"]["geometry"]["table"]
        table_pose = [table_params["x"], table_params["y"], table_params["z"]]

        table_indices = self.actor_indices["table"][reset_env_indices]

        table_poses = torch.zeros_like(self._actors_root_state[table_indices], device=self.device)
        table_poses[:, :3] = torch.tensor(table_pose, device=self.device)[None, ...]

        # root actor buffer
        self._actors_root_state[table_indices] = table_poses
        
    
    def _sample_object_poses(self, reset_env_indices: torch.Tensor):
        """Sample poses for the object.

        Args:
            reset_env_indices: A tensor contraining indices of environment instances to reset.
        """
        object_indices = self.actor_indices["object"][reset_env_indices]
        self._object_state_history[0][reset_env_indices, :7] = self._initial_object_pose_buffer[self._init_data_indices[reset_env_indices]]
        self._object_state_history[0][reset_env_indices, 7:13] = 0

        # reset object state history
        for idx in range(1, self._state_history_len):
            self._object_state_history[idx][reset_env_indices] = 0.0
        # root actor buffer
        self._actors_root_state[object_indices] = self._object_state_history[0][reset_env_indices]

    def _sample_object_goal_poses(self, reset_env_indices: torch.Tensor):
        """Sample goal poses for the object and sets them into the desired goal pose buffer.

        Args:
            reset_env_indices: A tensor contraining indices of environments to reset.
        """

        self._object_goal_poses_buf[reset_env_indices] = self._goal_buffer[self._init_data_indices[reset_env_indices]]

        if not self.cfg['env']["enableCameraSensors"]:
            goal_object_indices = self.actor_indices["goal_object"][reset_env_indices]
            self._actors_root_state[goal_object_indices, 0:7] = self._object_goal_poses_buf[reset_env_indices]

    def _hold_object(self):
        '''Run simulate few steps to hold the object with both hands
        '''
        hold_step = 50

        # Dummy action
        self.actions = torch.zeros((self.num_envs, self.action_dim)).to(self.rl_device)
        self.compute_observations()
        observation = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        state = torch.zeros((self.num_envs,)).to(self.rl_device)

        if self.cfg['env']["normalize_action"]:
            action_transformed = unscale_transform(
                self.actions,
                lower=self._action_scale.low,
                upper=self._action_scale.high
            )
        else:
            action_transformed = self.actions
        self.action_transformed = action_transformed

        for i in range(hold_step):
            actions, _, _, _, _ = self.hold_policy.step(observation, state)
            obs_dict, _, _, _ = self._hold_step(actions)
            observation = obs_dict['obs']

        return True
    
    def _hold_step(self, actions: torch.Tensor):
        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self._hold_pre_physics_step(action_tensor)
       
        # step physics and render each frame
        for i in range(self.control_freq_inv):
            self.step_controller()
            self.gym.simulate(self.sim)
            self.refresh_buffer()
        self.render()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras
    
    def _hold_pre_physics_step(self, actions):
        """
        Setting of input actions into simulator before performing the physics simulation step.
        """
        
        # Light randomization
        if self.cfg['env']['scene_randomization']['light']:
            self.randomize_light()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.env_ids = env_ids
        self.actions = actions.clone().to(self.device)
        self.actions[env_ids, :] = 0.0
        
        # If normalized_action is true, then denormalize them.
        if self.cfg['env']["normalize_action"]:
            action_transformed = unscale_transform(
                self.actions,
                lower=self._action_scale.low,
                upper=self._action_scale.high
            )
        else:
            action_transformed = self.actions
        self.action_transformed = action_transformed

        # Compute target ee pose or joint pose
        self.compute_target()


    def apply_dof_torque_indexed(self, env_indices: torch.Tensor):
        
        dummy_action = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        dummy_action[:, L_JOINT_P_START:L_JOINT_P_START+DOF] = self._action_scale.high[L_JOINT_P_START:L_JOINT_P_START+DOF]
        dummy_action[:, L_JOINT_D_START:L_JOINT_D_START+DOF] = self._action_scale.high[L_JOINT_D_START:L_JOINT_D_START+DOF]
        dummy_action[:, R_JOINT_P_START:R_JOINT_P_START+DOF] = self._action_scale.high[R_JOINT_P_START:R_JOINT_P_START+DOF]
        dummy_action[:, R_JOINT_D_START:R_JOINT_D_START+DOF] = self._action_scale.high[R_JOINT_D_START:R_JOINT_D_START+DOF]

        ct = self.robot_control.compute_torque([LEFT, RIGHT],
                                               self.desired_joint_position,
                                               self.dof_position,
                                               self.dof_velocity,
                                               dummy_action)

        self.computed_torque[env_indices, :DOF*2] = ct[env_indices, :]

        applied_torque = saturate(self.computed_torque, 
                                  lower=-self.franka_dof_effort_scales,
                                  upper=self.franka_dof_effort_scales)
        self.max_torque = torch.maximum(self.max_torque, torch.norm(applied_torque[:,:DOF*2],dim=-1))
        self.applied_torque[env_indices, :DOF*2] = applied_torque[env_indices, :]

        # Extract actor indices
        robot_indices = self.actor_indices["robot"][env_indices].to(torch.int32)

        # Set computed torques to simulator buffer.
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim, 
                                                        gymtorch.unwrap_tensor(self.applied_torque[env_indices, :]),
                                                        gymtorch.unwrap_tensor(robot_indices),
                                                        len(robot_indices))


    def from_left_to_right(self, left_q: torch.Tensor):
        
        right_q = left_q.clone()
        right_q *= -1

        return right_q
    
    def get_initial_joint_position(self, num_samples: int) -> torch.Tensor:
        l_init_q = torch.tensor(self.cfg['env']['geometry']['init_q'], device=self.device)
        r_init_q = self.from_left_to_right(l_init_q)
        bi_init_q = torch.cat((l_init_q, r_init_q))
        q_R = bi_init_q.repeat(num_samples, 1)

        return q_R

    def get_hold_joint_position(self, num_samples: int) -> torch.Tensor:
        l_init_q = torch.tensor(self.cfg['env']['geometry']['hold_q'], device=self.device)
        r_init_q = self.from_left_to_right(l_init_q)
        bi_init_q = torch.cat((l_init_q, r_init_q))
        q_R = bi_init_q.repeat(num_samples, 1)

        return q_R
    
    def read_dof_state(self):
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state: torch.Tensor = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self.dof_position = self.dof_state[..., 0]
        self.dof_velocity = self.dof_state[..., 1]


    def push_data(self, initial_object_pose: torch.Tensor, goal_object_pose: torch.Tensor, initial_joint_position: torch.Tensor):
        """
            Fill the object pose, goal pose, and robot configuration made by pi_pre and sampler
        """
        if initial_object_pose.dim() == 3:
            initial_object_pose = initial_object_pose.squeeze(1)
            goal_object_pose = goal_object_pose.squeeze(1)
            initial_joint_position = initial_joint_position.squeeze(1)
        self._robot_buffer = initial_joint_position
        self._initial_object_pose_buffer = initial_object_pose
        self._goal_buffer = goal_object_pose
        self._init_data_use_count = 0
        self._init_data_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
    def axisaToquat(self, axisA: torch.Tensor, the_opposite=False):
        if the_opposite==False:
            num_rotations = axisA.shape[0]
            angle = torch.norm(axisA, dim=-1)
            small_angle = (angle <= 1e-3)
            large_angle = ~small_angle

            scale = torch.empty((num_rotations,), device=self.device, dtype=torch.float)
            scale[small_angle] = (0.5 - angle[small_angle] ** 2 / 48 +
                                angle[small_angle] ** 4 / 3840)
            scale[large_angle] = (torch.sin(angle[large_angle] / 2) /
                                angle[large_angle])
            quat = torch.empty((num_rotations, 4), device=self.device, dtype=torch.float)
            quat[:,:3] = scale[:, None] * axisA
            quat[:,-1] = torch.cos(angle/2)
            return quat
        
        elif the_opposite==True:
            return axis_angle_from_quat(axisA)
    
    def compute_camera_intrinsics_matrix(self, image_width, image_heigth, horizontal_fov, device):
        vertical_fov = (image_heigth / image_width * horizontal_fov) * np.pi / 180
        horizontal_fov *= np.pi / 180

        f_x = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
        f_y = (image_heigth / 2.0) / np.tan(vertical_fov / 2.0)

        K = torch.tensor([[f_x, 0.0, image_width / 2.0], [0.0, f_y, image_heigth / 2.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)

        return K
    

    def randomize_light(self):
        intensity = torch.rand((4, 3), device=self.rl_device) * 0.01 + 0.2
        ambient = torch.rand((4, 3), device=self.rl_device) * 0.01 + 0.8
        intensity[1, :] = 0.2 + (-0.005 + 0.01 * torch.rand(1, device=self.rl_device))
        ambient[1, :] = 0.9 + (-0.005 + 0.01 * torch.rand(1, device=self.rl_device))
        intensity[2, :] = 0.1 + (-0.005 + 0.01 * torch.rand(1, device=self.rl_device))
        ambient[2, :] = 0.9 + (-0.005 + 0.01 * torch.rand(1, device=self.rl_device))
        intensity[3:] = 0.0
        ambient[3:] = 0.0
        direction = torch.tensor([[1.0,-0.05,1.6],[2.4,2.0,3.0],[0.6,0,0.6]], device=self.rl_device)+(-0.005+0.01*torch.rand((3,3), device=self.rl_device))
        self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(intensity[0,0],intensity[0,0],intensity[0,0]),\
                gymapi.Vec3(ambient[0,0],ambient[0,0],ambient[0,0]),gymapi.Vec3(*direction[0]) )
        self.gym.set_light_parameters(self.sim, 1, gymapi.Vec3(intensity[1,0],intensity[1,0],intensity[1,0]),\
                gymapi.Vec3(ambient[1,0],ambient[1,0],ambient[1,0]),gymapi.Vec3(*direction[1]) )
        self.gym.set_light_parameters(self.sim, 2, gymapi.Vec3(intensity[2,0],intensity[2,0],intensity[2,0]),\
                gymapi.Vec3(ambient[2,0],ambient[2,0],ambient[2,0]),gymapi.Vec3(*direction[2]) )
        self.gym.set_light_parameters(self.sim, 3, gymapi.Vec3(intensity[3,0],intensity[3,0],intensity[3,0]),\
                gymapi.Vec3(ambient[3,0],ambient[3,0],ambient[3,0]),gymapi.Vec3(0.,-0.1,0.5) )

    @property
    def camera_image(self) -> torch.Tensor:
        image = self._camera_image.detach().clone()
        if self.image_randomize:
            randomized_image: torch.Tensor = self.image_randomizer(self.randomization_params, image.permute((0, 3, 1, 2)), self.segmentation)
            return randomized_image.permute((0, 2, 3, 1))
        else:
            return image

    @property
    def segmentation(self) -> torch.Tensor:
        return self._segmentation.detach().clone()

    @property
    def env_steps_count(self) -> int:
        """Returns the total number of environment steps aggregated across parallel environments."""
        return self.gym.get_frame_count(self.sim) * self.num_envs
    
    @property
    def env_succeed(self) -> torch.Tensor:
        """Returns the succeded infromation of each environment."""
        return self._successes.detach()
    
    @property
    def env_pointing_indices(self) -> torch.Tensor:
        """Returns the indices of data used to reset environments."""
        return self._init_data_indices.detach()
    
    @property
    def env_succeed_count(self) -> torch.Tensor:
        """Returns the total number of environment succeded aggregated across parallel environments."""
        return self._successes_count.detach()

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6):
    """Convert tensor of quaternions to tensor of axis-angles."""

    mag = torch.linalg.norm(quat[:, 0:3], dim=1)
    half_angle = torch.atan2(mag, quat[:, 3])
    angle = 2.0 * half_angle
    sin_half_angle_over_angle = torch.where(torch.abs(angle) > eps,
                                            torch.sin(half_angle) / angle,
                                            1 / 2 - angle ** 2.0 / 48)
    axis_angle = quat[:, 0:3] / sin_half_angle_over_angle.unsqueeze(-1)

    return axis_angle

@torch.jit.script
def orientation_error(desired: torch.Tensor, current: torch.Tensor, version: int = 2):
    if version==1:
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)
    elif version==2:
        quat_norm = quat_mul(current, quat_conjugate(current))[:, 3]  # scalar component
        quat_inv = quat_conjugate(current) / quat_norm.unsqueeze(-1)
        quat_error = quat_mul(desired, quat_inv)

        # Convert to axis-angle error
        axis_angle_error = axis_angle_from_quat(quat_error)
        return axis_angle_error
    else:
        return -(current[:,:3]*desired[:,-1:]-desired[:,:3]*current[:,-1:]+\
            torch.cross(desired[:,:3], current[:,:3],-1))

    
@torch.jit.script
def gen_keypoints(pose: torch.Tensor, num_keypoints: int = 8, size: Tuple[float, float, float] = (0.065, 0.065, 0.065)):
    num_envs = pose.shape[0]
    keypoints_buf = torch.ones(num_envs, num_keypoints, 3, dtype=torch.float32, device=pose.device)
    for i in range(num_keypoints):
        # which dimensions to negate
        n = [((i >> k) & 1) == 0 for k in range(3)]
        corner_loc = [(1 if n[k] else -1) * s / 2 for k, s in enumerate(size)],
        corner = torch.tensor(corner_loc, dtype=torch.float32, device=pose.device) * keypoints_buf[:, i, :]
        keypoints_buf[:, i, :] = local_to_world_space(corner, pose)
    return keypoints_buf

@torch.jit.script
def compute_card_reward(
    obs_buf: torch.Tensor,
    reset_buf: torch.Tensor,
    progress_buf: torch.Tensor,
    episode_length: int,
    object_dist_weight: Tuple[float, float, float],
    inductive_weight: float,
    epsilon: Tuple[float, float],
    object_goal_poses_buf: torch.Tensor,
    max_torque: torch.Tensor,
    mean_energy: torch.Tensor,
    object_state: torch.Tensor,
    ee_state: torch.Tensor,
    size: Tuple[float, float, float],
    use_inductive_reward: bool,
    use_energy_reward: bool,
    energy_weight: float,
    position_only: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # distance from each finger to the centroid of the object, shape (N, 3).
    ee_state = torch.mean(ee_state, 1)
    curr_norms = torch.norm(ee_state[:, 0:3] - object_state[:, 0:3], p=2, dim=-1)
    
    last_action = torch.norm(obs_buf[:, -18:-12], 2, -1)
    residual = torch.norm(obs_buf[:, :7], 2, -1)

    # Reward for object distance
    if not position_only:
        object_keypoints = gen_keypoints(pose=object_state[:, 0:7], size=size)
        goal_keypoints = gen_keypoints(pose=object_goal_poses_buf[:, 0:7], size=size)
        delta = object_keypoints - goal_keypoints
        dist = torch.norm(delta, p=2, dim=-1)
        object_dist_reward = torch.sum((object_dist_weight[0] / (dist + epsilon[0])), -1)
        obj_reward = object_dist_weight[2]*object_dist_reward
    else:
        delta_pos = object_state[:, :3] - object_goal_poses_buf[:, :3]
        dist_pos = torch.norm(delta_pos, p=2, dim=-1) 
        object_dist_reward_2 = (object_dist_weight[0] / (dist_pos + epsilon[0]))
        obj_reward = object_dist_weight[2]*object_dist_reward_2

    energy_reward = torch.zeros_like(obj_reward)
    if use_energy_reward:
        energy_reward = energy_weight * mean_energy

    total_reward = obj_reward  # - 1. * last_action #- 0.5 * residual
    

    inductive_reward = torch.zeros_like(total_reward)
    if use_inductive_reward:
        inductive_reward = inductive_weight / (curr_norms + epsilon[0])
    total_reward += inductive_reward

    # reset agents
    reset = torch.zeros_like(reset_buf)
    reset = torch.where((progress_buf >= episode_length - 1), torch.ones_like(reset_buf), reset)

    return total_reward, obj_reward, inductive_reward, energy_reward, reset, last_action

@torch.jit.script
def control_osc(num_envs: int, j_eef, hand_vel, mm, dpose, dof_vel, dof_pos, kp, damping_ratio, variable: bool, decouple: bool, device: str):
    null=False
    kd = 2.0*torch.sqrt(kp)*damping_ratio if variable else 2.0 *math.sqrt(kp)
    kp_null = 10.
    kd_null = 2.0 *math.sqrt(kp_null)
    mm_inv = torch.inverse(mm)
    error = (kp * dpose - kd * hand_vel).unsqueeze(-1)

    if decouple:
        m_eef_pos_inv = j_eef[:, :3, :] @ mm_inv @ torch.transpose(j_eef[:, :3, :], 1, 2)
        m_eef_ori_inv = j_eef[:, 3:, :] @ mm_inv @ torch.transpose(j_eef[:, 3:, :], 1, 2)
        m_eef_pos = torch.inverse(m_eef_pos_inv)
        m_eef_ori = torch.inverse(m_eef_ori_inv)
        wrench_pos = m_eef_pos @ error[:, :3, :]
        wrench_ori = m_eef_ori @ error[:, 3:, :]
        wrench = torch.cat([wrench_pos, wrench_ori], dim=1)
    else:
        m_eef_inv = j_eef@ mm_inv @ torch.transpose(j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)
        wrench = m_eef @ error

    u = torch.transpose(j_eef, 1, 2) @ wrench
    
    return u.squeeze(-1)

@torch.jit.script
def compute_projected_points(T_matrix: torch.Tensor, keypoints: torch.Tensor, camera_matrix: torch.Tensor, device: str, num_points: int=8):
    num_envs=keypoints.shape[0]
    p_CO=torch.matmul(T_matrix, torch.cat([keypoints,torch.ones((num_envs, num_points,1),device=device)],-1).transpose(1,2))
    image_coordinates=torch.matmul(camera_matrix, p_CO).transpose(1,2)
    mapped_coordinates=image_coordinates[:,:,:2]/(image_coordinates[:,:,2].unsqueeze(-1))
    return mapped_coordinates

@torch.jit.script
def compute_friction(num_envs: int, joint_vel:torch.Tensor, rho1: torch.Tensor, rho2: torch.Tensor, rho3: torch.Tensor, device:str="cuda:0") ->torch.Tensor:
    friction_torque=rho1*(torch.sigmoid(rho2*(joint_vel+rho3))-torch.sigmoid(rho2*rho3))
    return friction_torque

@torch.jit.script
def compute_reset(reset_buf: torch.Tensor,
                  progress_buf: torch.Tensor,
                  episode_length: int) -> torch.Tensor:
    # reset agents
    reset = torch.zeros_like(reset_buf)
    reset = torch.where((progress_buf >= episode_length - 1), torch.ones_like(reset_buf), reset)

    return reset

@torch.jit.script
def compute_flight_distance_reward(object_state: torch.Tensor,
                                   robot_pos: torch.Tensor) -> torch.Tensor:
    
    obj_pos = object_state[:, :3]
    
    # return torch.norm((obj_pos[:,:2] - robot_pos[:2]), dim=1)

    return torch.abs(obj_pos[:,1])


@torch.jit.script
def compute_fall_penalty(object_state: torch.Tensor,
                         object_dims: Tuple[float, float, float],
                         table_dims: Tuple[float, float, float],
                         table_position: Tuple[float, float, float]) -> torch.Tensor:
    '''
    NOTE(dlee): Get following args like this:
        object_height=self.cfg['env']['geometry']['object']['height']
        table_dims=(self.cfg['env']['geometry]['table']['height'], ...)
        table_position=(self.cfg['env']['geometry']['table']['x'], ...)
    '''
    
    obj_pos = object_state[:, :3]
    fall_height = object_dims[2]/2+table_dims[2]
    eps = 0.05
    
    # Check height
    check_height = torch.abs(obj_pos[:, 2]-fall_height) < eps

    # Check on-table
    x_min = table_position[0] - table_dims[0]/2
    x_max = table_position[0] + table_dims[0]/2
    y_min = table_position[1] - table_dims[1]/2
    y_max = table_position[1] + table_dims[1]/2

    check_x = torch.logical_and(obj_pos[:,0] >= x_min, obj_pos[:,0] <= x_max)
    check_y = torch.logical_and(obj_pos[:,1] >= y_min, obj_pos[:,1] <= y_max)
    check_on = torch.logical_and(check_x, check_y)

    # Compute fall
    fall = torch.logical_and(check_height, check_on)

    return fall

@torch.jit.script
def compute_inductive_reward(epsilon: float, 
                             object_state: torch.Tensor,
                             ee_state: torch.Tensor) -> torch.Tensor:
    '''
    inductive reward to make each hand closer to the object
    object_state: (N_env, 13) - first 3 dim = object center position
    ee_state: (N_env, 2, 7) - left/right arm, first 3 dim = end-effector position
    '''
    
    curr_norms = torch.norm(ee_state[:, :, 0:3] - object_state[:, None, 0:3], p=2, dim=-1) # (N_env, 2)
    inductive_reward = 1. / (curr_norms + epsilon)
    inductive_reward = torch.mean(inductive_reward, dim=1) # mean inductive reward over two arms

    return inductive_reward

@torch.jit.script
def compute_height_reward(object_state: torch.Tensor,
                          object_dims: Tuple[float, float, float],
                          table_dims: Tuple[float, float, float],
                          table_position: Tuple[float, float, float]) -> torch.Tensor:
    '''
    dense reward to make the object higher
    object_state: (N_env, 13) - first 3 dim = object center position
    '''

    height_reward = object_state[:, 2] - (table_position[2] + table_dims[2]/2. + object_dims[2]/2.)

    return height_reward


@torch.jit.script
def compute_apogee_reward(apogee: torch.Tensor,
                          terminated: torch.Tensor) -> torch.Tensor:
    '''
    dense reward to make the object higher
    object_state: (N_env, 13) - first 3 dim = object center position
    '''

    return terminated*apogee


@torch.jit.script
def check_object_below_table(object_state: torch.Tensor,
                             object_dims: Tuple[float, float, float],
                             table_dims: Tuple[float, float, float],
                             table_position: Tuple[float, float, float]) -> torch.Tensor:
    '''
    dense reward to make the object higher
    object_state: (N_env, 13) - first 3 dim = object center position
    '''

    return object_state[:, 2] < (table_position[2] + table_dims[2]/2)


@torch.jit.script
def compute_velocity_reward(object_state: torch.Tensor) -> torch.Tensor:
    '''
    dense reward to make the object higher
    object_state: (N_env, 13) - first 3 dim = object center position
    '''

    obj_vel = object_state[:,7:10]

    # # Clip negative x-velocity
    # idx = obj_vel[:,0]<0
    # obj_vel[idx,:0]=0
    # xy_vel = torch.norm(obj_vel[:,:2], dim=1)

    # return xy_vel
    return torch.clamp(obj_vel[:,1], min=0)

@torch.jit.script
def compute_max_speed_reward(max_speed: torch.Tensor,
                             terminated: torch.Tensor) -> torch.Tensor:
    '''
    dense reward to make the object higher
    object_state: (N_env, 13) - first 3 dim = object center position
    '''

    return terminated*max_speed