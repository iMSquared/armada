import sys, os
import math
import numpy as np
import torch
import abc
from pathlib import Path

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from collections import OrderedDict
from simulation.utils.torch_jit_utils import *
from simulation.utils.delay_buffer import DelayBuffer
from simulation.im2gym.tasks.base.vec_task import VecTask
from types import SimpleNamespace
from omegaconf import OmegaConf
from collections import deque
from typing import Deque, Dict, Tuple, List, Union
from scipy.spatial.transform import Rotation as R

from simulation.im2gym import *


BASEDIR = Path(__file__).parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))


class Domain(VecTask):
    """
    Simple card object on the table
    """
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
            high=np.array([1, 1, 1], dtype=np.float32)
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
        "2Dkeypoint": SimpleNamespace(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([320, 240], dtype=np.float32) #TODO: make this to be changed by the config file
        )
    }
    # gripper links state list([num. of instances, num. of fingers, 13]) where 13: (x, y, z, quat, v, omega)
    # The length of list is the history of the state: 0: t, 1: t-1, 2: t-2, ... step.
    _EE_frames_state_history: Deque[torch.Tensor] = deque(maxlen=_state_history_len)
    # Object prim state [num. of instances, 13] where 13: (x, y, z, quat, v, omega)
    # The length of list is the history of the state: 0: t, 1: t-1, 2: t-2, ... step.
    _object_state_history: Deque[torch.Tensor] = deque(maxlen=_state_history_len)
   
    def __init__(self, cfg, sim_device, graphics_device_id, headless, use_state=False, gym=None, record=False):
        """Initializes the card environment configure the buffers.

        Args:
            cfg: Dictionory containing the configuration.
            sim_device: Torch device to store created buffers at (cpu/gpu).
            graphics_device_id: device to render the objects
            headless: if True, it will open the GUI, otherwise, it will just run the server.
        """
        # load default config
        self.cfg = cfg
        self.asymmetric_obs = use_state
        self._set_table_dimension(self.cfg)
        self._object_dims: CuboidalObject = self._set_object_dimension(self.cfg["env"]["geometry"]["object"])

        # action_dim = 19(residual eef(6) + grriper position(1) + gains(12))
        if self.cfg["env"]["restrict_gripper"]:
            self.action_dim = 18 if self.cfg["env"]["controller"] == "JP" else 18 # residual 6 + pd gain 7*2 -> 6*2. fixed to 18.
        else:
            self.action_dim = 19

        self.keypoints_num = int(self.cfg['env']['keypoint']['num'])
        
        if self.cfg['env']["position_history"]:
            self.n_history = 5
            # observations
            self.obs_spec = {
                # robot joint position history
                "robot_q_hist": self._dims.GeneralizedCoordinatesDim.value*self.n_history,
                # object position represented as 2D kepoints
                "object_q": (self._dims.TwoDimensionKeypointDim.value * self.keypoints_num),
                # object goal position represented as 2D kepoints
                "object_q_des": (self._dims.TwoDimensionKeypointDim.value * self.keypoints_num),
                # hand pose
                "hand_state": self._dims.ObjectPoseDim.value,
                # previous action
                "command": self.action_dim
            }

        else:
            self.obs_spec = {
                # robot joint
                "robot_q": self._dims.GeneralizedCoordinatesDim.value,
                # robot joint velocity
                "robot_u": self._dims.GeneralizedVelocityDim.value,
                # object position represented as 2D kepoints
                "object_q": (self._dims.TwoDimensionKeypointDim.value * self.keypoints_num),
                # object goal position represented as 2D kepoints
                "object_q_des": (self._dims.TwoDimensionKeypointDim.value * self.keypoints_num),
                # hand pose
                "hand_state": self._dims.ObjectPoseDim.value,
                # previous action
                "command": self.action_dim
            }

        # states
        if self.asymmetric_obs:
            self.state_spec = self.obs_spec
            self.obs_spec = {
            # robot joint
            "robot_q": self._dims.GeneralizedCoordinatesDim.value,
            # robot joint velocity
            "robot_u": self._dims.GeneralizedVelocityDim.value,
            # object position represented as 2D kepoints
            "object_q": (self._dims.TwoDimensionKeypointDim.value * self.keypoints_num),
            # object goal position represented as 2D kepoints
            "object_q_des": (self._dims.TwoDimensionKeypointDim.value * self.keypoints_num),
            # # gripper (finger) poses
            "ee_state": (self._dims.StateDim.value),
            # hand pose
            "hand_state": self._dims.ObjectPoseDim.value,
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
            # previous action
            "command": self.action_dim
        }
        self.cfg["env"]["numObservations"] = sum(self.obs_spec.values())
        if self.asymmetric_obs:
            self.cfg["env"]["numStates"] = sum(self.state_spec.values())
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
        prim_names += self._get_table_prim_names()
        prim_names += ["object", "goal_object"]
        if self.cfg['env']['scene_randomization']['background']:
            prim_names += ["floor", "back"]
        # mapping from name to asset instance
        self.asset_handles = dict.fromkeys(prim_names)

        # mapping from name to gym rigid body handles
        # name of finger tips links i.e. end-effector frames
        self.hand = self.cfg['env']['hand']     # NOTE: left/right different. tool1 for left arm
        ee_frames = list()
        robot_dof_names = list()
        for hand in [LEFT, RIGHT]:
            if self.hand == hand or self.hand == BIMANUAL:
                ee_frames.append(EE_NAMES[hand])
                for i in range(JOINT_FIRST[hand], JOINT_LAST[hand]+1):
                    robot_dof_names.append(f'joint{i}')

        assert len(robot_dof_names) > 0, 'Hand name might be wrong (left/right/both)'
        self._ee_handles = OrderedDict.fromkeys(ee_frames, None)
        self._robot_dof_indices = OrderedDict.fromkeys(robot_dof_names, None)

        # Inductive reward. This is used for the baseline for claim 1 only.
        self.inductive_reward = self.cfg["env"]["reward_terms"]["inductive_reward"]["activate"]
        self.energy_reward = self.cfg["env"]["reward_terms"]["energy_reward"]["activate"]
        self.controllers={"JP": self.step_jp, "cartesian_impedance": self.step_cartesian_impedance, "position": self.step_jp, "position_2": self.step_jp}
        self.controller = self.controllers[self.cfg["env"]["controller"]]
        self.compute_target = self.compute_ee_target if not self.cfg["env"]["controller"] == "JP" else self.compute_joint_target
        if self.cfg["env"]["controller"] == "position": self.compute_target=self.compute_joint_target
        elif self.cfg["env"]["controller"]=="position_2": self.compute_target=self.compute_joint_target
        # Camera sensor
        if self.cfg["env"]["enableCameraSensors"]:
            self.camera_handles = list()
            self._torch_camera_rgba_images: List[torch.Tensor] = list()
            if self.cfg['env']["camera"]["segmentation"]:
                self._torch_camera_segmentation: List[torch.Tensor] = list()
            # if self.cfg['env']["camera"]["depth"]:
            #     self._torch_camera_depth_images: List[torch.Tensor] = list()
        
        # During initialization its parent create_sim is called
        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, gym=gym, record=record)

        # initialize the buffers
        self.__initialize()
        
        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.8, 0.0, 0.8)
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

        _camera_position = torch.tensor(
            OmegaConf.to_object(self.cfg["env"]["camera"]["position"]), device=self.device
        ).unsqueeze(-1)
        _camera_angle = float(self.cfg["env"]["camera"]["angle"])
        rotation_matrix = torch.tensor((R.from_rotvec(np.array([0.,1.,0.])*np.radians(-90-_camera_angle))\
                                        *R.from_rotvec(np.array([0.,0.,1.,])*np.radians(90))\
                                        *R.from_rotvec(np.array([0.,0.,1.,])*np.radians(180))).inv().as_matrix(),dtype=torch.float).to(self.device)
        self.translation_from_camera_to_object = torch.zeros((3, 4), device=self.device)
        self.translation_from_camera_to_object[:3, :3] = rotation_matrix
        self.translation_from_camera_to_object[:3, 3] = -rotation_matrix.mm(_camera_position)[:, 0]
        self.camera_matrix = self.compute_camera_intrinsics_matrix(
            int(self.cfg["env"]["camera"]["size"][1]), int(self.cfg["env"]["camera"]["size"][0]), 79.3706, self.device
        ) # 55.368
        
        # Observation for the student policy
        if self.cfg["env"]["student_obs"] or self.cfg["env"]["enableCameraSensors"]:
            # self.num_student_obs = 69
            self.student_obs = torch.zeros((self.num_envs, 69), dtype=torch.float, device=self.device) # 69 is the extended obs number
        # set the mdp spaces
        self.__configure_mdp_spaces()

        # Initialize for photorealistic rendering
        if self.cfg["env"]["nvisii"]["photorealistic_rendering"]:
            self.__init_photorealistic_rendering(headless)
        
        # save previous smoothed action for interpolation
        # dim=19 because quaternion of the residual is saved
        self.smoothing_coefficient = self.cfg["env"]["smoothing_coefficient"]
        self.previous_smoothed_action = torch.zeros((self.num_envs, 18), dtype=torch.float, device=self.device)

        # save previous keypoints
        self.prev_keypoints = torch.ones(self.num_envs, 8, 3, dtype=torch.float32, device=self.device)

        # table initial pose
        self.table_pose = torch.zeros((self.num_envs, 13), dtype=torch.float32, device=self.device)
        self.is_hole_wide = False
        if self.cfg['name'] == 'Hole_wide': self.is_hole_wide = True

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

        if self.cfg["env"]["delay"]["activate"]:
            
            self.desired_positions_delay_buffer = DelayBuffer(self.cfg["env"]["delay"]["max_delay"], self.num_envs, device=self.device)
            self.desired_kp_delay_buffer = DelayBuffer(self.cfg["env"]["delay"]["max_delay"], self.num_envs, device=self.device)
            self.desired_kd_delay_buffer = DelayBuffer(self.cfg["env"]["delay"]["max_delay"], self.num_envs, device=self.device)

    def allocate_buffers(self):
        super().allocate_buffers()
        self.object_rewards = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.inductive_rewards = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.energy_rewards = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)


    def use_uniform_random_contact(self):
        self.uniform_random_contact = True

    @abc.abstractmethod
    def _set_object_dimension(self, object_dims) -> CuboidalObject:
        pass

    @abc.abstractmethod
    def _set_table_dimension(self):
        pass

    @abc.abstractmethod
    def _get_table_prim_names(self) -> List[str]:
        pass

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
        if self.cfg['env']['scene_randomization']['light']:
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
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.env_ids = env_ids
        self.actions = actions.clone().to(self.device)
        self.actions[env_ids, :] = 0.0
        # if normalized_action is true, then denormalize them.
        if self.cfg['env']["normalize_action"]:
            action_transformed = unscale_transform(
                self.actions,
                lower=self._action_scale.low,
                upper=self._action_scale.high
            )
        else:
            action_transformed = self.actions
        self.action_transformed = action_transformed

        ### compute target ee pose or joint pose
        self.compute_target()
        self.mean_energy[:] = 0
        self.max_torque[:] = 0

        if self.record:
            self.record_data['act'].append(actions.detach().cpu().numpy())

    def compute_ee_target(self):
        """compute the endeffector target to run the controller"""
        self.sub_goal_pos = self._rigid_body_state[:, self._tool_handle, :3] + self.action_transformed[:, :3]
        rot_residual = self.axisaToquat(self.action_transformed[:, 3:6])
        rot_residual[self.env_ids, 0:3] = 0.0
        rot_residual[self.env_ids, -1] = 1.0
        self.sub_goal_rot = quat_mul(rot_residual, self._rigid_body_state[:, self._tool_handle, 3:7])

    def compute_joint_target(self): # HERE
        """compute the joint target to run the controller"""

        if self.cfg["env"]["controller"] == "position":
            delta_joint_position = self.action_transformed[:, :6]

        elif self.cfg["env"]["controller"] == "position_2":
            delta_mul_kp = self.action_transformed[:, :6]
            kp = self.action_transformed[:, 6:12]
            delta_joint_position = delta_mul_kp / kp
        else:
            delta_joint_position = self._get_delta_dof_pos(
                delta_pose=self.action_transformed[:, :6], ik_method='dls',
                jacobian=self.j_eef.clone(), device=self.device
            )
        
        delta_joint_position[self.env_ids, :] = 0
        desired_joint_position = self.dof_position + delta_joint_position

        # TODO: change to 10% -> 5%
        dof_lower_safety_limit = 0.95*self.franka_dof_lower_limits + 0.05*self.franka_dof_upper_limits
        dof_upper_safety_limit = 0.05*self.franka_dof_lower_limits + 0.95*self.franka_dof_upper_limits

        desired_joint_position = torch.clamp(desired_joint_position, dof_lower_safety_limit, dof_upper_safety_limit)

        self.desired_joint_position = desired_joint_position

    def step_cartesian_impedance(self):
        pos_err = self.sub_goal_pos - self._rigid_body_state[:, self._hand_handle, :3]
        orn_err = orientation_error(self.sub_goal_rot, self._rigid_body_state[:, self._hand_handle, 3:7], 3)
        err = torch.cat([pos_err, orn_err], -1)
        
        hand_vel = self._rigid_body_state[:, self._hand_handle, 7:]

        kp = self.action_transformed[:, 6:12]
        kd = 2 * torch.sqrt(kp) * self.action_transformed[:, 12:18]
        xddt = (kp * err - kd * hand_vel).unsqueeze(-1)
        u = torch.transpose(self.j_eef, 1, 2) @ xddt
        
        return u.squeeze(-1)


    def step_jp(self):

        if self.cfg["env"]["delay"]["activate"]:
            desired_joint_position_delayed = self.desired_positions_delay_buffer.compute(self.desired_joint_position)
            joint_position_error = desired_joint_position_delayed - self.dof_position

            p_gains = self.desired_kp_delay_buffer.compute(self.action_transformed[:, 6:12])
            d_gains = self.desired_kd_delay_buffer.compute(self.action_transformed[:, 12:])

        else:
            joint_position_error = self.desired_joint_position - self.dof_position
            p_gains = self.action_transformed[:, 6:12]
            d_gains = self.action_transformed[:, 12:]

        # p_gains = 50
        # d_gains = np.sqrt(p_gains)
        ct = p_gains * joint_position_error - d_gains * self.dof_velocity

        self.P_gains_ = p_gains
        self.d_gains_ = d_gains

        # joint_position_error = - self.dof_position + torch.tensor([0.0, 0.0, 0.0, 0.6, 1.5708, 0.0], device=self.rl_device)
        # ct = 4. * joint_position_error - 2. * self.dof_velocity
        
        
        
        return ct
    
    def step_controller(self):
        computed_torque = torch.zeros(
            self.num_envs, self._dims.JointTorqueDim.value, dtype=torch.float32, device=self.device
        )
        ct = self.controller()
        computed_torque[:, :6] = ct
        # computed_torque[:, :6] = 0.

        # if not self.cfg["env"]["restrict_gripper"]:
        #     computed_torque[:, 7:] = self.franka_dof_stiffness[7] * (-self.dof_position[:, -2:] + self.action_transformed[:, 6:7])
        #     computed_torque[:, 7:] -= self.franka_dof_damping[7] * self.dof_velocity[:, -2:]
        
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
        self.max_torque = torch.maximum(self.max_torque, torch.norm(applied_torque[:,:7],dim=-1))
        self.applied_torque = applied_torque

        # set computed torques to simulator buffer.
        zero_torque = torch.zeros_like(self.applied_torque)
        # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(zero_torque))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(applied_torque))

        ################################################### 0.003 Hz
        # self.end_time_ = time.time()
        # self.elapsed_time = self.end_time_ - self.start_time_
        # print(f'Hz = {self.elapsed_time}')
        # self.start_time_ = time.time()
        ###################################################

        # self._obs__.append(self.obs_buf[0].to('cpu'))
        # self.__velocity__.append(self.dof_velocity[0].to('cpu'))
        # data_to_save = {
        #     # 'obs_buf': self._obs__,
        #     'dof_velocity' : self.__velocity__
        # }

        # def save_data_as_pkl(data, file_name):
        #     with open(file_name, 'wb') as f:
        #         pickle.dump(data, f)
        # if len(self.__velocity__) < 3162:
        #     save_data_as_pkl(data_to_save, 'velo_save_sim_low_friction.pkl')
        #     print(len(self.__velocity__))

        # apply random external force
        if self.random_external_force:
            object_contact_force = self.net_cf[:, 13]
            contact_force_magnitude = torch.norm(object_contact_force, dim=-1)
            existence_of_contact = (contact_force_magnitude < 0.5)
            rigid_body_count: int = self.gym.get_env_rigid_body_count(self.envs[0])
            force_tensor = (torch.rand(size=(self.num_envs, rigid_body_count, 3), device=self.rl_device) - 0.5) * 0.001
            force_tensor[existence_of_contact, 13, :] = 0
            torque_tensor = (torch.rand(size=(self.num_envs, rigid_body_count, 3), device=self.rl_device) - 0.5) * 0.001
            torque_tensor[existence_of_contact, 13, :] = 0
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(force_tensor.reshape((-1, 3))), 
                gymtorch.unwrap_tensor(torque_tensor.reshape((-1, 3))), 
                gymapi.ENV_SPACE
            )

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

    def _get_delta_dof_pos(self, delta_pose, ik_method, jacobian, device):
        """Get delta Franka DOF position from delta pose using specified IK method."""
        # References:
        # 1) https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf
        # 2) https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf (p. 47)

        if ik_method == 'pinv':  # Jacobian pseudoinverse
            k_val = 1.0
            jacobian_pinv = torch.linalg.pinv(jacobian)
            delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)

        elif ik_method == 'trans':  # Jacobian transpose
            k_val = 1.0
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            delta_dof_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)

        elif ik_method == 'dls':  # damped least squares (Levenberg-Marquardt)
            lambda_val = 0.1     # 0.1
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            lambda_matrix = (lambda_val ** 2) * torch.eye(n=jacobian.shape[1], device=device)
            delta_dof_pos = jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)

        elif ik_method == 'svd':  # adaptive SVD
            k_val = 1.0
            U, S, Vh = torch.linalg.svd(jacobian)
            S_inv = 1. / S
            min_singular_value = 1.0e-5
            S_inv = torch.where(S > min_singular_value, S_inv, torch.zeros_like(S_inv))
            jacobian_pinv = torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6] @ torch.diag_embed(S_inv) @ torch.transpose(U, dim0=1, dim1=2)
            delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)

        return delta_dof_pos

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
        
        # check termination e.g. box is dropped from table.
        self._check_termination()

        if self.observation_randomize: # TODO: change self.randomize option separately
            # vel
            # self.obs_buf[:, 6:12] = self.keypoints_randomizer(self.randomization_params, self.obs_buf[:, 6:12])
            # obs_buf, torch.randn_like(obs_buf) * var + mu

            self.obs_buf[:, 12:28] = self.observation_randomizer(self.randomization_params["keypoints"], self.obs_buf[:, 12:28]) # cur keypoint
            self.obs_buf[:, :6] = self.observation_randomizer(self.randomization_params["joint_pos"], self.obs_buf[:, :6]) # joint pos
            self.obs_buf[:, 6:12] = self.observation_randomizer(self.randomization_params["joint_vel"], self.obs_buf[:, 6:12]) # joint vel
            self.obs_buf[:, 44:51] = self.observation_randomizer(self.randomization_params["hand_pos"], self.obs_buf[:, 44:51]) # hand pose
            # self.extras["regularization"]=self.regularization
        
    def compute_reward(self, actions):
        self.rew_buf[:] = 0
        self.reset_buf[:] = 0
        if self.asymmetric_obs:
            self.rew_buf[:], self.object_rewards[:], self.inductive_rewards[:], self.energy_rewards[:], self.reset_buf[:], self.regularization[:] = compute_card_reward(
                self.states_buf,
                self.reset_buf,
                self.progress_buf,
                self.max_episode_length,
                (self.cfg["env"]["reward_terms"]["object_dist"]["weight1"], 
                 self.cfg["env"]["reward_terms"]["object_dist"]["weight2"], 
                 self.cfg["env"]["reward_terms"]["object_dist"]["multiplier"]),
                self.cfg["env"]["reward_terms"]["inductive_reward"]["multiplier"],
                (self.cfg["env"]["reward_terms"]["object_dist"]["epsilon"], 
                 self.cfg["env"]["reward_terms"]["object_rot"]["epsilon"]),
                self._object_goal_poses_buf,
                self.max_torque,
                self.mean_energy,
                self._object_state_history[0],
                self._EE_frames_state_history[0],
                self._object_dims.size,
                self.inductive_reward,
                self.energy_reward,
                self.cfg["env"]["reward_terms"]["energy_reward"]["multiplier"],
                self.cfg["env"]["reward_terms"]["position_only"]
            )
        else:
            self.rew_buf[:], self.object_rewards[:], self.inductive_rewards[:], self.energy_rewards[:], self.reset_buf[:], _ = compute_card_reward(
                self.obs_buf,
                self.reset_buf,
                self.progress_buf,
                self.max_episode_length,
                (self.cfg["env"]["reward_terms"]["object_dist"]["weight1"], 
                 self.cfg["env"]["reward_terms"]["object_dist"]["weight2"],
                 self.cfg["env"]["reward_terms"]["object_dist"]["multiplier"]),
                self.cfg["env"]["reward_terms"]["inductive_reward"]["multiplier"],
                (self.cfg["env"]["reward_terms"]["object_dist"]["epsilon"], 
                 self.cfg["env"]["reward_terms"]["object_rot"]["epsilon"]),
                self._object_goal_poses_buf,
                self.max_torque,
                self.mean_energy,
                self._object_state_history[0],
                self._EE_frames_state_history[0],
                self._object_dims.size,
                self.inductive_reward,
                self.energy_reward,
                self.cfg["env"]["reward_terms"]["energy_reward"]["multiplier"],
                self.cfg["env"]["reward_terms"]["position_only"]
            )

    """
    Private functions
    """

    def __initialize(self):
        """Allocate memory to various buffers.
        """
        if self.cfg['env']["student_obs"] or self.cfg['env']["enableCameraSensors"]:
            # store the sampled initial poses for the object
            self._object_initial_poses_buf = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float)
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
        
        if self.cfg['env']["position_history"]:
            # dof position history
            self.dof_position_history = self.dof_position.unsqueeze(0).repeat(self.n_history, 1, 1)

        self.dof_velocity = self.dof_state[..., 1]
        # rigid body
        self._rigid_body_state: torch.Tensor = gymtorch.wrap_tensor(_rigid_body_tensor).view(self.num_envs, -1, 13)
        # root actors
        self._actors_root_state: torch.Tensor = gymtorch.wrap_tensor(_actor_root_state_tensor).view(-1, 13)
        self.robot_link_dict: Dict[str, int] = self.gym.get_asset_rigid_body_dict(self.asset_handles["robot"])
        hand_index: Union[int, List[int]] = self.robot_link_dict[EE_NAMES[self.hand]] if self.hand in [LEFT, RIGHT] else [self.robot_link_dict[EE_NAMES[hand]] for hand in [LEFT, RIGHT]]      # TODO(L/R): Account for bimanual hand
        # jacobian
        self.jacobian: torch.Tensor = gymtorch.wrap_tensor(_jacobian)
        self.j_eef = self.jacobian[:, (hand_index - 1), :, :7]
        # mass matirx
        self.mm: torch.Tensor = gymtorch.wrap_tensor(_mm)
        self.mm = self.mm[:, :(hand_index - 1), :(hand_index - 1)]
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
        if self.asymmetric_obs:
            self._states_scale = SimpleNamespace(low=None, high=None)
            self.regularization=torch.zeros(self.num_envs,dtype=torch.float, device=self.device)
        self._action_scale = SimpleNamespace(low=None, high=None)
        if self.cfg["env"]["student_obs"] or self.cfg["env"]["enableCameraSensors"]:
            self._std_obs_scale = SimpleNamespace(low=None, high=None)

        self._successes = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        if self.cfg["env"]["extract_successes"]:
            self.extract = False
            self._successes_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.max_torque = torch.zeros(self.num_envs,dtype=torch.float, device=self.device)
        self.mean_energy = torch.zeros(self.num_envs,dtype=torch.float, device=self.device)

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
            p_gain_scale = [1.8, 2.4, 1.9, 1.8, 0.6, 0.7]
            d_gain_scale = [0.04, 0.06, 0.06, 0.06, 0.01, 0.01]

        if self.cfg["env"]["restrict_gripper"]:
            if self.cfg["env"]["controller"]=="JP":
                self._action_scale.low = to_torch(
                    residual_scale_low+[0.1]*6+[0.0]*6, 
                    device=self.device
                )  # TODO: add explanations to this values
                # self._action_scale.high = to_torch(
                #     residual_scale_high+[6.0, 6.0, 6.0, 1.6, 0.7, 0.7]+[2.0, 2.0, 2.0, 0.1, 0.04, 0.04], 
                #     device=self.device
                # )
                self._action_scale.high = to_torch(
                    residual_scale_high+[1.8, 2.2, 1.6, 1.6, 0.7, 0.7] + [0.06, 0.08, 0.1, 0.1, 0.04, 0.04], 
                    device=self.device
                )            

            elif self.cfg["env"]["controller"]=="position":
                self._action_scale.low = to_torch(
                    # [-0.25, -0.3, -0.45, -0.40, -0.30, -0.30]+[0.1]*6+[0.0]*6, 
                    # [-0.30, -0.35, -0.45, -0.40, -0.30, -0.35]+[0.9, 1.2, 0.95, 0.9, 0.3, 0.35]+[0.0]*6, 
                   position_residual_scale_low +[0.9, 1.2, 0.95, 0.9, 0.3, 0.35]+[0.0]*6,
                    device=self.device
                )
                self._action_scale.high = to_torch(
                    # [0.30, 0.35, 0.45, 0.40, 0.30, 0.35]+[1.8, 2.4, 1.9, 1.8, 0.6, 0.7] + [0.04, 0.06, 0.06, 0.06, 0.01, 0.01], 
                    position_residual_scale_high + p_gain_scale + d_gain_scale,
                   device=self.device)
                
            elif self.cfg["env"]["controller"]=="position_2":
                self._action_scale.low = to_torch(
                    # [-0.54, -0.84, -0.855, -0.72, -0.245, -0.245]+[0.9, 1.2, 0.95, 0.9, 0.35, 0.35]+[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                    # [-1.5, -2.5, -2.5, -2.0, -0.6, -0.6]+[1.5, 2.0, 1.7, 1.7, 0.6, 0.6]+[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                    position_residual_scale_low +p_gain_scale_low+[0.0]*6,
                    # [-0.66241973, -1.04474772, -1.05964622, -0.88322631, -0.29306443, -0.29306443]+[0.99680971, 1.32907961, 1.06725616, 1.02207742, 0.38983846, 0.38983846]+[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    # [-0.7286617 , -1.14922249, -1.16561084, -0.97154894, -0.32237087, -0.32237087]+[1.09649068, 1.46198757, 1.17398177, 1.12428516, 0.42882231, 0.42882231]+[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    # [-0.495, -0.726, -0.792, -0.704, -0.294, -0.221, 1.8, 2.2, 1.6, 1.6, 0.862, 0.597, 0.06, 0.08, 0.1, 0.1, 0.102, 0.0926],
                    device=self.device
                )
                self._action_scale.high = to_torch(
                    # [0.54, 0.84, 0.855, 0.72, 0.245, 0.245]+[1.8, 2.4, 1.9, 1.8, 0.7, 0.7] + [0.03, 0.045, 0.45, 0.45, 0.01, 0.01], 
                    # [1.5, 2.5, 2.5, 2.0, 0.6, 0.6]+[3.0, 4.0, 3.4, 3.4, 1.2, 1.2]+[0.03, 0.045, 0.45, 0.45, 0.01, 0.01],
                    position_residual_scale_high + p_gain_scale + [0.03, 0.045, 0.45, 0.45, 0.01, 0.01],
                    # [0.66241973, 1.04474772, 1.05964622, 0.88322631, 0.29306443, 0.29306443]+[1.99361942, 2.65815922, 2.13451231, 2.04415483, 0.77967693, 0.77967693]+[0.03, 0.045, 0.45, 0.45, 0.01, 0.01],
                    # [0.7286617 , 1.14922249, 1.16561084, 0.97154894, 0.32237087, 0.32237087]+[2.19298136, 2.92397514, 2.34796354, 2.24857031, 0.85764462, 0.85764462]+[0.03, 0.045, 0.45, 0.45, 0.01, 0.01],
                    # [0.495, 0.726, 0.792, 0.704, 0.294, 0.221, 1.8, 2.2, 1.6, 1.6, 0.862, 0.597, 0.06, 0.08, 0.1, 0.1, 0.102, 0.0926],
                    device=self.device)


        #     else:
        #         self._action_scale.low = to_torch(residual_scale_low+[ 10.0]*6          +[0.0]*6, device=self.device)  # TODO: add explanations to this values
        #         self._action_scale.high = to_torch(residual_scale_high+[200.0]*3+[300.0]*3+[2.0]*6, device=self.device)
        # else:
        #     # plus gain
        #     self._action_scale.low = to_torch(residual_scale_low+[0.0]+[10.0]*6+[0.0]*6, device=self.device)
        #     self._action_scale.high = to_torch(residual_scale_high+[0.04]+[300.0]*6+[2.0]*6, device=self.device)

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
        if self.cfg["env"]["keypoint"]["activate"]:

            if self.cfg['env']["position_history"]:
                self._observations_scale.low = torch.cat([
                    # self.franka_dof_lower_limits,
                    # -self.franka_dof_speed_scales,
                    self.franka_dof_lower_limits.repeat(self.n_history),
                    self._object_limits["2Dkeypoint"].low.repeat(8),
                    self._object_limits["2Dkeypoint"].low.repeat(8),
                    self._ee_limits["ee_position"].low,
                    self._ee_limits["ee_orientation"].low,
                    obs_action_scale.low
                ])
                self._observations_scale.high = torch.cat([
                    # self.franka_dof_lower_limits,
                    # -self.franka_dof_speed_scales,
                    self.franka_dof_upper_limits.repeat(self.n_history),
                    self._object_limits["2Dkeypoint"].high.repeat(8),
                    self._object_limits["2Dkeypoint"].high.repeat(8),
                    self._ee_limits["ee_position"].high,
                    self._ee_limits["ee_orientation"].high,
                    obs_action_scale.high
                ])
            else:
                self._observations_scale.low = torch.cat([
                    self.franka_dof_lower_limits,
                    -self.franka_dof_speed_scales,
                    self._object_limits["2Dkeypoint"].low.repeat(8),
                    self._object_limits["2Dkeypoint"].low.repeat(8),
                    self._ee_limits["ee_position"].low,
                    self._ee_limits["ee_orientation"].low,
                    obs_action_scale.low
                ])
                self._observations_scale.high = torch.cat([
                    self.franka_dof_upper_limits,
                    self.franka_dof_speed_scales,
                    self._object_limits["2Dkeypoint"].high.repeat(8),
                    self._object_limits["2Dkeypoint"].high.repeat(8),
                    self._ee_limits["ee_position"].high,
                    self._ee_limits["ee_orientation"].high,
                    obs_action_scale.high
                ])
        else:
            # keypoints are always activated. 
            pass
        if self.cfg["env"]["student_obs"] or self.cfg["env"]["enableCameraSensors"]:
            self._std_obs_scale.low = torch.cat([
                self._ee_limits["ee_position"].low,
                self._ee_limits["ee_orientation"].low,
                torch.tensor([0.0], dtype=torch.float, device=self.device),
                self.franka_dof_lower_limits,
                -self.franka_dof_speed_scales,
                obs_action_scale.low,
                torch.tensor([0.0], dtype=torch.float, device=self.device),
                self._object_limits["position"].low,
                self._object_limits["orientation"].low,
                self._object_limits["position"].low,
                self._object_limits["orientation"].low,
                self._object_limits["position"].low,
                self._object_limits["orientation"].low,
                self._object_limits["position"].low,
                self._object_limits["orientation"].low,
            ])
            self._std_obs_scale.high = torch.cat([
                self._ee_limits["ee_position"].high,
                self._ee_limits["ee_orientation"].high,
                torch.tensor([0.08], dtype=torch.float, device=self.device),
                self.franka_dof_upper_limits,
                self.franka_dof_speed_scales,
                obs_action_scale.high,
                torch.tensor([1.0], dtype=torch.float, device=self.device),
                self._object_limits["position"].high,
                self._object_limits["orientation"].high,
                self._object_limits["position"].high,
                self._object_limits["orientation"].high,
                self._object_limits["position"].high,
                self._object_limits["orientation"].high,
                self._object_limits["position"].high,
                self._object_limits["orientation"].high,
            ])
        if self.asymmetric_obs:
            states_dim = sum(self.state_spec.values())
            self._states_scale.low = self._observations_scale.low  
            self._states_scale.high = self._observations_scale.high 
            self._observations_scale.low = torch.cat([
                self.franka_dof_lower_limits,
                -self.franka_dof_speed_scales,
                self._object_limits["2Dkeypoint"].low.repeat(8),
                self._object_limits["2Dkeypoint"].low.repeat(8),
                self._ee_limits["ee_position"].low,
                self._ee_limits["ee_orientation"].low,
                obs_action_scale.low
            ])
            self._observations_scale.high = torch.cat([
                self.franka_dof_upper_limits,
                self.franka_dof_speed_scales,
                self._object_limits["2Dkeypoint"].high.repeat(8),
                self._object_limits["2Dkeypoint"].high.repeat(8),
                self._ee_limits["ee_position"].high,
                self._ee_limits["ee_orientation"].high,
                obs_action_scale.high
            ])
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
        # states
        if self.asymmetric_obs:
            if self._states_scale.low.shape[0] != states_dim or self._states_scale.high.shape[0] != states_dim:
                msg = f"States scaling dimensions mismatch. " \
                    f"\tLow: {self._states_scale.low.shape[0]}, " \
                    f"\tHigh: {self._states_scale.high.shape[0]}, " \
                    f"\tExpected: {states_dim}."
                raise AssertionError(msg)
        # actions
        if self._action_scale.low.shape[0] != action_dim or self._action_scale.high.shape[0] != action_dim:
            msg = f"Actions scaling dimensions mismatch. " \
                  f"\tLow: {self._action_scale.low.shape[0]}, " \
                  f"\tHigh: {self._action_scale.high.shape[0]}, " \
                  f"\tExpected: {action_dim}."
            raise AssertionError(msg)
    
    def update_residual_scale(self, position_ratio, kp_ratio, kd_ratio):

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

    @abc.abstractmethod
    def _max_dist_btw_obj_and_goal(self) -> torch.Tensor:
        pass

    def __create_scene_assets(self):
        """ Define Gym assets for table, robot and object.
        """
        # define assets
        self.asset_handles["robot"] = self.__define_robot_asset()
        self._define_table_asset()
        if self.cfg['env']['scene_randomization']['background']:
            self.asset_handles["floor"] = self.__define_floor_asset()
            self.asset_handles["back"] = self.__define_back_asset()
        self.asset_handles["object"] = self._define_object_asset()
        self.asset_handles["goal_object"] = self.__define_goal_object_asset()
        # display the properties (only for debugging)
        # robot
        print("Franka Robot Asset: ")
        print(f'\t Number of bodies: {self.gym.get_asset_rigid_body_count(self.asset_handles["robot"])}')
        print(f'\t Number of shapes: {self.gym.get_asset_rigid_shape_count(self.asset_handles["robot"])}')
        print(f'\t Number of dofs: {self.gym.get_asset_dof_count(self.asset_handles["robot"])}')
        print(f'\t Number of actuated dofs: {self._dims.JointTorqueDim.value}')
        # table
        # print("Card table Asset: ")
        # print(f'\t Number of bodies: {self.gym.get_asset_rigid_body_count(self.asset_handles["table"])}')
        # print(f'\t Number of shapes: {self.gym.get_asset_rigid_shape_count(self.asset_handles["table"])}')

    def __create_envs(self):
        """Create various instances for the environment.
        """
        robot_dof_props = self.gym.get_asset_dof_properties(self.asset_handles["robot"])

        # robot_rigid_props = self.gym.get_asset_rigid_shape_properties(self.asset_handles["robot"])
        # collision_filter = [0b1100000, 0b0110000, 0b0011000, 0b0001100, 0b0000110, 0b0000011]

        # for cf, p in zip(collision_filter, robot_rigid_props):
        #     p.filter = cf
        # self.gym.set_asset_rigid_shape_properties(self.asset_handles["robot"], robot_rigid_props)

        # set dof properites based on the control mode
        # self.franka_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400], dtype=torch.float, device=self.device)
        # self.franka_dof_damping = to_torch([80, 80, 80, 80, 80, 80], dtype=torch.float, device=self.device)

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

            robot_dof_props['friction'][k]  = sysid_params[k, 0]
            robot_dof_props['damping'][k]   = sysid_params[k, 1]
            robot_dof_props['armature'][k]  = sysid_params[k, 2]

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

            # begin aggregration
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add robot to environment
            # robot_pose = gymapi.Transform()
            # robot_pose.p = gymapi.Vec3(0.16+0.13, -0.37, 1.0)
            # robot_pose.p = gymapi.Vec3(0.16+0.13, -0.155, 1.0)
            # robot_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), 1.5707963268) 

            robot_pose = gymapi.Transform()
            robot_pose.p = gymapi.Vec3(0, 0, 0)
            robot_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            # param6: bitwise filter for elements in the same collisionGroup to mask off collision
            # originally 0 but 0 makes self-collision. positive value will mask out self collision
            Franka_actor = self.gym.create_actor(env_ptr, self.asset_handles["robot"], robot_pose, "robot", env_index, 0, 3)

            self.gym.set_actor_dof_properties(env_ptr, Franka_actor, robot_dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, Franka_actor)
            Franka_idx = self.gym.get_actor_index(env_ptr, Franka_actor, gymapi.DOMAIN_SIM)
            actor_indices["robot"].append(Franka_idx)

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
            
            # add goal object to environment
            if not self.cfg['env']["enableCameraSensors"]:
                goal_handle = self.gym.create_actor(
                    env_ptr, self.asset_handles["goal_object"], gymapi.Transform(), "goal_object", (env_index + self.num_envs), 0, 0
                )
                goal_color = gymapi.Vec3(0.3, 0.3, 0.3)
                self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, goal_color)
                goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
                actor_indices["goal_object"].append(goal_object_idx)
            
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
    
    @abc.abstractmethod
    def _create_table(self, env_ptr, env_index: int, actor_indices: Dict[str, List[int]]):
        pass

    @abc.abstractmethod
    def _create_object(self, env_ptr, env_index: int, actor_indices: Dict[str, List[int]]):
        pass

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

        goal_reached = self._check_success()
        self.rew_buf[goal_reached] += self.cfg["env"]["reward_terms"]["object_dist"]["weight2"]
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

        # dof_pos_upper_lower_diff = self.franka_dof_upper_limits[:6] - self.franka_dof_lower_limits[:6]
        # dof_pos_curr_lower_diff = self.dof_position[:, :6] - self.franka_dof_lower_limits[:6]
        # dof_pos_curr_ratio = dof_pos_curr_lower_diff / dof_pos_upper_lower_diff
        # dof_pos_curr_lowest_ratio, _ = torch.min(dof_pos_curr_ratio, dim=-1)
        # dof_pos_curr_highest_ratio, _ = torch.max(dof_pos_curr_ratio, dim=-1)
        # dof_pos_low_envs = torch.le(dof_pos_curr_lowest_ratio, offset)
        # dof_pos_high_envs = torch.gt(dof_pos_curr_highest_ratio, 1.0-offset)

        dof_pos_limit_exceeded_envs = torch.logical_or(dof_pos_low_envs, dof_pos_high_envs)
        return dof_pos_limit_exceeded_envs

    def _check_dof_velocity_limit_reset(self, offset):
        dof_vel_upper_safety_limit = (1.0-offset)*self.franka_dof_speed_scales[:6]
        dof_vel_high_envs = torch.any(torch.gt(torch.abs(self.dof_velocity[:, :6]), dof_vel_upper_safety_limit[None, :]), dim=1)
        return dof_vel_high_envs

    @abc.abstractmethod
    def _check_failure(self) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def _check_success(self) -> torch.Tensor:
        pass

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
        # if joint training
        # we must save which data is used to reset environment
        if object_initial_state_config["type"] == "pre_contact_policy":
            indices = torch.arange(
                self._init_data_use_count, (self._init_data_use_count + env_ids.shape[0]), dtype=torch.long, device=self.device
            )
            indices %= self._robot_buffer.shape[0]
            self._init_data_use_count += env_ids.shape[0]
            self._init_data_indices[env_ids] = indices

        # -- Sampling of height of the table
        if not self.is_hole_wide:
            self._sample_table_poses(env_ids)
        # -- Sampling of initial pose of the object
        self._sample_object_poses(env_ids)
        # -- Sampling of goal pose of the object
        self._sample_object_goal_poses(env_ids)
        # -- Robot joint state
        self._sample_robot_state(env_ids)

        # C) Extract franka indices to reset
        robot_indices = self.actor_indices["robot"][env_ids].to(torch.int32)
        object_indices = self.actor_indices["object"][env_ids].to(torch.int32)
        if not self.is_hole_wide:
            table_indices = self.actor_indices["table"][env_ids].to(torch.int32)
        if not self.cfg['env']["enableCameraSensors"]:
            goal_object_indices = self.actor_indices["goal_object"][env_ids].to(torch.int32)
            if self.is_hole_wide:
                all_indices = torch.unique(torch.cat([robot_indices, object_indices, goal_object_indices]))
            else:
                all_indices = torch.unique(torch.cat([robot_indices, object_indices, goal_object_indices, table_indices]))
        else:
            if self.is_hole_wide:
                all_indices = torch.unique(torch.cat([robot_indices, object_indices]))
            else:
                all_indices = torch.unique(torch.cat([robot_indices, object_indices, table_indices]))
        # D) Set values into simulator
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(robot_indices), len(robot_indices)
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._actors_root_state), gymtorch.unwrap_tensor(all_indices), len(all_indices)
        )

        # envenv = self.envs[0]
        # robot_handle = self.gym.get_actor_handle(envenv, self.actor_indices["robot"][0])
        # print(self.gym.get_actor_dof_properties(envenv, robot_handle))

        if self.cfg["env"]["delay"]["activate"]:
                                                                 
            time_lags = torch.randint(
                low=self.cfg["env"]["delay"]["min_delay"],
                high=self.cfg["env"]["delay"]["max_delay"] + 1,
                size=(env_ids.shape[0],),
                dtype=torch.int,
                device=self.rl_device,
            )

            self.desired_positions_delay_buffer.set_time_lag(time_lags, env_ids)
            self.desired_kp_delay_buffer.set_time_lag(time_lags, env_ids)
            self.desired_kd_delay_buffer.set_time_lag(time_lags, env_ids)

            # reset buffers
            self.desired_positions_delay_buffer.reset(env_ids)
            self.desired_kp_delay_buffer.reset(env_ids)
            self.desired_kd_delay_buffer.reset(env_ids)
        
    
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
        robot_asset = self.gym.load_asset(self.sim, self._assets_dir, "urdf/RobotBimanualV5/urdf/Simplify_Robot_plate_gripper_LEFT.urdf", robot_asset_options)
        robot_props = self.gym.get_asset_rigid_shape_properties(robot_asset)
        self.EE_handle: Union[int, List[int]] = self.gym.find_asset_rigid_body_index(robot_asset, EE_LINK[self.hand]) if self.hand in [LEFT, RIGHT] else [self.gym.find_asset_rigid_body_index(robot_asset, EE_LINK[hand]) for hand in [LEFT, RIGHT]]      # TODO(L/R): account for bimanual
        # self.finger_handle = self.gym.find_asset_rigid_body_index(robot_asset, 'link6_finger')
        cnt=0
        ee_start = self.gym.get_asset_rigid_body_shape_indices(robot_asset)[self.EE_handle].start
        ee_count = self.gym.get_asset_rigid_body_shape_indices(robot_asset)[self.EE_handle].count

        for i, p in enumerate(robot_props):
            if ee_start <= i and i < ee_start+ee_count:
                p.friction = 2.0
            else:
                p.friction = 0.2
            # p.restitution = 0.2
        self.gym.set_asset_rigid_shape_properties(robot_asset, robot_props)
        after_update_props = self.gym.get_asset_rigid_shape_properties(robot_asset)
        
        # if self.cfg["env"]["hand_force_limit"]:
        #     hand_idx = self.gym.find_asset_rigid_body_index(robot_asset, "panda_hand")
        #     hand_force_sensor_pose = gymapi.Transform(gymapi.Vec3(0., 0.0, 0.0))
        #     hand_force_sensor_props = gymapi.ForceSensorProperties()
        #     hand_force_sensor_props.enable_forward_dynamics_forces = False
        #     hand_force_sensor_props.enable_constraint_solver_forces = True
        #     hand_force_sensor_props.use_world_frame = True
        #     hand_force_sensor_idx = self.gym.create_asset_force_sensor(robot_asset, hand_idx, hand_force_sensor_pose, hand_force_sensor_props)

        for frame_name in self._ee_handles.keys():
            self._ee_handles[frame_name] = self.gym.find_asset_rigid_body_index(robot_asset, frame_name)
            # check valid handle
            if self._ee_handles[frame_name] == gymapi.INVALID_HANDLE:
                msg = f"Invalid handle received for frame: `{frame_name}`."
                print(msg)
        # self._hand_handle = self.gym.find_asset_rigid_body_index(robot_asset, "panda_hand")
        # self._tool_handle = self.gym.find_asset_rigid_body_index(robot_asset, "panda_tool")
        self._hand_handle: int = self.EE_handle
        self._tool_handle: Union[int, List[int]] = self.gym.find_asset_rigid_body_index(robot_asset, EE_NAMES[self.hand]) if self.hand in [LEFT, RIGHT] else [self.gym.find_asset_rigid_body_index(robot_asset, EE_NAMES[hand]) for hand in [LEFT, RIGHT]]         # TODO(L/R): Account for bimanual

        for dof_name in self._robot_dof_indices.keys():
            self._robot_dof_indices[dof_name] = self.gym.find_asset_dof_index(robot_asset, dof_name)
            # check valid handle
            if self._robot_dof_indices[dof_name] == gymapi.INVALID_HANDLE:
                msg = f"Invalid index received for DOF: `{dof_name}`."
                print(msg)
        # return the asset
        return robot_asset

    @abc.abstractmethod
    def _define_table_asset(self):
        pass        

    def __define_floor_asset(self):
        """ Define Gym asset for a floor.
        """
        # define table asset
        floor_asset_options = gymapi.AssetOptions()
        floor_asset_options.disable_gravity = True
        floor_asset_options.fix_base_link = True
        floor_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        floor_asset_options.thickness = 0.001
        # load table asset
        floor_asset = self.gym.create_box(self.sim, 2, 5, 0.002, floor_asset_options)

        return floor_asset

    def __define_back_asset(self):
        """ Define Gym asset for the background.
        """
        # define table asset
        back_asset_options = gymapi.AssetOptions()
        back_asset_options.disable_gravity = True
        back_asset_options.fix_base_link = True
        back_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        back_asset_options.thickness = 0.001
        # load table asset
        back_asset = self.gym.create_box(self.sim, 0.002, 2.5, 1, back_asset_options)

        return back_asset

    @abc.abstractmethod
    def _define_object_asset(self):
        pass

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
        # extract frame handles
        EE_indices = list(self._ee_handles.values())
        object_indices = self.actor_indices["object"]
        # update state histories
        self._EE_frames_state_history.appendleft(self._rigid_body_state[:, EE_indices, :7])
        self._object_state_history.appendleft(self._actors_root_state[object_indices])
        # self._hand_vel_history[:,-1]=self._rigid_body_state[:,self._hand_handle,7:]
        # fill the observations and states buffer
        self.__compute_teacher_observations()
        if self.cfg["env"]["enableCameraSensors"]:
            self.__compute_camera_observations()
        if self.cfg["env"]["nvisii"]["photorealistic_rendering"]:
            self.__compute_photorealistic_rendering()
        
        # normalize observations if flag is enabled
        if self.cfg['env']["normalize_obs"]:
            # for normal obs
            self.obs_buf = scale_transform(
                self.obs_buf,
                lower=self._observations_scale.low,
                upper=self._observations_scale.high
            )
            if self.asymmetric_obs:
                self.states_buf = scale_transform(
                    self.states_buf,
                    lower=self._states_scale.low,
                    upper=self._states_scale.high
                )

        if self.record:
            self.record_data['obs'].append(self.obs_buf.detach().cpu().numpy())
            obs = unscale_transform(self.obs_buf, lower=self._observations_scale.low, upper=self._observations_scale.high)
            self.record_data['joint_pos'].append(obs.detach().cpu().numpy()[:,:self._dims.GeneralizedCoordinatesDim.value])


    def __compute_teacher_observations(self):

        if self.asymmetric_obs:

            # generalized coordinates
            start_offset = 0
            end_offset = start_offset + self._dims.GeneralizedCoordinatesDim.value
            self.states_buf[:, start_offset:end_offset] = self.dof_position

            # generalized velocities
            start_offset = end_offset
            end_offset = start_offset + self._dims.GeneralizedVelocityDim.value
            self.states_buf[:, start_offset:end_offset] = self.dof_velocity

            # object pose as keypoint
            start_offset = end_offset
            end_offset = start_offset + self._dims.KeypointDim.value*self.keypoints_num
            current_keypoints = gen_keypoints(pose=self._object_state_history[0][:, 0:7], size=self._object_dims.size)
            self.states_buf[:, (start_offset):(end_offset)] = current_keypoints.view(self.num_envs, 24)[:]
            
            # use previous keypoint to mimic delay

            # self.states_buf[:, (start_offset):(end_offset)] = self.prev_keypoints.view(self.num_envs, 24)[:]
            # self.prev_keypoints = current_keypoints

            # object desired pose as keypoint
            start_offset = end_offset
            end_offset = start_offset + self._dims.KeypointDim.value*self.keypoints_num
            goal_keypoints = gen_keypoints(pose=self._object_goal_poses_buf, size=self._object_dims.size)
            self.states_buf[:, start_offset:end_offset] = goal_keypoints.view(self.num_envs, 24)[:]
            
            # object velcity
            start_offset = end_offset
            end_offset = start_offset + self._dims.ObjectVelocityDim.value
            self.states_buf[:, start_offset:end_offset] = self._object_state_history[0][:, 7:13]

            # joint torque
            start_offset = end_offset
            end_offset = start_offset + self._dims.JointTorqueDim.value
            self.states_buf[:, start_offset:end_offset] = self.dof_torque

            # distance between object and goal
            start_offset = end_offset
            end_offset = start_offset + 1
            self.states_buf[:, start_offset:end_offset] = torch.norm(self._object_state_history[0][:,0:3]-self._object_goal_poses_buf[:,0:3],2,-1).unsqueeze(-1)

            # angle differences between object and goal
            start_offset = end_offset
            end_offset = start_offset + 1
            self.states_buf[:, start_offset:end_offset] = quat_diff_rad(self._object_state_history[0][:,3:7], self._object_goal_poses_buf[:,3:7]).unsqueeze(-1)

            # previous action
            start_offset = end_offset
            end_offset = start_offset + self.action_dim 
            self.states_buf[:, start_offset:end_offset] = self.actions

            ########################## observation
            # robot pose
            start_state_offset = 0
            end_state_offset = start_state_offset + self._dims.GeneralizedCoordinatesDim.value
            self.obs_buf[:, start_state_offset:end_state_offset] = self.dof_position

            # robot velocity
            start_state_offset = end_state_offset
            end_state_offset = start_state_offset + self._dims.GeneralizedVelocityDim.value
            self.obs_buf[:, start_state_offset:end_state_offset] = self.dof_velocity

            # 2D projected object keypoints
            start_state_offset = end_state_offset
            end_state_offset = start_state_offset + self._dims.TwoDimensionKeypointDim.value*self.keypoints_num
            self.obs_buf[:, start_state_offset:end_state_offset] = compute_projected_points(self.translation_from_camera_to_object, current_keypoints, self.camera_matrix, self.device).reshape(self.num_envs, 16)[:]

            # 2D projected goal keypoints
            start_state_offset = end_state_offset
            end_state_offset = start_state_offset + self._dims.TwoDimensionKeypointDim.value*self.keypoints_num
            self.obs_buf[:, start_state_offset:end_state_offset] = compute_projected_points(self.translation_from_camera_to_object, goal_keypoints, self.camera_matrix, self.device).reshape(self.num_envs, 16)[:]

            # hand pose
            start_state_offset = end_state_offset
            end_state_offset = start_state_offset + self._dims.ObjectPoseDim.value
            self.obs_buf[:, start_state_offset:end_state_offset] = self._rigid_body_state[:, self._tool_handle, :7]

            # previous action
            start_state_offset = end_state_offset
            end_state_offset = start_state_offset + self.action_dim
            self.obs_buf[:, start_state_offset:end_state_offset] = self.actions

        else:

            if self.cfg['env']["position_history"]:
                # position history
                self.dof_position_history = torch.roll(self.dof_position_history, shifts=-1, dims=0)
                self.dof_position_history[-1] = self.dof_position

                start_offset = 0
                end_offset = start_offset + self._dims.GeneralizedVelocityDim.value*self.n_history
                self.obs_buf[:, start_offset:end_offset] = self.dof_position_history.transpose(0, 1).reshape(self.num_envs, -1)
            
            else:
                # generalized coordinates
                start_offset = 0
                end_offset = start_offset + self._dims.GeneralizedCoordinatesDim.value
                self.obs_buf[:, start_offset:end_offset] = self.dof_position

                # generalized velocities
                start_offset = end_offset
                end_offset = start_offset + self._dims.GeneralizedVelocityDim.value
                self.obs_buf[:, start_offset:end_offset] = self.dof_velocity

            # 2D projected object keypoints
            start_offset = end_offset
            end_offset = start_offset + self._dims.TwoDimensionKeypointDim.value * self.keypoints_num
            current_keypoints = gen_keypoints(pose=self._object_state_history[0][:, 0:7], size=self._object_dims.size)
            self.obs_buf[:, start_offset:end_offset] = compute_projected_points(self.translation_from_camera_to_object, current_keypoints, self.camera_matrix, self.device).reshape(self.num_envs, 16)[:]

            # NOTE (dlee): temporary test
            # init_keypoints = torch.tensor([171.0, 153.0, 138.0, 147.0, 171.0, 147.0, 133.0, 139.0, 158.0, 190.0, 126.0, 182.0, 162.0, 190.0, 128.0, 180.0], device='cuda:0')
            # self.obs_buf[:, start_offset:end_offset] = init_keypoints

            # 2D projected goal keypoints
            start_offset = end_offset
            end_offset = start_offset + self._dims.TwoDimensionKeypointDim.value * self.keypoints_num
            goal_keypoints = gen_keypoints(pose=self._object_goal_poses_buf, size=self._object_dims.size)
            self.obs_buf[:, start_offset:end_offset] = compute_projected_points(self.translation_from_camera_to_object, goal_keypoints, self.camera_matrix, self.device).reshape(self.num_envs, 16)[:]

            # NOTE (dlee): temporary test
            # goal_keypoints = torch.tensor([136.79872, 89.090065, 135.30827, 124.140144, 170.82927, 89.090065, 171.52495, 124.140144, 140.09914, 104.3515, 139.01251, 134.97673, 169.28879, 104.3515, 169.79597, 134.97673], device='cuda:0')
            # self.obs_buf[:, start_offset:end_offset] = goal_keypoints

            # hand pose
            start_offset = end_offset
            end_offset = start_offset + self._dims.ObjectPoseDim.value
            ee_poses = self._rigid_body_state[:, self._tool_handle, :7].clone()

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

        # keypoint check
        # img = self._camera_image[0, ..., :3].clone().detach().cpu().numpy()
        # im = Image.fromarray(img)

        # current_keypoints = gen_keypoints(pose=self._object_state_history[0][:, 0:7], size=self._object_dims.size)
        # projected_keypoints = compute_projected_points(self.translation_from_camera_to_object, current_keypoints, self.camera_matrix, self.device).reshape(self.num_envs, 8, 2)[0, :]

        # import cv2
        # for idx in range(8):
        #     image_numpy = cv2.circle(img, (int(projected_keypoints[idx, 0]), int(projected_keypoints[idx, 1])), 3, (0, 0, 255), -1)
        # # keyp_img = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)

        # import matplotlib.pyplot as plt
        # plt.imshow(image_numpy)
        # plt.show()
        
        # # im.save('IG_d455.jpg')
        # print("debug")
        
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

        if self.cfg['env']["position_history"]:
            self.dof_position_history[:, reset_env_indices] = (self.dof_position[reset_env_indices].unsqueeze(0)).repeat(self.n_history, 1, 1)

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
        self.table_pose[reset_env_indices, :] = 0.
        self.table_pose[reset_env_indices, 0] = table_pose[0]
        self.table_pose[reset_env_indices, 1] = table_pose[1]

        if self.env_randomize:
            self.table_pose[reset_env_indices, 2] = torch.tensor(self.table_z_randomizer(self.randomization_params, reset_env_indices),  dtype=torch.float32, device=self.device)
        else:
            self.table_pose[reset_env_indices, 2] = table_pose[2]

        self.table_pose[reset_env_indices, 3:7] = torch.tensor([0., 0., 0., 1.], device=self.device)

        # root actor buffer
        self._actors_root_state[table_indices] = self.table_pose[reset_env_indices, :]
        
        if True: # only for bump domain
            if self.env_randomize:
                bump_indices = self.actor_indices["bump"][reset_env_indices]
                self._actors_root_state[bump_indices, 2] = self._actors_root_state[bump_indices, 2] + (self.table_pose[reset_env_indices, 2] - table_params["z"])

    def _sample_object_poses(self, reset_env_indices: torch.Tensor):
        """Sample poses for the object.

        Args:
            reset_env_indices: A tensor contraining indices of environment instances to reset.
        """
        object_indices = self.actor_indices["object"][reset_env_indices]
        self._object_state_history[0][reset_env_indices, :7] = self._initial_object_pose_buffer[self._init_data_indices[reset_env_indices]]
        self._object_state_history[0][reset_env_indices, 7:13] = 0

        if self.env_randomize:
            table_params = self.cfg["env"]["geometry"]["table"]
            self._object_state_history[0][reset_env_indices, 2] = self._object_state_history[0][reset_env_indices, 2] + (self.table_pose[reset_env_indices, 2] - table_params["z"])

        # reset object state history
        for idx in range(1, self._state_history_len):
            self._object_state_history[idx][reset_env_indices] = 0.0
        # root actor buffer
        self._actors_root_state[object_indices] = self._object_state_history[0][reset_env_indices]
        if self.cfg['env']["student_obs"] or self.cfg['env']["enableCameraSensors"]:
            self._object_initial_poses_buf[reset_env_indices, :] = self._object_state_history[0][reset_env_indices, :7]

    def _sample_object_goal_poses(self, reset_env_indices: torch.Tensor):
        """Sample goal poses for the object and sets them into the desired goal pose buffer.

        Args:
            reset_env_indices: A tensor contraining indices of environments to reset.
        """
        table_dims = self.cfg["env"]["geometry"]["table"]
        table_dim = [table_dims["width"], table_dims["length"], table_dims["height"]]
        object_dims = self.cfg["env"]["geometry"]["object"]
        object_dim = [object_dims["width"], object_dims["length"], object_dims["height"]]

        self._object_goal_poses_buf[reset_env_indices] = self._goal_buffer[self._init_data_indices[reset_env_indices]]
        if self.env_randomize:
            table_params = self.cfg["env"]["geometry"]["table"]
            self._object_goal_poses_buf[reset_env_indices, 2] = self._object_goal_poses_buf[reset_env_indices, 2] + (self.table_pose[reset_env_indices, 2] - table_params["z"])

        if not self.cfg['env']["enableCameraSensors"]:
            goal_object_indices = self.actor_indices["goal_object"][reset_env_indices]
            self._actors_root_state[goal_object_indices, 0:7] = self._object_goal_poses_buf[reset_env_indices]

    def get_initial_joint_position(self, num_samples: int) -> torch.Tensor:
        q_R = torch.zeros((num_samples, 6), dtype=torch.float, device=self.rl_device)

        return q_R

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
