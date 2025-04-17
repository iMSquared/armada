import sys
import math
import torch
from pathlib import Path

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from simulation.utils.torch_jit_utils import *
from typing import Deque, Dict, Tuple, List, Union
from scipy.spatial.transform import Rotation as R
from simulation.im2gym import *

JOINT_RES_START = {LEFT: L_JOINT_RES_START, RIGHT: R_JOINT_RES_START}
JOINT_P_START = {LEFT: L_JOINT_P_START, RIGHT: R_JOINT_P_START}
JOINT_D_START = {LEFT: L_JOINT_D_START, RIGHT: R_JOINT_P_START}

BASEDIR = Path(__file__).parent.parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))

class RobotControl():
    def __init__(self, 
                 config: Dict, 
                 sim_device: str, 
                 graphics_device_id: str,
                 j_eef: Dict[str, torch.Tensor]=None):
        self.cfg = config

        split_device = sim_device.split(":")
        self.device_type = split_device[0]
        self.device_id = graphics_device_id

        self.device = "cpu"
        if config["sim"]["use_gpu_pipeline"]:
            if self.device_type.lower() == "cuda" or self.device_type.lower() == "gpu":
                self.device = "cuda" + ":" + str(self.device_id)
            else:
                print("GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline.")
                config["sim"]["use_gpu_pipeline"] = False

        # Jacobian
        self.j_eef = j_eef

    
    def compute_torque(self,
                       arm: List[str],
                       desired_joint_position: torch.Tensor,
                       dof_position: torch.Tensor,
                       dof_velocity: torch.Tensor,
                       action_transformed: torch.Tensor):
        
        l_p_gains = action_transformed[:, L_JOINT_P_START:L_JOINT_P_START+DOF]
        l_d_gains = action_transformed[:, L_JOINT_D_START:L_JOINT_D_START+DOF]
        r_p_gains = action_transformed[:, R_JOINT_P_START:R_JOINT_P_START+DOF]
        r_d_gains = action_transformed[:, R_JOINT_D_START:R_JOINT_D_START+DOF]

        p_gains = {LEFT: l_p_gains, RIGHT: r_p_gains}
        d_gains = {LEFT: l_d_gains, RIGHT: r_d_gains}
        
        ct = torch.zeros_like(dof_position)
        joint_position_error = desired_joint_position - dof_position

        for name in arm:
            ct[:, JOINT_FIRST[name]-1:JOINT_LAST[name]] = \
                p_gains[name] * joint_position_error[:, JOINT_FIRST[name]-1:JOINT_LAST[name]] \
                - d_gains[name] * dof_velocity[:, JOINT_FIRST[name]-1:JOINT_LAST[name]]


        return ct
    

    def compute_ee_target(self,
                          arm: str, 
                          rigid_body_state: torch.Tensor,
                          tool_handle: torch.Tensor,
                          action_transformed: torch.Tensor,
                          env_ids: torch.Tensor) -> Tuple[torch.Tensor]:
        """compute the endeffector target to run the controller"""
        sub_goal_pos = rigid_body_state[:, tool_handle, :3] + action_transformed[:, :3]
        rot_residual = self.axisaToquat(action_transformed[:, 3:6])
        rot_residual[env_ids, 0:3] = 0.0
        rot_residual[env_ids, -1] = 1.0
        sub_goal_rot = quat_mul(rot_residual, rigid_body_state[:, tool_handle, 3:7])

        return sub_goal_pos, sub_goal_rot
    

    def compute_joint_target(self,
                             arm: str, 
                             action_transformed: torch.Tensor,
                             env_ids: torch.Tensor,
                             dof_position: torch.Tensor,
                             dof_lower_limits: torch.Tensor,
                             dof_upper_limits: torch.Tensor) -> torch.Tensor: 
        '''compute the joint target to run the controller'''

        if self.cfg['env']['controller'] == 'position':
            delta_joint_position = action_transformed[:, JOINT_RES_START[arm]:JOINT_RES_START[arm]+DOF]

        elif self.cfg['env']['controller'] == 'position_2':
            delta_mul_kp = action_transformed[:, JOINT_RES_START[arm]:JOINT_RES_START[arm]+DOF]
            kp = action_transformed[:, JOINT_P_START[arm]:JOINT_P_START[arm]+DOF]
            delta_joint_position = delta_mul_kp / kp

        else:
            assert self.j_eef is not None, 'Must provide Jacobian matrix to use this control'
            delta_joint_position = self._get_delta_dof_pos(delta_pose=action_transformed[:, JOINT_RES_START[arm]:JOINT_RES_START[arm]+DOF], 
                                                           ik_method='dls',
                                                           jacobian=self.j_eef[arm].clone(), 
                                                           device=self.device)
        
        delta_joint_position[env_ids, :] = 0
        desired_joint_position = dof_position[:, JOINT_FIRST[arm]-1:JOINT_LAST[arm]] + delta_joint_position

        # TODO: change to 10% -> 5%
        dof_lower_safety_limit = 0.95*dof_lower_limits[JOINT_FIRST[arm]-1:JOINT_LAST[arm]] + 0.05*dof_upper_limits[JOINT_FIRST[arm]-1:JOINT_LAST[arm]]
        dof_upper_safety_limit = 0.05*dof_lower_limits[JOINT_FIRST[arm]-1:JOINT_LAST[arm]] + 0.95*dof_upper_limits[JOINT_FIRST[arm]-1:JOINT_LAST[arm]]

        desired_joint_position = torch.clamp(desired_joint_position, dof_lower_safety_limit, dof_upper_safety_limit)

        return desired_joint_position


    def step_cartesian_impedance(self, 
                                 sub_goal_pos: torch.Tensor,
                                 sub_goal_rot: torch.Tensor,
                                 rigid_body_state: torch.Tensor,
                                 hand_handle: torch.Tensor,
                                 action_transformed: torch.Tensor):
        pos_err = sub_goal_pos - rigid_body_state[:, hand_handle, :3]
        orn_err = orientation_error(sub_goal_rot, rigid_body_state[:, hand_handle, 3:7], 3)
        err = torch.cat([pos_err, orn_err], -1)
        
        hand_vel = rigid_body_state[:, hand_handle, 7:]

        kp = action_transformed[:, 6:12]
        kd = 2 * torch.sqrt(kp) * action_transformed[:, 12:18]
        xddt = (kp * err - kd * hand_vel).unsqueeze(-1)
        u = torch.transpose(self.j_eef, 1, 2) @ xddt
        
        return u.squeeze(-1)


    def _get_delta_dof_pos(self, 
                           delta_pose: torch.Tensor, 
                           ik_method: str,
                           jacobian: torch.Tensor,
                           device: str):
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
    




    # Helpers
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