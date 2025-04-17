from torch.distributions import Categorical, Uniform
import torch.nn as nn
import torch
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

class DiagGaussianPd:
    def __init__(self):
        self.const = 0.5 * math.log(2 * math.pi)

    def sample(self, mean, logstd):
        eps = torch.normal(mean=0.0, std=1.0, size=mean.shape, device=mean.device)
        return mean + eps * torch.exp(logstd)

    def neglogp(self, mean, logstd, action):
        std = torch.exp(logstd)
        return self.const * mean.shape[-1] + 0.5 * torch.sum((logstd * 2 + torch.square((mean - action) / std)), dim=-1) 

    def entropy(self):
        return torch.tensor(0.0, device='cuda:0')


class RunningMeanStd(nn.Module):
    def __init__(self, insize, rl_device, epsilon=1e-05):
        super(RunningMeanStd, self).__init__()
        self.epsilon = epsilon
        self.axis = [0]

        self.register_buffer("running_mean", torch.zeros(insize, dtype=torch.float64, device=rl_device))
        self.register_buffer("running_var", torch.ones(insize, dtype=torch.float64, device=rl_device))
        self.register_buffer("count", torch.ones((), dtype=torch.float64, device=rl_device))

    def _update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count

    def forward(self, input, unnorm=False):
        if self.training:
            mean = input.mean(self.axis)
            var = input.var(self.axis)
            self.running_mean, self.running_var, self.count = self._update_mean_var_count_from_moments(
                self.running_mean, self.running_var, self.count, mean, var, input.size()[0]
            )

        current_mean = self.running_mean
        current_var = self.running_var

        if unnorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon) * y + current_mean.float()
        else:
            y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
            y = torch.clamp(y, min=-5.0, max=5.0)

        return y


class Policy(nn.Module):
    def __init__(self, input_size, action_size, units, normalize_input, normalize_value, asymmetric_obs=False):
        super(Policy, self).__init__()
        self.normalize_input = normalize_input
        self.normalize_value = normalize_value
        in_dim = input_size
        layers = []
        for unit in units:
            layers.append(nn.Linear(in_dim, unit))
            layers.append(nn.ELU())
            in_dim = unit
        self.common_network = nn.Sequential(*layers)
        self.action_network = nn.Linear(in_dim, action_size)
        self.asymmetric_obs=asymmetric_obs
        if not asymmetric_obs:
            self.value_network = nn.Linear(in_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(action_size, requires_grad=True, dtype=torch.float32), requires_grad=True)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.log_std, 0.0)
        if normalize_input:
            self.running_mean_std = RunningMeanStd(input_size, 'cuda:0')
        if normalize_value:
            self.value_mean_std = RunningMeanStd(1, 'cuda:0')

    def forward(self, observation):
        observation = self.norm_obs(observation)
        latent = self.common_network(observation)
        mu = self.action_network(latent)
        logstd = mu * 0.0 + self.log_std # mu * 0.0 is needed to match the dimension 

        value = self.value_network(latent)
        if not self.training:
            value = self.unnorm_value(value)
        value = torch.squeeze(value)
        return mu, logstd, value

    def norm_obs(self, observation):
        return self.running_mean_std(observation) if self.normalize_input else observation

    def unnorm_value(self, value):
        return self.value_mean_std(value, unnorm=True) if self.normalize_value else value

class Value(nn.Module):
    def __init__(self, input_size,units, normalize_input, normalize_value):
        super(Value, self).__init__()
        self.normalize_input = normalize_input
        self.normalize_value = normalize_value
        in_dim = input_size
        layers = []
        for unit in units:
            layers.append(nn.Linear(in_dim, unit))
            layers.append(nn.ELU())
            in_dim = unit
        self.common_network = nn.Sequential(*layers)
        self.value_network = nn.Linear(in_dim, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        if normalize_input:
            self.running_mean_std = RunningMeanStd(input_size, 'cuda:0')
        if normalize_value:
            self.value_mean_std = RunningMeanStd(1, 'cuda:0')

    def forward(self, observation):
        observation = self.norm_obs(observation)
        latent = self.common_network(observation)
        value = self.value_network(latent)
        if not self.training:
            value = self.unnorm_value(value)
        value = torch.squeeze(value)
        return value

    def norm_obs(self, observation):
        return self.running_mean_std(observation) if self.normalize_input else observation

    def unnorm_value(self, value):
        return self.value_mean_std(value, unnorm=True) if self.normalize_value else value


class Model:
    def __init__(self):
        normalize_input = True
        normalize_value = True
        mlp_config = [512, 256, 256, 128]
        obs_shape = 69 # 6+6+16+16+7+18 # env_cfg["num_obs"]

        # self.obs_spec = {
        #     # robot joint
        #     "robot_q": self._dims.GeneralizedCoordinatesDim.value, 6
        #     # robot joint velocity
        #     "robot_u": self._dims.GeneralizedVelocityDim.value, 6
        #     # object position represented as 2D kepoints
        #     "object_q": (self._dims.TwoDimensionKeypointDim.value * self.keypoints_num), 16
        #     # object goal position represented as 2D kepoints
        #     "object_q_des": (self._dims.TwoDimensionKeypointDim.value * self.keypoints_num), 16
        #     # hand pose
        #     "hand_state": self._dims.ObjectPoseDim.value, 7
        #     # previous action
        #     "command": self.action_dim 18

        action_shape = 18 # env_cfg["num_acts"]
        
        self.policy = Policy(obs_shape, action_shape, mlp_config, normalize_input, normalize_value, False)
        self.policy.to('cuda:0')
        self.dist = DiagGaussianPd()
        
    def step(self, observation):
        self.policy.value_mean_std.eval()
        with torch.no_grad():
            mu, logstd, value = self.policy(observation)
            action = self.dist.sample(mu, logstd)
            neglogp = self.dist.neglogp(mu, logstd, action)
        return action, value, neglogp, mu, logstd

    def value(self, observation):
        with torch.no_grad():
            _, _, value = self.policy(observation)
        return value

    def load(self, checkpoint):
        state = torch.load(checkpoint)
        post_state = state.pop('post')
        self.policy.load_state_dict(post_state)
        self.last_mean_reward = state.pop('last_mean_rewards')
        self.frames = state.pop('frames')

        print(f"restored running mean for value(count, mean, std):\
         {self.policy.value_mean_std.count} {self.policy.value_mean_std.running_mean} {self.policy.value_mean_std.running_var}")
        print(f"restored running mean for obs(count, mean, std):\
         {self.policy.running_mean_std.count} {self.policy.running_mean_std.running_mean} {self.policy.running_mean_std.running_var}")

def gen_keypoints(pose: torch.Tensor, num_keypoints: int = 8, size = (0.05, 0.07, 0.005)):

    num_envs = 1
    keypoints_buf = torch.ones(num_envs, num_keypoints, 3, dtype=torch.float32, device=pose.device)
    for i in range(num_keypoints):
        # which dimensions to negate
        n = [((i >> k) & 1) == 0 for k in range(3)]
        corner_loc = [(1 if n[k] else -1) * s / 2 for k, s in enumerate(size)],
        corner = torch.tensor(corner_loc, dtype=torch.float32, device=pose.device) * keypoints_buf[:, i, :]
        keypoints_buf[:, i, :] = local_to_world_space(corner, pose[None, ...])
    return keypoints_buf

def compute_projected_points(T_matrix: torch.Tensor, keypoints: torch.Tensor, camera_matrix: torch.Tensor, device: str, num_points: int=8):
    num_envs=1
    p_CO=torch.matmul(T_matrix, torch.cat([keypoints,torch.ones((num_envs, num_points,1),device=device)],-1).transpose(1,2))
    image_coordinates=torch.matmul(camera_matrix, p_CO).transpose(1,2)
    mapped_coordinates=image_coordinates[:,:,:2]/(image_coordinates[:,:,2].unsqueeze(-1))
    return mapped_coordinates

# @torch.jit.script
def local_to_world_space(pos_offset_local: torch.Tensor, pose_global: torch.Tensor):
    """ Convert a point from the local frame to the global frame
    Args:
        pos_offset_local: Point in local frame. Shape: [N, 3]
        pose_global: The spatial pose of this point. Shape: [N, 7]
    Returns:
        Position in the global frame. Shape: [N, 3]
    """
    quat_pos_local = torch.cat(
        [pos_offset_local, torch.zeros(pos_offset_local.shape[0], 1, dtype=torch.float32, device=pos_offset_local.device)],
        dim=-1
    )
    quat_global = pose_global[:, 3:7]
    quat_global_conj = quat_conjugate(quat_global)
    pos_offset_global = quat_mul(quat_global, quat_mul(quat_pos_local, quat_global_conj))[:, 0:3]

    result_pos_gloal = pos_offset_global + pose_global[:, 0:3]

    return result_pos_gloal

# @torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)

# @torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat

def scale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a given input tensor to a range of [-1, 1].

    @note It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape (dims,)
        upper: The maximum value of the tensor. Shape (dims,)

    Returns:
        Normalized transform of the tensor. Shape (N, dims)
    """
    # default value of center
    offset = (lower + upper) * 0.5
    # return normalized tensor
    return 2 * (x - offset) / (upper - lower)

def unscale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """
    Denormalizes a given input tensor from range of [-1, 1] to (lower, upper).

    @note It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape (dims,)
        upper: The maximum value of the tensor. Shape (dims,)

    Returns:
        Denormalized transform of the tensor. Shape (N, dims)
    """
    # default value of center
    offset = (lower + upper) * 0.5
    # return normalized tensor
    return x * (upper - lower) * 0.5 + offset

def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

def get_delta_dof_pos(delta_pose, jacobian):
    lambda_val = 0.1
    jacobian_T = np.transpose(jacobian)
    lambda_matrix = (lambda_val ** 2) * np.eye(jacobian.shape[0])
    delta_dof_pos = jacobian_T @ np.linalg.inv(jacobian @ jacobian_T + lambda_matrix) @ delta_pose
    return delta_dof_pos

def transform_pose(base_posquat, ee_posquat):
    # Extract position and orientation of the base in the world frame
    base_position = base_posquat[:3]
    base_orientation = base_posquat[3:]
    
    # Extract position and orientation of the end-effector in the robot base frame
    ee_position = ee_posquat[:3]
    ee_orientation = ee_posquat[3:]
    
    # Convert base orientation quaternion to rotation matrix
    base_rotation = R.from_quat(base_orientation).as_matrix()
    
    # Convert end-effector orientation quaternion to rotation matrix
    ee_rotation = R.from_quat(ee_orientation).as_matrix()
    
    # Calculate the end-effector position in the world frame
    ee_position_world = base_rotation @ ee_position + base_position
    
    # Calculate the end-effector orientation in the world frame
    ee_rotation_world = base_rotation @ ee_rotation
    
    # Convert the world frame rotation matrix back to a quaternion
    ee_orientation_world = R.from_matrix(ee_rotation_world).as_quat()
    
    # Combine the position and orientation into a single posquat vector
    ee_posquat_world = np.hstack((ee_position_world, ee_orientation_world))
    
    return ee_posquat_world