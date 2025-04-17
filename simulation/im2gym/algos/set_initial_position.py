from pybullet_utils import bullet_client
import pybullet_data
import pybullet
import torch
import pickle
import numpy as np

    
class setFixedInitPosition():
    def __init__(self, init_position):
        self.jointPoses = torch.tensor(init_position, dtype=torch.float, device='cuda')

    def fixed_init_pos(self, env_size): 
        q_R = self.jointPoses.repeat(env_size, 1)
        return q_R

class PreCalculatedIK():

    def __init__(self, pickle_file='simulation/im2gym/bump_robot_init_config.pkl'):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            self.ee_pos = torch.tensor(data['ee_poses'], dtype=torch.float, device='cuda')
            self.joint_pos = torch.tensor(data['thetas'], dtype=torch.float, device='cuda')

    def pre_calculated_IK(self, query_EE_pos: torch.Tensor):
        # Calculate the Euclidean distances between query and pre-calculated positions
        distances = torch.cdist(query_EE_pos, self.ee_pos, p=2)
        
        # Find the indices of the minimum distances along axis 1
        min_indices = torch.argmin(distances, dim=1)
        
        # Gather the corresponding joint positions
        q_R = self.joint_pos[min_indices]
        
        return q_R
                        
class PreCalculatedRandom():

    def __init__(self, pickle_file='simulation/im2gym/bump_robot_init_config.pkl'):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            self.joint_pos = torch.tensor(data['thetas'], dtype=torch.float, device='cuda')
            self.num_thetas = self.joint_pos.shape[0]

    def pre_calculated_random(self, n_env):
        ran_indices = torch.randint(0, self.num_thetas, (n_env,), device='cuda')
        
        # Gather the corresponding joint positions
        q_R = self.joint_pos[ran_indices]
        
        return q_R
                        



