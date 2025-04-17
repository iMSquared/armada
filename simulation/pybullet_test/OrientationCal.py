import numpy as np
import math
import torch
from scipy.spatial.transform import Rotation

class OrientationCal:
    def __init__(self):
        self.desired_quat = torch.zeros(4)
        self.current_quat = torch.zeros(4)

    def quaternion_conjugate(self, q):
        assert q.shape[-1] == 4
        conj = torch.tensor([1, -1, -1, -1], device=q.device)  
        return q * conj.expand_as(q)

    def quaternion_normalize(self, q):
        assert q.shape[-1] == 4
        norm = torch.sqrt(torch.sum(torch.square(q), dim=-1))  # ||q|| = sqrt(w²+x²+y²+z²)
        # assert not torch.any(torch.isclose(norm, torch.zeros_like(norm, device=q.device)))  # check for singularities
        return  norm  # q_norm = q / ||q||

    def quaternion_mul(self, q1, q2):
        assert q1.shape[-1] == 4
        assert q2.shape[-1] == 4
        original_shape = q1.shape

        # Compute outer product
        terms = torch.bmm(q2.view(-1, 4, 1), q1.view(-1, 1, 4))
        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
        y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
        z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]

        return torch.stack((x, y, z, w), dim=1).view(original_shape)
    
    def axis_angle_from_quat(self, quat: torch.Tensor):
        """Convert tensor of quaternions to tensor of axis-angles."""

        a_a = Rotation.from_quat(quat)
        axis_angle = a_a.as_rotvec()

        # test = Rotation.from_quat(quat)
        # test = test.as_euler('xyz', degrees=True)
        # print(f'Desired quat -> Euler Angle ={test}')

        return axis_angle
    

    def quaternion_cal_error(self, desired_quat, current_quat):
        assert desired_quat.shape[-1] == 4
        assert current_quat.shape[-1] == 4

        if isinstance(desired_quat, np.ndarray):
            desired_quat = torch.Tensor(desired_quat)

        if isinstance(current_quat, np.ndarray):
            current_quat = torch.Tensor(current_quat)


        desired_quat_last_value = desired_quat[-1]
        desired_quat_remaining_tensor = desired_quat[:-1]
        desired_quat = torch.cat((desired_quat_last_value.unsqueeze(0), desired_quat_remaining_tensor))
        current_quat_last_value = current_quat[-1]
        current_quat_remaining_tensor = current_quat[:-1]
        current_quat = torch.cat((current_quat_last_value.unsqueeze(0), current_quat_remaining_tensor))



        # quat_norm = self.quaternion_mul(current_quat, self.quaternion_conjugate(current_quat))
        quat_norm = self.quaternion_normalize(current_quat)
        # quat_norm = 1
        quat_inv = self.quaternion_conjugate(current_quat) / quat_norm
        quat_error = self.quaternion_mul(desired_quat, quat_inv)

        axis_angle_error = self.axis_angle_from_quat(quat_error)
        return axis_angle_error
    
    def quaternion_cal_Add(self, q1, q2):
        assert q1.shape[-1] == 4
        assert q2.shape[-1] == 4
        
        if isinstance(q1, np.ndarray):
            q1 = torch.Tensor(q1)

        if isinstance(q2, np.ndarray):
            q2 = torch.Tensor(q2)


        q1_quat_last_value = q1[-1]
        q1_quat_remaining_tensor = q1[:-1]
        q1 = torch.cat((q1_quat_last_value.unsqueeze(0), q1_quat_remaining_tensor))
        q2_last_value = q2[-1]
        q2_quat_remaining_tensor = q2[:-1]
        q2 = torch.cat((q2_last_value.unsqueeze(0), q2_quat_remaining_tensor))

        quat_sum = self.quaternion_mul(q1, q2)
        axis_angle_sum = self.axis_angle_from_quat(quat_sum)

        return quat_sum, axis_angle_sum
