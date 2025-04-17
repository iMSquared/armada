from typing import Union, Dict, List, Tuple
from enum import IntEnum
import torch
import numpy as np
import numpy.typing as npt
from imitation.utils.transforms import matrix_to_quaternion, ensure_quaternion_consistency
from torch.nn import functional as F

class MANOIndex(IntEnum):
    WRIST = 0
    J11 = 1
    J12 = 2
    J13 = 3
    J14 = 4
    J21 = 5
    J22 = 6
    J23 = 7
    J24 = 8
    J31 = 8
    J32 = 10
    J33 = 11
    J34 = 12
    J41 = 13
    J42 = 14
    J43 = 15
    J44 = 16
    J51 = 17
    J52 = 18
    J53 = 19
    J54 = 20


class NoHandPredictionException(Exception):
    def __init__(self):
        super().__init__('We lost track of hand pose')


def extract_hand_global_traj(mano_global_traj: torch.Tensor, 
                             return_torch:bool=True) -> Union[torch.Tensor, npt.NDArray]:

    hand_global_traj = matrix_to_quaternion(mano_global_traj)
    # hand_global_traj = mano_global_traj

    return hand_global_traj if return_torch else hand_global_traj.detach().numpy()


def extract_finger_traj(mano_finger_traj: torch.Tensor, 
                        return_torch:bool=True) -> Union[torch.Tensor, npt.NDArray]:

    finger_global_traj = matrix_to_quaternion(mano_finger_traj)
    # finger_global_traj = mano_finger_traj

    return finger_global_traj if return_torch else finger_global_traj.detach().numpy()


def gaussian_smooth(data: torch.Tensor, window_size: int=30, sigma:float=3.0):
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
    kernel = kernel.view(1, -1)  # Shape: (1, 1, window_size)
    
    # Expand kernel to match the number of channels
    kernel = kernel.expand(data.size(2), -1, -1)  # Shape: (D, 1, window_size)
    
    # Apply convolution along the second dimension
    smoothed_data = F.conv1d(
        data.permute(0, 2, 1),  # Permute to (B, D, T)
        kernel,  # Expanded kernel
        padding=(window_size - 1) // 2 + 1,  # Dynamically adjust padding
        groups=data.size(2)  # Ensure channel-wise convolution
    ).permute(0, 2, 1)  # Return to original shape (B, T, D)
    
    # Truncate to original size if necessary
    if smoothed_data.shape[1] > data.shape[1]:
        smoothed_data = smoothed_data[:, :data.shape[1], :]
    
    return smoothed_data


def extract_grasp_from_keypoints(hand_keypoints_3d: torch.Tensor, threshold: float=0.025):
    finger_dist = extract_finger_dist_from_keypoints(hand_keypoints_3d)

    # smoothed_finger_dist = gaussian_smooth(finger_dist.view(1, -1, 1), window_size=30, sigma=3.0).squeeze()
    smoothed_finger_dist = finger_dist

    grasp = smoothed_finger_dist < threshold
    
    return grasp


def extract_finger_dist_from_keypoints(hand_keypoints_3d: torch.Tensor):
    finger_dist = hand_keypoints_3d[:, MANOIndex.J14, :] - hand_keypoints_3d[:, MANOIndex.J24, :]
    finger_dist = torch.linalg.norm(finger_dist, axis=1)

    return finger_dist