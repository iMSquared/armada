from typing import Union, Tuple
from enum import Enum
import torch
import numpy as np
import numpy.typing as npt
from imitation.utils.transforms import axis_angle_to_quaternion, quaternion_multiply, quaternion_to_axis_angle
from loguru import logger

class SMPLindex(Enum):
    PELVIS = (0, 1, 2)
    L_HIP = (3, 4, 5)
    R_HIP = (6, 7, 8)
    SPINE1 = (9, 10, 11)
    L_KNEE = (12, 13, 14)
    R_KNEE = (15, 16, 17)
    SPINE2 = (18, 19, 20)
    L_ANKLE = (21, 22, 23)
    R_ANKLE = (24, 25, 26)
    SPINE3 = (27, 28, 29)
    L_FOOT = (30, 31, 32)
    R_FOOT = (33, 34, 35)
    NECK = (36, 37, 38)
    L_COLLAR = (39, 40, 41)
    R_COLLAR = (42, 43, 44)
    HEAD = (45, 46, 47)
    L_SHOULDER = (48, 49, 50)
    R_SHOULDER = (51, 52, 53)
    L_ELBOW = (54, 55, 56)
    R_ELBOW = (57, 58, 59)
    L_WRIST = (60, 61, 62)
    R_WRIST = (63, 64, 65)
    L_HAND = (66, 67, 68)
    R_HAND = (69, 70, 71)


def extract_pelvis_traj(smpl_pose_traj: torch.Tensor, return_torch:bool=True):
    '''
    Pelvis is the root of all joints
    Computed in Quaternion
    '''
    pelvis_traj = axis_angle_to_quaternion(smpl_pose_traj[:,SMPLindex.PELVIS.value])

    return pelvis_traj if return_torch else pelvis_traj.detach().numpy()


def extract_shoulder_base(smpl_pose_traj: torch.Tensor, 
                          arm: str, 
                          return_torch:bool=True) -> Union[torch.Tensor, npt.NDArray]:
    '''
    Computed in Quaternion
    '''

    assert arm.lower() in ['left', 'right'], 'Must provide which side of the arm in [left, right]'

    spine1_traj = axis_angle_to_quaternion(smpl_pose_traj[:,SMPLindex.SPINE1.value])
    spine2_traj = axis_angle_to_quaternion(smpl_pose_traj[:,SMPLindex.SPINE2.value])
    spine3_traj = axis_angle_to_quaternion(smpl_pose_traj[:,SMPLindex.SPINE3.value])
    if arm.lower() == 'left':
        collar_traj = axis_angle_to_quaternion(smpl_pose_traj[:,SMPLindex.L_COLLAR.value])
    else:
        collar_traj = axis_angle_to_quaternion(smpl_pose_traj[:,SMPLindex.R_COLLAR.value])

    base_traj = quaternion_multiply(collar_traj, spine3_traj)
    base_traj = quaternion_multiply(base_traj, spine2_traj)
    # base_traj = quaternion_multiply(spine3_traj, spine2_traj)
    base_traj = quaternion_multiply(base_traj, spine1_traj)

    return base_traj if return_torch else base_traj.detach().numpy()


def extract_shoulder_local_traj(smpl_pose_traj: torch.Tensor, 
                                arm: str, 
                                return_torch:bool=True) -> Union[torch.Tensor, npt.NDArray]:

    assert arm.lower() in ['left', 'right'], 'Must provide which side of the arm in [left, right]'

    if arm.lower() == 'left':
        shoulder_local_traj = axis_angle_to_quaternion(smpl_pose_traj[:,SMPLindex.L_SHOULDER.value])
    else:
        shoulder_local_traj = axis_angle_to_quaternion(smpl_pose_traj[:,SMPLindex.R_SHOULDER.value])

    return shoulder_local_traj if return_torch else shoulder_local_traj.detach().numpy()


def extract_shoulder_traj(smpl_pose_traj: torch.Tensor, 
                          arm: str, 
                          return_torch:bool=True) -> Union[Tuple[npt.NDArray, npt.NDArray, npt.NDArray], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

    assert arm.lower() in ['left', 'right'], 'Must provide which side of the arm in [left, right]'

    base_traj = extract_shoulder_base(smpl_pose_traj, arm)
    shoulder_local_traj = extract_shoulder_local_traj(smpl_pose_traj, arm)

    shoulder_traj = quaternion_multiply(base_traj, shoulder_local_traj)

    if return_torch:
        return shoulder_traj, shoulder_local_traj, base_traj
    
    else:
        return shoulder_traj.detach().numpy(), shoulder_local_traj.detach().numpy(), base_traj.detach().numpy()


def extract_elbow_local_traj(smpl_pose_traj: torch.Tensor, 
                             arm: str, 
                             return_torch:bool=True) -> Union[npt.NDArray, torch.Tensor]:

    assert arm.lower() in ['left', 'right'], 'Must provide which side of the arm in [left, right]'

    if arm.lower() == 'left':
        elbow_local_traj = axis_angle_to_quaternion(smpl_pose_traj[:,SMPLindex.L_ELBOW.value])
    else:
        elbow_local_traj = axis_angle_to_quaternion(smpl_pose_traj[:,SMPLindex.R_ELBOW.value])

    return elbow_local_traj if return_torch else elbow_local_traj.detach().numpy()


def extract_elbow_traj(smpl_pose_traj: torch.Tensor, 
                       arm: str, 
                       shoulder_traj: Union[npt.NDArray, torch.Tensor, None]=None, 
                       return_torch:bool=True) -> Union[Tuple[npt.NDArray, npt.NDArray, npt.NDArray], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

    assert arm.lower() in ['left', 'right'], 'Must provide which side of the arm in [left, right]'

    if shoulder_traj is None:
        shoulder_traj, _, _ = extract_shoulder_traj(smpl_pose_traj, arm)

    if not isinstance(shoulder_traj, torch.Tensor):
        shoulder_traj = torch.tensor(shoulder_traj)

    elbow_local_traj = extract_elbow_local_traj(smpl_pose_traj, arm)

    elbow_traj = quaternion_multiply(shoulder_traj, elbow_local_traj)

    if return_torch:
        return elbow_traj, elbow_local_traj, shoulder_traj
    else:
        return elbow_traj.detach().numpy(), elbow_local_traj.detach().numpy(), shoulder_traj.detach().numpy()



def extract_wrist_local_traj(smpl_pose_traj: torch.Tensor,
                             arm: str,
                             return_torch:bool=True) -> Union[npt.NDArray, torch.Tensor]:
    
    assert arm.lower() in ['left', 'right'], 'Must provide which side of the arm in [left, right]'

    if arm.lower() == 'left':
        wrist_local_traj = axis_angle_to_quaternion(smpl_pose_traj[:,SMPLindex.L_WRIST.value])
    else:
        wrist_local_traj = axis_angle_to_quaternion(smpl_pose_traj[:,SMPLindex.R_WRIST.value])

    return wrist_local_traj if return_torch else wrist_local_traj.detach().numpy()


def extract_wrist_traj(smpl_pose_traj: npt.NDArray,
                       arm:str,
                       elbow_traj:Union[npt.NDArray, torch.Tensor, None]=None,
                       return_torch:bool=True) -> Union[Tuple[npt.NDArray, npt.NDArray, npt.NDArray], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    
    assert arm.lower() in ['left', 'right'], 'Must provide which side of the arm in [left, right]'

    if elbow_traj is None:
        elbow_traj, _, _ = extract_elbow_traj(smpl_pose_traj, arm)

    if not isinstance(elbow_traj, torch.Tensor):
        elbow_traj = torch.tensor(elbow_traj)

    wrist_local_traj = extract_wrist_local_traj(smpl_pose_traj, arm)

    wrist_traj = quaternion_multiply(elbow_traj, wrist_local_traj)

    if return_torch:
        return wrist_traj, wrist_local_traj, wrist_local_traj
    else:
        return wrist_traj.detach().numpy(), wrist_local_traj.detach().numpy(), wrist_local_traj.detach().numpy()
