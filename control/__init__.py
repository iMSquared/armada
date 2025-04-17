from enum import IntEnum
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Dict, Tuple

class JointNames(IntEnum):
    L_WRIST_YAW = 5
    L_WRIST_PITCH = 6
    R_WRIST_YAW = 11
    R_WRIST_PITCH = 12

class CANIDs(IntEnum):
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 7
    R1 = 11
    R2 = 12
    R3 = 13
    R4 = 14

class CANSockets():
    L_TMOTOR = 'can_arm'
    L_YAW = 'can_lw_yaw'
    L_PITCH = 'can_lw_pitch'
    R_TMOTOR = 'can_rarm'
    R_YAW = 'can_rw_yaw'
    R_PITCH = 'can_rw_pitch'

@dataclass
class DefaultControllerValues:
    # Naming
    LEFT = 'left'
    RIGHT = 'right'
    BIMANUAL = 'both'


    # Communication
    PORT: str = '50051'
    IP: str = '[::]'

    # URDF paths
    LEFT_URDF_PATH: str = 'simulation/assets/urdf/RobotBimanualV5/urdf/Simplify_Robot_plate_gripper_LEFT.urdf'
    # LEFT_URDF_PATH: str = 'simulation/assets/urdf/RobotBimanualV7/urdf/Simplify_Robot_real_hammer.urdf'
    RIGHT_URDF_PATH: str = 'simulation/assets/urdf/RobotBimanualV5/urdf/Simplify_Robot_plate_gripper_RIGHT.urdf'

    EE_NAMES = {LEFT: 'tool1', RIGHT: 'tool2'}
    EE_LINK = {LEFT: 'link6', RIGHT: 'link12'}

    # Kinematics
    DOF: int = 6
    DOF_ID_OFFSET = 7
    
    # Joint limits and offset
    L_JOINT_LIMIT_MIN: npt.NDArray = np.pi/180.*np.array([-90.,  -90., -180.,  -45., -210., -125.]) 
    L_JOINT_LIMIT_MAX: npt.NDArray = np.pi/180.*np.array([50.,  80., 80.,  90., 210. , 125.])
    R_JOINT_LIMIT_MIN: npt.NDArray = np.pi/180.*np.array([-50.,  -80., -80.,  -90., -210., -125.]) 
    R_JOINT_LIMIT_MAX: npt.NDArray = np.pi/180.*np.array([90.,  90., 180.,  45., 210. , 125.])

    JOINT_INIT_OFFSET: npt.NDArray = np.pi/180.*np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    L_JOINT_SMPL_OFFSET: npt.NDArray = np.pi/180*np.array([0, 90, 0, 90, -10, 20])
    R_JOINT_SMPL_OFFSET: npt.NDArray = np.pi/180*np.array([0, 90, 0, -90, 10, -20])
    

    GRIPPER_SIM_MIN: npt.NDArray = np.pi/180*np.array([-270,])
    GRIPPER_SIM_MAX: npt.NDArray = np.pi/180*np.array([270,])
    L_GRIPPER: Tuple[float] = (76, 126)
    R_GRIPPER: Tuple[float] = (142, 192)


    JOINT_VEL_UPPER_LIMIT = np.array([10., 10., 10., 10., 15., 15.])


    # Gains
    DEFAULT_P_GAIN: npt.NDArray = np.array([1.8, 2.4, 1.9, 1.8, 0.6, 0.7])
    DEFAULT_D_GAIN: npt.NDArray = np.array([0.04, 0.06, 0.06, 0.06, 0.01, 0.01])
    POS_P_GAIN: npt.NDArray = np.array([1.8, 2.2, 1.6, 1.6, 0.7, 0.7])*50
    POS_D_GAIN: npt.NDArray = np.array([0.02, 0.02, 0.02, 0.02, 0.01, 0.01])*37
    HAMMER_P_GAIN: npt.NDArray = np.array([1.8, 2.2, 1.6, 1.6, 0.7, 0.7])*10
    HAMMER_D_GAIN: npt.NDArray = np.array([0.02, 0.02, 0.02, 0.02, 0.01, 0.01])*7
    SNATCH_P_GAIN: npt.NDArray = np.array([1.8, 2.2, 1.6, 1.6, 0.7, 0.7])*10
    SNATCH_D_GAIN: npt.NDArray = np.array([0.02, 0.02, 0.02, 0.02, 0.01, 0.01])*7
    SHADOW_P_GAIN: npt.NDArray = np.array([1.8, 2.2, 1.6, 1.6, 0.7, 0.7])*5
    SHADOW_D_GAIN: npt.NDArray = np.array([0.02, 0.02, 0.02, 0.02, 0.01, 0.01])*3
    
    # Torque limit
    MAX_TAU = np.array([4., 4., 4, 4, 1., 1.])