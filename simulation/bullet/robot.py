from typing import Tuple, List, Dict
import numpy.typing as npt
from pybullet_utils.bullet_client import BulletClient
import numpy as np
from simulation.utils.PinocchioInterface import PinocchioInterface
from control import DefaultControllerValues as RobotParams

JOINT_MAX = np.array([1.309, 0.5236, 0.3491, 1.0472, 2.0, 0.2, 2.0, 0.7854, 0.5236, 0.5708, 0.7854, 2.0, 3.0, 2.0])
JOINT_MIN = np.array([-0.7854, -0.5236, -.5708, -0.7854, -2.0, -3.0, -2.0, -1.309, -0.5236, -0.3491, -1.0472, -2.0, -0.2])
LEFT = 'left'
RIGHT = 'right'
BIMANUAL = 'both'
# EE_NAME = {LEFT: 'tool1', RIGHT: 'tool2', BIMANUAL: ['tool1', 'tool2']}
EE_NAME = {LEFT: 'tool1', RIGHT: 'tool2'}
MODE = 'pos'

COLOR_PALLETE = [
    [1., 0., 0., 1.],
    [0., 1., 0., 1.],
    [0., 0., 1., 1.],
    [0., 1., 1., 1.],
    [1., 0., 1., 1.],
    [1., 1., 0., 1.],
]

class Robot:
    def __init__(self, 
                 bc: BulletClient, 
                 start_pos: npt.ArrayLike, 
                 start_orn: npt.ArrayLike,
                 urdf_name: str,
                 arm: str='all',
                 ee_name: Dict[str, str]=EE_NAME,
                 pallete: List[List[float]]=COLOR_PALLETE, 
                 joint_min: npt.NDArray=JOINT_MIN,
                 joint_max: npt.NDArray=JOINT_MAX,
                 joint_id_offset: int=RobotParams.DOF_ID_OFFSET,
                 mode: str=MODE,
                 debug: bool=False, 
                 log: bool=False):
        
        # Bullet client
        self.bc = bc

        # EE
        self.arm = arm
        self.ee = EE_NAME

        # Joint
        self.joint_max = joint_max
        self.joint_min = joint_min

        # Control mode
        self.mode = mode

        # Load robot
        # urdf_dir = f'simulation/assets/urdf/{urdf_name}/urdf/{urdf_name}_LEFT_V2gripper_separated.urdf'
        if urdf_name is None:
            urdf_dir = f'simulation/assets/urdf/RobotBimanualV3/urdf/RobotBimanualV3_coacd.urdf'
        else:
            urdf_dir = urdf_name
        start_orn = self.bc.getQuaternionFromEuler(start_orn)
        self.uid = self.bc.loadURDF(urdf_dir, start_pos, start_orn, useFixedBase=1, flags=self.bc.URDF_USE_SELF_COLLISION)
        self.single_dof = RobotParams.DOF
        self.dof = 2*RobotParams.DOF
        self.joint_id_offset = joint_id_offset

        self._pose = (start_pos, start_orn)

        # Load Pinocchio interface
        self.pino: Dict[str, PinocchioInterface] = dict()
        for arm in self.ee.keys():
            pino_urdf = f'simulation/assets/urdf/RobotBimanualV3/urdf/RobotBimanualV3_coacd_{arm.upper()}.urdf'
            self.pino[arm] = PinocchioInterface(urdf_dir, arm, self.dof-1)

        self.link_id_dict = dict()
        self.joint_id_dict = dict()
        for joint_id in range(self.bc.getNumJoints(self.uid)):
            joint_info =self.bc.getJointInfo(self.uid, joint_id)
            link_name = joint_info[12].decode('UTF-8')
            self.link_id_dict[link_name] = joint_id
            joint_name = joint_info[1].decode('UTF-8')
            self.joint_id_dict[joint_name] = joint_id
            if log:
                print(link_name, joint_name, joint_id)
        
        for i, link in enumerate(self.link_id_dict.values()):
            self.bc.changeVisualShape(self.uid, link, rgbaColor=pallete[i%len(pallete)])

        

        # Debug slider
        if debug:
            self.debugparams = []
            self.set_debug_slider()

    @property
    def pose(self):
        return self.bc.getBasePositionAndOrientation(self.uid)
    
    @pose.setter
    def pose(self, pose: Tuple[Tuple[float], Tuple[float]]):
        self.bc.resetBasePositionAndOrientation(self.uid, *pose)


    def set_debug_slider(self):
        # ee JP control
        if self.mode == 'jp' or self.mode == 'inv_dyn':
            self.debugparams.append(self.bc.addUserDebugParameter("end-effector X",-0.3,0.3))
            self.debugparams.append(self.bc.addUserDebugParameter("end-effector Y",-0.3,0.3))
            self.debugparams.append(self.bc.addUserDebugParameter("end-effector Z",-0.3,0.3))

            for i in range(self.dof-1):
                self.bc.setJointMotorControl2(self.uid, i, self.bc.VELOCITY_CONTROL, force=0.01)

        elif self.mode == 'pos':
            for i in range(6):
                self.debugparams.append(self.bc.addUserDebugParameter(f"theta_{i+1}", self.joint_min[i], self.joint_max[i], 0))

        elif self.mode == 'sinusoidal':
            self.sin_t = 0

            # Define the range for each axis
            self.x_range = (-0.05, 0.05)
            self.y_range = (-0.05, 0.05)
            self.z_range = (-0.10, 0.0)

            # Define frequencies for each axis
            self.freq_x = 0.05  # Frequency for x-axis
            self.freq_y = 0.055  # Frequency for y-axis
            self.freq_z = 0.06  # Frequency for z-axis

    def update_debug_slider(self, mode: str, value: npt.ArrayLike):
        self.bc.removeAllUserParameters()
        self.debugparams = []
        if self.mode == 'pos':
            assert RobotParams.DOF == len(value)
            for i, theta in enumerate(value):
                self.debugparams.append(self.bc.addUserDebugParameter(f"theta_{i+1}", self.joint_min[i], self.joint_max[i], theta))

        else:
            raise NotImplementedError


    def getJointStates(self, arm:str='all'):
        if arm == 'all':
            joint_ids = list(range(self.single_dof)) + list(range(self.joint_id_offset, self.dof+1))
        else:
            joint_ids = list(range(self.single_dof))

        joint_states = self.bc.getJointStates(self.uid, joint_ids)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]

        if arm == LEFT:
            return joint_positions[:self.single_dof], joint_velocities[:self.single_dof], joint_torques[:self.single_dof]
        elif arm==RIGHT:
            return joint_positions[-self.single_dof:], joint_velocities[-self.single_dof:], joint_torques[-self.single_dof:]
        else:

            return joint_positions, joint_velocities, joint_torques

    def getMotorJointStates(self, arm:str):
        joint_ids = list(range(self.single_dof)) + list(range(self.joint_id_offset, self.dof+1))
        joint_states = self.bc.getJointStates(self.uid, joint_ids)
        joint_infos = [self.bc.getJointInfo(self.uid, i) for i in range(self.bc.getNumJoints(self.uid))]
        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]

        if arm == LEFT:
            return joint_positions[:self.single_dof], joint_velocities[:self.single_dof], joint_torques[:self.single_dof]
        else:
            return joint_positions[-self.single_dof:], joint_velocities[-self.single_dof:], joint_torques[-self.single_dof:]


    def setJointStates(self, arm: str, joint_pos: npt.ArrayLike):
        if arm == LEFT:
            joint_ids = list(range(self.single_dof))
        elif arm == RIGHT:
            joint_ids = list(range(self.joint_id_offset, self.dof+1))
        else:
            joint_ids = list(range(self.single_dof)) + list(range(self.joint_id_offset, self.dof+1))

        for id, value in zip(joint_ids, joint_pos):
            self.bc.resetJointState(self.uid, id, value) 


    def setArmJointStates(self, arm: str, joint_pos: npt.ArrayLike):
        id_offset = RobotParams.DOF_ID_OFFSET if arm == 'right' else 1
        for id, value in enumerate(joint_pos):
            joint_id = self.joint_id_dict[f'joint{id_offset+id}']
            self.bc.resetJointState(self.uid, joint_id, value) 
    

    def get_ee_pose(self, arm: str) -> Tuple[Tuple[float], Tuple[float]]:
        if arm != 'all':
            link_state = self.bc.getLinkState(self.uid, self.link_id_dict[self.ee[arm]])

            return link_state[0], link_state[1]
        
        else:
            left_state = self.bc.getLinkState(self.uid, self.link_id_dict[self.ee[LEFT]])
            right_state = self.bc.getLinkState(self.uid, self.link_id_dict[self.ee[RIGHT]])

            return (left_state[0], left_state[1]), (right_state[0], right_state[1])
    
    def get_link_state(self, link_name: str):
        link_state = self.bc.getLinkState(self.uid, self.link_id_dict[link_name])
        xyz = link_state[4]
        rpy = self.bc.getEulerFromQuaternion(link_state[5])

        return xyz, rpy

    def getGravityCompensation(self, arm: str):
        act_joint_angle, act_joint_angular_velocity, _ = self.getMotorJointStates()
        self.pino[arm].SetRobotParameter(act_joint_angle, act_joint_angular_velocity)
        self.pino[arm].SetGravity()
        return self.pino[arm].GetGravity().squeeze()

    def SaveGravityCompensation(self, arm: str, act_joint_angle: npt.ArrayLike):
        act_joint_angular_velocity = [0, 0, 0, 0, 0, 0]
        self.pino[arm].SetRobotParameter(act_joint_angle, act_joint_angular_velocity)
        self.pino[arm].SetGravity()
        return self.pino[arm].GetGravity().squeeze()