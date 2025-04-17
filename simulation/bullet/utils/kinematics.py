import sys
from pathlib import Path
BASEDIR = Path(__file__).parent.parent.parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.append(str(BASEDIR))
from typing import List, Tuple, Union
import numpy.typing as npt
import numpy as np

import pybullet as p
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

from simulation.bullet.utils.capture_env import *
from simulation.bullet.robot import Robot, LEFT, RIGHT
from simulation.utils.misc import time_it

from scipy.spatial.transform import Rotation as R
# import pyikfast_im2r

DOF = 6

class Kinematics:
    def __init__(self, analytical: bool, 
                 bc: Union[BulletClient, None]=None,
                 robot: Robot=None,
                 robot_pose: Union[Tuple[npt.ArrayLike, npt.ArrayLike], None]=None):
        
        assert analytical or (not analytical and bc is not None), 'Numerical IK requires bullet client'

        self.analytical = analytical
        self.bc = bc
        self.robot = robot

        if self.analytical:
            self.ikfast = pyikfast_im2r
        
        if robot is not None:
            self.robot_pose = self.robot.pose
        elif robot_pose is not None:
            self.robot_pose = robot_pose
        else:
            raise ValueError('Provide either robot_pose or bullet client')

        self.dof = self.robot.single_dof if self.robot is not None else DOF


    def forward(self, arm:str, joint_positions: npt.ArrayLike):
        if len(joint_positions) != self.dof:
            raise ValueError(f'The joint position must have length of {self.dof}')
        
        if self.analytical:
            pos, orn = self.ikfast.forward(joint_positions)
            orn = np.reshape(np.array(orn), (3,3))
            orn = R.from_matrix(orn).as_quat().tolist()
            ee_pose = (tuple(pos), tuple(orn))

        else:
            with dream(self.bc, self.robot):
                self.robot.setArmJointStates(arm, joint_positions)
                ee_pose = self.robot.get_ee_pose(arm)

        return ee_pose
    

    @time_it
    def inverse(self, 
                arm: str, 
                pos: npt.ArrayLike, 
                orn: npt.ArrayLike,
                max_iter: int=20) -> List[npt.NDArray]:

        if self.analytical:
            pos = list(pos)
            orn = R.from_quat(orn).as_matrix().flatten().tolist()
            joint_positions = self.ikfast.inverse(pos, orn)

        joint_positions = self.bc.calculateInverseKinematics(self.robot.uid, 
                                                             self.robot.link_id_dict[self.robot.ee[arm]], 
                                                             targetPosition=pos,
                                                             targetOrientation=orn,
                                                             maxNumIterations=max_iter)
            
        return joint_positions
    
    
def compareFK(arm: str, akin: Kinematics, nkin: Kinematics, joint_pos: npt.ArrayLike, profile:bool=True):

    apose = akin.forward(arm, joint_pos)
    npose = nkin.forward(arm, joint_pos)

    print(f'Analytical FK Pose {apose}')
    print(f'Numerical FK Pose {npose}')
    a_orn_euler = R.from_quat(apose[1]).as_euler('XYZ')
    n_orn_euler = R.from_quat(npose[1]).as_euler('XYZ')

    print(f'Analytical FK Orn: {a_orn_euler}')
    print(f'Numerical FK Orn: {n_orn_euler}')


def compareIK(akin: Kinematics, nkin: Kinematics, ee_pose: npt.ArrayLike, profile:bool=True):

    print('Doing Analytical IK...')
    a_jp = akin.inverse(arm, *ee_pose)
    print('Doing Numerical IK...')
    n_jp = nkin.inverse(arm, *ee_pose)

    # print(f'Numerical IK: {np.array(n_jp)}')
    # print(f'Analytical IK: {np.array(a_jp)}')
    

    

if __name__ == '__main__':
    float_formatter = '{:.5f}'.format
    np.set_printoptions(formatter={'float_kind':float_formatter})

    bc = BulletClient(connection_mode=p.GUI)
    bc.setAdditionalSearchPath(pybullet_data.getDataPath())
    bc.loadURDF("plane.urdf", [0,0,0])

    sim_dir = Path(__file__).parent.parent
    urdf_name = 'RobotBimanualV5'

    start_pos = [0., 0., 0.]
    start_orn = [0., 0., 0.]

    robot = Robot(bc, urdf_name, start_pos, start_orn)
    arm = 'right'
   
    # akin = Kinematics(analytical=True, robot=robot)
    nkin = Kinematics(analytical=False, bc=bc, robot=robot)

    joint_pos = [-1.2, 0.4, 0.12, 0, np.pi/6, -np.pi/3]
    # a_jp = akin.forward(joint_pos)
    n_jp = nkin.forward(arm, joint_pos)
    # compareFK(akin, nkin, joint_pos)
    # compareIK(akin, nkin, a_jp)
    nkin.robot.setArmJointStates(arm, joint_pos)

    print('Sim ended')