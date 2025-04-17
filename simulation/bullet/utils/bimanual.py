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
from simulation.bullet.utils.common import draw_axes
from simulation.bullet.robot import Robot, LEFT, RIGHT

from scipy.spatial.transform import Rotation as R

DOF = 6

class BimanualHandPoseGenerator:
    '''
        Generate antipodal hand pose for bimanual manipulation.
        Args:
            main_hand: Name of control arm (<----> subordinate arm). Default to LEFT.
            bc: Bullet client
            robot: Robot object in bullet sim
            distance: Distance between two hands
            offset: Orientation offset of two hands. In degree and Euler.

    '''
    def __init__(self, 
                 main_arm: str=LEFT, 
                 bc: Union[BulletClient, None]=None,
                 robot: Robot=None,
                 offset_position: Union[float, npt.ArrayLike, None]=None,
                 offset_orientation: npt.ArrayLike=(0,0,0)):
        
        assert main_arm in [LEFT, RIGHT], '[BimanualHandPoseGenerator] main_hand must be either left or right'
        

        self.main_hand = main_arm
        self.sub_hand = LEFT if main_arm == RIGHT else RIGHT
        self.bc = bc
        self.robot = robot

        # Offset
        if isinstance(offset_position, float):
            self.position = (0, offset_position, 0)
        else:
            self.position = offset_position
        self.orientation = offset_orientation


    def __call__(self, 
                 T_mw: Union[Tuple[Tuple[float]],None]=None, 
                 offset_position: Union[float, npt.ArrayLike, None]=None, 
                 offset_orientation: Tuple[float]=(0,0,0), 
                 debug: bool=False):
        
        if T_mw is None:
            T_mw = self.robot.get_ee_pose(self.main_hand)

        if offset_position is not None:
            if isinstance(offset_position, float):
                self.position = (0, offset_position, 0)
            else:
                self.position = offset_position
        
        if offset_orientation is not None:
            self.orientation = offset_orientation

        # Compute flip
        flip_euler = np.array(self.orientation)*np.pi/180 + np.array((0,0,np.pi))
        orn_flip = self.bc.getQuaternionFromEuler(flip_euler)

        # Transform from main to sub
        # m: main, s: subordinate, w: world
        T_sm = (self.position, orn_flip)
        T_sw = self.bc.multiplyTransforms(*T_mw, *T_sm)

        if debug:
            draw_axes(self.bc, *T_sw)

        return T_sw