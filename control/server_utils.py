import sys
from pathlib import Path
BASEDIR = Path(__file__).parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.append(str(BASEDIR))

from control import DefaultControllerValues as RobotParams
from simulation.bullet.manipulation import Manipulation
import numpy as np
import numpy.typing as npt
from collections import deque

URDF_DIR = 'simulation/assets/urdf/RobotBimanualV5/urdf/Simplify_Robot_plate_gripper.urdf'

class MovingAverageFilter():

    def __init__(self, window_size: int):

        assert window_size > 1, 'window size must be at least 1'

        self.window_size = window_size
        self.kernel = np.ones(window_size)/window_size
        self.history = deque(maxlen=window_size)


    def filter(self, data: npt.NDArray):
        self.history.append(data)

        # Populate history with data if it is first time
        if len(self.history) == 1:
            for _ in range(self.window_size - 1):
                self.history.append(data)

        filtered_data = np.mean(np.array(self.history), axis=0)

        return filtered_data
    


def build_manip(urdf_dir=URDF_DIR, **kwargs) -> Manipulation:
        joint_pos_lower_limit = np.concatenate((RobotParams.L_JOINT_LIMIT_MIN, RobotParams.GRIPPER_SIM_MIN, RobotParams.R_JOINT_LIMIT_MIN, RobotParams.GRIPPER_SIM_MIN))
        joint_pos_upper_limit = np.concatenate((RobotParams.L_JOINT_LIMIT_MAX, RobotParams.GRIPPER_SIM_MAX, RobotParams.R_JOINT_LIMIT_MAX, RobotParams.GRIPPER_SIM_MAX))
        joint_vel_upper_limit = RobotParams.JOINT_VEL_UPPER_LIMIT

        robot_name = urdf_dir
        
        start_pos = [0, 0, 0]
        start_orn = [0, 0, 0]
        
        manip = Manipulation(start_pos, 
                             start_orn,
                             planner='rrt',
                             arm='all',
                             scene_on=False,
                             robot_name=robot_name, 
                             joint_min=joint_pos_lower_limit, 
                             joint_max=joint_pos_upper_limit, 
                             joint_vel_upper_limit=joint_vel_upper_limit,
                             debug=False,
                             **kwargs)

        return manip