from control import DefaultControllerValues as RobotProp
import enum
from typing import Deque, Dict, Tuple, List, Union

# ################### #
# Dimensions of robot #
# ################### #

LEFT = RobotProp.LEFT
RIGHT = RobotProp.RIGHT
BIMANUAL = RobotProp.BIMANUAL

DOF = RobotProp.DOF

EE_NAMES = RobotProp.EE_NAMES
EE_LINK = RobotProp.EE_LINK
JOINT_FIRST = {LEFT: 1, RIGHT: DOF+1}     # (1, 7)
JOINT_LAST = {LEFT: DOF, RIGHT: DOF*2}      # (6, 12)

L_JOINT_RES_START = JOINT_FIRST[LEFT] - 1       # 0
L_JOINT_P_START = L_JOINT_RES_START + DOF       # 6
L_JOINT_D_START = L_JOINT_P_START + DOF         # 12
R_JOINT_RES_START = JOINT_FIRST[RIGHT] - 1 + 2*DOF  # 18
R_JOINT_P_START = R_JOINT_RES_START + DOF           # 24
R_JOINT_D_START = R_JOINT_P_START +  DOF            # 30

class Dimensions(enum.Enum):
    """
    Dimensions of the Franka with gripper robot.

    """
    # general state
    # cartesian position + quaternion orientation
    PoseDim = 7
    # linear velocity + angular velcoity
    VelocityDim = 6

    # Width 
    WidthDim = 0.013

    # position of keypoint
    KeypointDim = 3
    TwoDimensionKeypointDim = 2

    # tool frame pose
    StateDim = 7
    # force + torque
    WrenchDim = 6
    # for all joints
    JointPositionDim = DOF # the number of joint
    JointVelocityDim = DOF # the number of joint
    JointTorqueDim = DOF # the number of joint

    # generalized coordinates
    GeneralizedCoordinatesDim = JointPositionDim
    GeneralizedVelocityDim = JointVelocityDim

    # for objects
    ObjectPoseDim = 7
    ObjectVelocityDim = 6


class CuboidalObject:
    def __init__(self, size: Tuple[float, float, float]):
        """Initialize the cuboidal object.

        Args:
            size: The size of the object along x, y, z in meters. 
        """
        self._size = size

    """
    Properties
    """

    @property
    def size(self) -> Tuple[float, float, float]:
        """
        Returns the dimensions of the cuboid object (x, y, z) in meters.
        """
        return self._size

    """
    Configurations
    """

    @size.setter
    def size(self, size: Tuple[float, float, float]):
        """ Set size of the object.

        Args:
            size: The size of the object along x, y, z in meters. 
        """
        self._size = size
