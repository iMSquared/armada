import sys
from pathlib import Path
BASEDIR = Path(__file__).parent
if str(BASEDIR) not in sys.path:
    sys.path.append(str(BASEDIR))

import pybullet as p
import pybullet_data
import numpy as np
import time

# URDF_PATH = str(Path.joinpath(BASEDIR, 'urdf/Simplify_Robot_plate_gripper.urdf'))
URDF_PATH = '/home/jiho/URDFs/RobotBimaualV5/urdf/Simplify_Robot_plate_gripper.urdf'

L_JOINT_LIMIT_MIN = np.pi/180.*np.array([-90.,  -90., -180.,  -45., -210., -125.]) 
L_JOINT_LIMIT_MAX = np.pi/180.*np.array([50.,  80., 80.,  75., 210. , 125.])

R_JOINT_LIMIT_MIN = np.pi/180.*np.array([-50.,  -80., -80.,  -75., -210., -125.]) 
R_JOINT_LIMIT_MAX = np.pi/180.*np.array([90.,  90., 180.,  45., 110. , 125.])

def getJointLimits(robotId, numJoints):
    lower_limits = []
    upper_limits = []
    for i in range(numJoints):
        joint_info = p.getJointInfo(robotId, i)
        lower_limits.append(joint_info[8])  # joint lower limit
        upper_limits.append(joint_info[9])  # joint upper limit
    return np.array(lower_limits), np.array(upper_limits)

if __name__ == "__main__":
    sim = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Load Plane
    p.loadURDF("plane.urdf")

    # Load the robot
    startPos = [0, 0, 0]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    robotId = p.loadURDF(URDF_PATH, startPos, startOrientation, useFixedBase=True)

    numJoints = p.getNumJoints(robotId)

    # Get joint limits for the 12 revolute joints
    joint_pos_lower_limit, joint_pos_upper_limit = getJointLimits(robotId, numJoints)

    # Initialize debug parameters (sliders) for each joint
    debug_params = []
    for i in range(numJoints):
        debug_params.append(
            p.addUserDebugParameter(f"Joint_{i+1}", joint_pos_lower_limit[i], joint_pos_upper_limit[i], 0)
        )

    p.setRealTimeSimulation(1)
    p.setGravity(0, 0, -9.81)

    while True:
        # Read slider values and apply them to the joints
        for i in range(numJoints):
            joint_value = p.readUserDebugParameter(debug_params[i])
            p.resetJointState(robotId, i, targetValue=joint_value)

        time.sleep(0.01)
