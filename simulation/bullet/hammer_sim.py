#%% Import libraries and initialize global variable
import os, sys
from pathlib import Path
BASEDIR = Path(__file__).parent.parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))

from numpy import typing as npt
from typing import List
import time
import numpy as np
import pybullet_data
from pybullet_utils.bullet_client import BulletClient
import pybullet as p

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

# Experiment
from experiment.utils.exp_utils import *

# Robot parameters
from control import DefaultControllerValues as RobotParams

# Manipulation
from simulation.bullet.manipulation import *

DOF = RobotParams.DOF
FPS = 200

VIZ = True

def set_joint_state(sim, robotId, joint_id_dict, thetas):
    for i in range(6):
        sim.resetJointState(robotId, joint_id_dict[f'joint{7+i}'], targetValue=thetas[i])

def get_joint_state(sim, robotId, joint_id_dict):
    thetas = np.zeros(6)
    for i in range(6):
        thetas[i] = sim.getJointState(robotId, joint_id_dict[f'joint{7+i}'])[0]

    return thetas


def set_position_control(sim, robotId, joint_id_dict, thetas, kp=None, kd=None):
    if kp is None:
        for i in range(6):
            sim.setJointMotorControl2(robotId, joint_id_dict[f'joint{7+i}'], sim.POSITION_CONTROL, thetas[i])
    else:
        for i in range(6):
            sim.setJointMotorControl2(robotId, joint_id_dict[f'joint{7+i}'], sim.POSITION_CONTROL, thetas[i], \
                                      positionGain=kp[i], velocityGain=kd[i])


def interpolate_vectors(init: npt.NDArray, term: npt.NDArray, steps: int):
    delta = term - init
    trajectory = [init + delta*(i/steps) for i in range(1, steps+1)]

    return trajectory


def interpolate_trajectory(cur: npt.NDArray, goal: npt.NDArray, 
                           action_duration: float, control_dt: float) -> List[npt.NDArray]:
    '''
    This function returns linear-interpolated (dividing straight line)
    trajectory between current and goal pose.
    Acc, Jerk is not considered.
    '''
    # Interpolation steps
    steps = math.ceil(action_duration/control_dt)

    return interpolate_vectors(cur, goal, steps)


#%% Driver function
def main():    

    sim = BulletClient(connection_mode=p.GUI)
    sim.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Plane
    sim.loadURDF("plane.urdf", [0,0,0])

    TABLE = True
    table_height = 0.5065

    if TABLE:
        tableShape = (0.2, 0.4, table_height/2)
        tablePosition = (0.3, 0.0, table_height/2)
        boxColor = (np.array([170, 170, 170, 255]) / 255.0).tolist()
        tableVisualShapeId = sim.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=tableShape,
            rgbaColor=boxColor
        )
        tableCollisionShapeId = sim.createCollisionShape(
            shapeType=p.GEOM_BOX, 
            halfExtents=tableShape
        )
        tableId = sim.createMultiBody(
            baseMass=1000,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )


    urdf_path = 'simulation/assets/urdf/RobotBimanualV6/urdf/Simplify_Robot_plate_gripper_RIGHT.urdf'
    end_effector_name = 'tool2'
    startPos = [0., 0., 0.]
    startOrientation = sim.getQuaternionFromEuler([0, 0, 0.])
    robotId = sim.loadURDF(urdf_path, startPos, startOrientation, useFixedBase=1, flags=p.URDF_USE_SELF_COLLISION)

    numJoints = sim.getNumJoints(robotId)
    link_id_dict = dict()
    joint_id_dict = dict()
    link_names = []

    for _id in range(numJoints):
        joint_info = sim.getJointInfo(robotId, _id)
        link_name = joint_info[12].decode('UTF-8')
        link_id_dict[link_name] = _id
        joint_name = joint_info[1].decode('UTF-8')
        joint_id_dict[joint_name] = _id
        link_names.append(link_name)
        print(link_name, joint_name, _id)

    sim.changeVisualShape(robotId, link_id_dict['link7'], rgbaColor=[1., 0., 0., 1.])
    sim.changeVisualShape(robotId, link_id_dict['link8'], rgbaColor=[0., 1., 0., 1.])
    sim.changeVisualShape(robotId, link_id_dict['link9'], rgbaColor=[0., 0., 1., 1.])
    sim.changeVisualShape(robotId, link_id_dict['link10'], rgbaColor=[0., 1., 1., 1.])
    sim.changeVisualShape(robotId, link_id_dict['link11'], rgbaColor=[1., 0., 1., 1.])
    sim.changeVisualShape(robotId, link_id_dict['link12'], rgbaColor=[1., 1., 0., 1.])

    init_hammering_pose = np.array([0., 0., 0., 0., -1.582, 0.])
    prepare_pose = np.array([-0.362, 0.294, 0.501, 0.245, -1.582, -1.102])
    hitting_pose = np.array([0.088, 0.156, -0.344, -0.118, -1.505, -0.390])

    ee_state = sim.getLinkState(robotId, link_id_dict[end_effector_name])
    print(f"init ee pose: {ee_state[0]}")

    duration = 2
    control_dt = 1./200.
    traj1 = interpolate_trajectory(init_hammering_pose, prepare_pose, 2, control_dt)
    traj2 = interpolate_trajectory(prepare_pose, hitting_pose, 1, control_dt)
    traj = np.concatenate([traj1, traj2], axis=0)

    # torque control

    sim.setRealTimeSimulation(False)
    sim.setGravity(0, 0, 0)
    sim.setTimeStep(control_dt)
    sim.stepSimulation()

    for id in range(6):
        sim.setJointMotorControl2(robotId, id, sim.VELOCITY_CONTROL, targetVelocity=0, force=0.02)

    POS_P_GAIN: npt.NDArray = np.array([1.8, 2.2, 1.6, 1.6, 0.7, 0.7])
    POS_D_GAIN: npt.NDArray = np.array([0.02, 0.02, 0.02, 0.02, 0.01, 0.01])

    for _ in range(3):
        
        cur_joint_state = get_joint_state(sim, robotId, joint_id_dict)
        traj1 = interpolate_trajectory(cur_joint_state, prepare_pose, 2, control_dt)
        traj2 = interpolate_trajectory(prepare_pose, hitting_pose, 1, control_dt)
        traj = np.concatenate([traj1, traj2], axis=0)

        for q in traj:
            sim.stepSimulation()
            set_position_control(sim, robotId, joint_id_dict, q, kp=POS_P_GAIN, kd=POS_D_GAIN)
            time.sleep(control_dt)

        for _ in range(200):
            sim.stepSimulation()
            set_position_control(sim, robotId, joint_id_dict, np.zeros(6), kp=np.zeros(6), kd=POS_D_GAIN)
            time.sleep(control_dt)

        
    # # position control
    # for q in traj1:
    #     set_joint_state(sim, robotId, joint_id_dict, q)
    #     time.sleep(1/200)

    # for q in traj2:
    #     set_joint_state(sim, robotId, joint_id_dict, q)
    #     time.sleep(1/200)


def simulate():
    urdf_path = 'simulation/assets/urdf/RobotBimanualV6/urdf/Simplify_Robot_hammer_LEFT.urdf'

    start_pos = [0, 0, 0]
    start_orn = [0, 0, 0]

    manip = Manipulation(start_pos, 
                         start_orn, 
                         'rrt', 
                         arm=LEFT,
                         robot_name=urdf_path, 
                         control_freq=FPS,
                         resolution=0.01, 
                         joint_min=JOINT_POS_LOWER_LIMIT, 
                         joint_max=JOINT_POS_UPPER_LIMIT, 
                         joint_vel_upper_limit=JOINT_VEL_UPPER_LIMIT,
                         jonit_id_offset=JOINT_ID_OFFSET,
                         debug=True)
        
 
    ## 3: Bimanual snatch
    init = np.array([0.362, -0.294, -0.501, -0.245, 1.582, 1.102])
    term = np.array([-0.088, -0.156, 0.344, 0.118, 1.505, 0.390])

    # Move to grasping mode
    l_0, _, _ = manip.robot.getJointStates(LEFT)

    left_traj = manip.motion_plan(LEFT, np.array(l_0), init)

    manip.simulate_traj(LEFT, left_traj, draw=True)
    
    # p.removeAllUserDebugItems()

    hammering_traj = manip.motion_plan(LEFT, init, term, allow_uid_list=[manip.scene['table']])
    manip.simulate_traj(LEFT, hammering_traj, draw=True)

    manip.close()


if __name__ == '__main__':
    simulate()

