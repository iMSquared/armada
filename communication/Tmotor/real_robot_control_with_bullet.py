from pybullet_utils import bullet_client
from abc import ABC, abstractmethod
from pathlib import Path
from enum import IntEnum
import pybullet_data
import numpy as np
import pybullet
import time

from src.motor_driver.canmotorlib import CanMotorController
from can_util import *

def getJointStates(robot):
  joint_states = sim.getJointStates(robot, range(sim.getNumJoints(robot)))
  joint_positions = [state[0] for state in joint_states]
  joint_velocities = [state[1] for state in joint_states]
  joint_torques = [state[3] for state in joint_states]
  return joint_positions, joint_velocities, joint_torques


def getMotorJointStates(robot):
  joint_states = sim.getJointStates(robot, range(sim.getNumJoints(robot)))
  joint_infos = [sim.getJointInfo(robot, i) for i in range(sim.getNumJoints(robot))]
  joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
  joint_positions = [state[0] for state in joint_states]
  joint_velocities = [state[1] for state in joint_states]
  joint_torques = [state[3] for state in joint_states]
  return joint_positions, joint_velocities, joint_torques

if __name__ == "__main__":
    sim = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    sim.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Plane
    sim.loadURDF("plane.urdf")

    # Panda
    startPos = [0, 0, 0.5]
    startOrientation = sim.getQuaternionFromEuler([0, 0, 0])
    robotId = sim.loadURDF('asset/new_4dof/new_4dof.urdf', startPos, startOrientation, useFixedBase=1)

    numJoints = sim.getNumJoints(robotId)
    link_id_dict = dict()
    joint_id_dict = dict()
    for _id in range(numJoints):
        joint_info = sim.getJointInfo(robotId, _id)
        link_name = joint_info[12].decode('UTF-8')
        link_id_dict[link_name] = _id
        joint_name = joint_info[1].decode('UTF-8')
        joint_id_dict[joint_name] = _id
        print(link_name, joint_name, _id)
    
    sim.changeVisualShape(robotId, link_id_dict['link1'], rgbaColor=[1., 0., 0., 1.])
    sim.changeVisualShape(robotId, link_id_dict['link2'], rgbaColor=[0., 1., 0., 1.])
    sim.changeVisualShape(robotId, link_id_dict['link3'], rgbaColor=[0., 0., 1., 1.])
    sim.changeVisualShape(robotId, link_id_dict['link4'], rgbaColor=[0., 1., 1., 1.])
    
    # link1 joint_01 0
    # link2 joint_12 1
    # link3 joint_23 2
    # link4 joint_34 3
    # link5 joint_45 4

    end_effector_name = 'link5'
    # sim.getLinkState(robotId, link_id_dict[end_effector])[0]
        
    REAL_ROBOT = True
    MODE = 'sinusoidal' # 'jp' or 'pos' or 'sinusoidal' or 'inv_dyn'
    
    
    debugparams = []
    # ee JP control
    if MODE == 'jp' or MODE == 'inv_dyn':
        debugparams.append(sim.addUserDebugParameter("end-effector X",-0.3,0.3))
        debugparams.append(sim.addUserDebugParameter("end-effector Y",-0.3,0.3))
        debugparams.append(sim.addUserDebugParameter("end-effector Z",-0.3,0.3))

        for i in range(numJoints-1):
            sim.setJointMotorControl2(robotId, i, sim.VELOCITY_CONTROL, force=0.01)

    elif MODE == 'pos':
        for i in range(4):
            debugparams.append(sim.addUserDebugParameter(f"theta_{i+1}",-np.pi/2,np.pi/2,0))

    elif MODE == 'sinusoidal':
        sin_t = 0

        # Define the range for each axis
        x_range = (-0.05, 0.05)
        y_range = (-0.05, 0.05)
        z_range = (-0.10, 0.0)

        # Define frequencies for each axis
        freq_x = 0.05  # Frequency for x-axis
        freq_y = 0.055  # Frequency for y-axis
        freq_z = 0.06  # Frequency for z-axis

    sim.setRealTimeSimulation(False)
    sim.setGravity(0, 0, 0)

    init_pos, _, _ = getJointStates(robotId)
    init_EE_pos = sim.getLinkState(robotId, link_id_dict[end_effector_name])[0]

    # timeStepId = sim.addUserDebugParameter("timeStep", 0.001, 0.1, 0.01)

    

    ################## real robot communication ##################

    if REAL_ROBOT:
        motor_controller_list = []
        for motor_id in MOTOR_IDS:
            motor_controller_list.append(CanMotorController(
                CAN_DEVICE_NAME, motor_id, motor_type="AK70_10_V1p1"
            ))
        time.sleep(1)

        enableMotors(motor_controller_list)
        time.sleep(1)

        setZeroPositions(motor_controller_list)
        time.sleep(1)

    ######################################################


    for _ in range(3000):
        sim.stepSimulation()
        # timeStep = sim.readUserDebugParameter(timeStepId)
        # sim.setTimeStep(timeStep)
        # time.sleep(timeStep)
        thetas = []
        for param in debugparams:
           thetas.append(sim.readUserDebugParameter(param))

        if MODE == 'jp':
            targetEEPosition = init_EE_pos + np.array(thetas)
            jointPoses = sim.calculateInverseKinematics(robotId, 4, targetEEPosition)
            for i in range(numJoints-1):
                # sim.resetJointState(robotId, i, jointPoses[i])
                sim.setJointMotorControl2(robotId,i,sim.POSITION_CONTROL,jointPoses[i])
            
            ################## real robot communication ##################
            if REAL_ROBOT:
                print(np.array(jointPoses) - np.array(init_pos[:4]))

                real_joint_target = - (np.array(jointPoses) - np.array(init_pos[:4]))
                real_joint_target[3] -= real_joint_target[2] # considering linkage

                setJointPositions(motor_controller_list, real_joint_target, 4, 2)

            ######################################################

        elif MODE == 'inv_dyn':
            targetEEPosition = init_EE_pos + np.array(thetas)
            cur_EE_state = sim.getLinkState(robotId, link_id_dict[end_effector_name], computeLinkVelocity=1)
            cur_EE_pos = cur_EE_state[0]
            cur_EE_vel = cur_EE_state[6]
            dpose = targetEEPosition - cur_EE_pos

            mpos, mvel, mtorq = getMotorJointStates(robotId)
            pos, vel, torq = getJointStates(robotId)

            result = sim.getLinkState(robotId,
                        link_id_dict[end_effector_name],
                        computeLinkVelocity=1,
                        computeForwardKinematics=1)
            link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result

            zero_vec = [0.0] * len(mpos)
            jac_t, jac_r = sim.calculateJacobian(robotId, link_id_dict[end_effector_name], com_trn, mpos, zero_vec, zero_vec)
            jac_t = np.array(jac_t)
            mm = sim.calculateMassMatrix(robotId, pos)
            
            def control_osc(j_eef, hand_vel, mm, dpose, kp):
                kd = 2.0 * np.sqrt(kp)
                mm_inv = np.linalg.inv(mm)
                m_eef_inv = np.matmul(np.matmul(j_eef, mm_inv), np.transpose(j_eef))
                m_eef = np.linalg.inv(m_eef_inv)
                u = np.matmul(np.matmul(np.transpose(j_eef), m_eef), (
                    kp * dpose - kd * hand_vel)[..., None])
                return u.reshape(-1)
            
            torques = control_osc(jac_t, cur_EE_vel, mm, dpose, kp=np.array([10, 10, 10]))

            print(torques)
                    
            for i in range(numJoints-1):
                # sim.resetJointState(robotId, i, jointPoses[i])
                sim.setJointMotorControl2(robotId, i, sim.TORQUE_CONTROL, force=torques[i])
        
        elif MODE == 'pos':
            sim.resetJointState(robotId, joint_id_dict['joint_01'], targetValue=thetas[0])
            sim.resetJointState(robotId, joint_id_dict['joint_12'], targetValue=thetas[1])
            sim.resetJointState(robotId, joint_id_dict['joint_23'], targetValue=thetas[2])
            sim.resetJointState(robotId, joint_id_dict['joint_34'], targetValue=thetas[3])

            jointPoses = np.array(thetas)

            cur_joint_pos = []
            for i in range(4):
                link_name = f'link{i+1}'
                joint_pos = sim.getJointState(robotId, link_id_dict[link_name])[0]
                cur_joint_pos.append(joint_pos)
            cur_joint_pos = np.array(cur_joint_pos)
            print(cur_joint_pos)

            ################## real robot communication ##################
            if REAL_ROBOT:    
                print(np.array(jointPoses) - np.array(init_pos[:4]))

                real_joint_target = - (np.array(jointPoses) - np.array(init_pos[:4]))
                real_joint_target[3] -= real_joint_target[2] # considering linkage

                setJointPositions(motor_controller_list, real_joint_target, 4, 2)

            ######################################################
        
        elif MODE == 'sinusoidal':

            x_traj = (x_range[1] - x_range[0]) * np.sin(freq_x * sin_t) / 2
            y_traj = (y_range[1] - y_range[0]) * np.sin(freq_y * sin_t) / 2
            z_traj = (z_range[1] - z_range[0]) * np.sin(freq_z * sin_t) / 2

            targetPosition = init_EE_pos + np.array([x_traj, y_traj, z_traj])
            jointPoses = sim.calculateInverseKinematics(robotId, 4, targetPosition)
            for i in range(numJoints-1):
                # sim.resetJointState(robotId, i, jointPoses[i])
                sim.setJointMotorControl2(robotId,i,sim.POSITION_CONTROL,jointPoses[i])

            sin_t += 1

            ################## real robot communication ##################
            if REAL_ROBOT:
                print(np.array(jointPoses) - np.array(init_pos[:4]))

                real_joint_target = - (np.array(jointPoses) - np.array(init_pos[:4]))
                real_joint_target[3] -= real_joint_target[2] # considering linkage

                setJointPositions(motor_controller_list, real_joint_target, 6, 2.5)

            ######################################################
        time.sleep(0.01)
    
    if REAL_ROBOT: 
        disableMotors(motor_controller_list)

    sim.disconnect()