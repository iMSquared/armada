import time
import numpy as np
from pathlib import Path
import sys
from typing import List, Tuple
import numpy.typing as npt

from communication.motorC2T.RealMotorT2C import RealMotorT2C
from communication.Tmotor.src.motor_driver.canmotorlib import CanMotorController


BASEDIR = Path(__file__).parent
if BASEDIR not in sys.path:
    sys.path.insert(0, str(BASEDIR))

# from src.motor_driver.canmotorlib import CanMotorController


def enableMotors(motor_controller_list: List[CanMotorController]):
    joint_status = []

    for motor_controller in motor_controller_list:
        pos, vel, curr = motor_controller.enable_motor()
        joint_status.append([pos, vel, curr])
    
    return np.array(joint_status)

def enableMotorsWithoutSetZero(motor_controller_list):
    joint_status = []

    for motor_controller in motor_controller_list:
        pos, vel, curr = motor_controller.enable_motor_without_set_zero()
        joint_status.append([pos, vel, curr])
    
    return np.array(joint_status)

def disableMotors(motor_controller_list):
    joint_status = []

    for motor_controller in motor_controller_list:
        pos, vel, curr = motor_controller.disable_motor()
        joint_status.append([pos, vel, curr])
    
    return np.array(joint_status)

# def setZeroPosition(motor):
#     pos, _, _ = motor.set_zero_position()
#     while abs(np.rad2deg(pos)) > 0.5:
#         pos, vel, curr = motor.set_zero_position()
#         print("Position: {}, Velocity: {}, Torque: {}".format(np.rad2deg(pos), np.rad2deg(vel), curr))

# def setZeroPositions(motor_controller_list):

#     for motor_controller in motor_controller_list:
#         setZeroPosition(motor_controller)

def setRawJointPositions(motor_controller_list, joint_positions, kp, kd):
    # send_rad_command(p_des_rad, v_des_rad, kp, kd, tau_ff):
    joint_status = []

    # WARNING: idx is not MOTOR_IDS. it is range(0, n_motors)
    for idx, motor_controller in enumerate(motor_controller_list):
        pos, vel, curr = motor_controller.send_rad_command(joint_positions[idx], 0, kp, kd, 0)
        joint_status.append([pos, vel, curr])
    
    return np.array(joint_status)

def convertTorque2Current(t2c: RealMotorT2C, torques):
    return t2c.AK70_Torque2Current(torques)

def convertCurrent2Torque(t2c: RealMotorT2C, currents):
    return t2c.AK70_Current2Torque(currents)
    

def convertJointFromRealtoURDF(real_joint_positions, offset=None) -> npt.NDArray:
    # Caution: unit is radian
    urdf_joint_positions = -real_joint_positions
    if offset is not None:
        urdf_joint_positions = urdf_joint_positions + offset
    # urdf_joint_positions[1] -= np.pi*90/180
    # urdf_joint_positions[3] -= np.pi*50/180
    return urdf_joint_positions

def convertJointFromURDFtoReal(urdf_joint_positions, offset=None):
    # Caution: unit is radian
    real_joint_positions = -urdf_joint_positions
    if offset is not None:
        urdf_joint_positions = urdf_joint_positions + offset
    # real_joint_positions[1] -= np.pi*90./180.
    # real_joint_positions[3] -= np.pi*50./180.
    return real_joint_positions

def urdfJointPDControl(t2c:RealMotorT2C, 
                       motor_controller_list: List[CanMotorController], 
                       joint_positions: npt.NDArray,
                       joint_velocities: npt.NDArray,
                       kps: npt.NDArray,
                       kds: npt.NDArray,
                       ff_torque: npt.NDArray=None,
                       offset: npt.NDArray=None):
    # Joint position PD control given URDF-based joint positions
    # joint_positions ==> URDF position
    joint_status = []
    real_joint_positions = convertJointFromURDFtoReal(joint_positions, offset)
    real_joint_velocities = convertJointFromURDFtoReal(joint_velocities, None)
    

    if ff_torque is not None: 
        # print("input torque: ", ff_torque)
        gravity_current = convertTorque2Current(t2c, ff_torque)
        # print("input current: ", gravity_current)

    # print("input current: ", gravity_current)

    # WARNING: idx is not MOTOR_IDS. it is range(0, n_motors)
    for idx, motor_controller in enumerate(motor_controller_list):
        if ff_torque is None:
            # print(f'idx = {idx}')
            pos, vel, curr = motor_controller.send_rad_command(real_joint_positions[idx], real_joint_velocities[idx], kps[idx], kds[idx], 0)
        else:
            pos, vel, curr = motor_controller.send_rad_command(real_joint_positions[idx], real_joint_velocities[idx], kps[idx], kds[idx], -1*gravity_current[idx]) # direction convention difference

        joint_status.append([pos, vel, curr])
    
    joint_status = np.array(joint_status)
    real_pos = joint_status[:, 0]
    real_vel = joint_status[:, 1]
    real_curr = joint_status[:, 2]

    # print("measured motor driver values: ", real_pos, real_vel, real_curr)

    # Caution: init position
    return convertJointFromRealtoURDF(real_pos, offset=offset), -1.*real_vel, -1.*convertCurrent2Torque(t2c, real_curr) # direction convention difference

def torque_control_Tmotor(t2c:RealMotorT2C, 
                          motor_controller_list: List[CanMotorController],
                          ff_torque: npt.NDArray,
                          offset: npt.NDArray=None) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    # Joint position PD control given URDF-based joint positions
    # joint_positions ==> URDF position
    joint_status = []
    single = False
    if len(motor_controller_list) == 1: single=True
    ff_current = convertTorque2Current(t2c, ff_torque)


    for idx, motor_controller in enumerate(motor_controller_list):
        pos, vel, curr = motor_controller.send_rad_command(0, 0, 0, 0, -1*ff_current[idx]) # direction convention difference
        joint_status.append([pos, vel, curr])
    
    joint_status = np.array(joint_status)
    real_pos = joint_status[:, 0]
    real_vel = joint_status[:, 1]
    real_curr = joint_status[:, 2]

    # print("measured motor driver values: ", real_pos, real_vel, real_curr)

    # Caution: init position
    return convertJointFromRealtoURDF(real_pos, offset=offset), -1.*real_vel, -1.*convertCurrent2Torque(t2c, real_curr) # direction convention difference

def current_control_Tmotor(motor_controller_list: List[CanMotorController],
                          ff_current: npt.NDArray,
                          offset: npt.NDArray=None) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    # Joint position PD control given URDF-based joint positions
    # joint_positions ==> URDF position
    joint_status = []
    # start_time = time.time()

    for idx, motor_controller in enumerate(motor_controller_list):
        
        pos, vel, curr = motor_controller.send_rad_command(0, 0, 0, 0, -1*ff_current[idx]) # direction convention difference
        joint_status.append([pos, vel, curr])
        
    
    joint_status = np.array(joint_status)
    real_pos = joint_status[:, 0]
    real_vel = joint_status[:, 1]
    real_curr = joint_status[:, 2]

    # print("measured motor driver values: ", real_pos, real_vel, real_curr)
    # dur = time.time()-start_time
    # print(f'tmotor duration: {dur*1000} ms')

    # Caution: init position
    return convertJointFromRealtoURDF(real_pos, offset=offset), -1.*real_vel, -1.*real_curr # direction convention difference
