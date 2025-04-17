import sys
from pathlib import Path
BASEDIR = Path(__file__).parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.append(str(BASEDIR))

from concurrent import futures
import threading
import logging
import time
import argparse

import grpc
from control import JointNames, CANIDs, CANSockets
from control.im2control_pb2 import *
from control.im2control_pb2_grpc import IM2ControlServicer, add_IM2ControlServicer_to_server
from typing import List, Tuple, Dict
import numpy.typing as npt
from concurrent.futures import ThreadPoolExecutor

# Pinocchio
from simulation.utils.PinocchioInterface import *

# Real robot
import communication.RMD.motor.RMD as RMD
from communication.Tmotor.src.motor_driver.canmotorlib import CanMotorController as Tmotor
from communication.Tmotor.can_util import *
import can
from communication.Dynamixel.Gripper_Control import GripperControl, L_ID, R_ID, L_INIT_POS, R_INIT_POS, L_RANGE, R_RANGE
from communication.motorC2T.RealMotorT2C import RealMotorT2C
from control import DefaultControllerValues as ControlParams

# Simulator
from simulation.bullet.manipulation import Manipulation

# filter
from control.server_utils import *

GRIPPER = {'left': False, 'right': False}
HAMMER = False
NO_DEVICE_DEBUG = False
LOG = False
PARALLEL = True
FPS = 200
EPS = 1e-5
SUPER_SAFE = False

class IM2ControlServer(IM2ControlServicer):
    def __init__(self, 
                 joint_limit_min: Dict[str, npt.NDArray]={'left': ControlParams.L_JOINT_LIMIT_MIN, 'right': ControlParams.R_JOINT_LIMIT_MIN},
                 joint_limit_max: Dict[str, npt.NDArray]={'left': ControlParams.L_JOINT_LIMIT_MAX, 'right': ControlParams.R_JOINT_LIMIT_MAX},
                 joint_init_offset: npt.NDArray=ControlParams.JOINT_INIT_OFFSET,
                 max_tau: npt.NDArray=ControlParams.MAX_TAU,
                 urdf_list: Dict[str, str]={'left': ControlParams.LEFT_URDF_PATH, 'right': ControlParams.RIGHT_URDF_PATH},
                 arm: List[str]=[LEFT], # use [LEFT, RIGHT] for bimanual
                 gripper: bool=GRIPPER,
                 dof: int=ControlParams.DOF,
                 fps: int=FPS,
                 eps: float=EPS,
                 initialization: bool=True,
                 log: bool=LOG,
                 smooth: bool=False,
                 **kwargs):
        super().__init__()

        self.initialization = initialization
        self.joint_init_offset = joint_init_offset
        self.joint_limit_min = joint_limit_min
        self.joint_limit_max = joint_limit_max
        self.max_tau = max_tau

        # Arm hardwares
        self.can_idx = 0
        self.tmotor_joints: Dict[str, List[Tmotor]] = dict()
        self.rmd_joints: Dict[str, Dict[int, RMD.RMD]] = dict()
        self.cur_joint_pos: Dict[str, npt.NDArray] = dict()
        self.cur_joint_vel: Dict[str, npt.NDArray] = dict()
        self.cur_joint_tau: Dict[str, npt.NDArray] = dict()
        self.cur_grav_tau: Dict[str, npt.NDArray] = dict()

        self.target_joint_pos: Dict[str, npt.NDArray] = dict()
        self.target_joint_vel: Dict[str, npt.NDArray] = dict()
        self.p_gains: Dict[str, npt.NDArray] = dict()
        self.d_gains: Dict[str, npt.NDArray] = dict()
        
        self.dof = dof
        self.arm = arm
        self.pino: Dict[str, PinocchioInterface] = dict()

        self.M = RealMotorT2C()

        # Gripper hardware
        self.grippers: Dict[str, GripperControl] = dict()
        self.gripper_states: Dict[str, float] = dict()

        # No robot for debug
        if NO_DEVICE_DEBUG:
            self.manip = build_manip()

        # initialize
        for arm_name in arm:
            self.initialize_single_arm(arm_name, self.joint_init_offset, urdf_list[arm_name])
            if gripper[arm_name]:
                if arm_name == 'left':
                    self.initialize_gripper(arm_name, id=L_ID, init_pos=L_INIT_POS, range=L_RANGE)
                elif arm_name == 'right':
                    self.initialize_gripper(arm_name, id=R_ID, init_pos=R_INIT_POS, range=R_RANGE)
                else:
                    raise KeyError('Either right or left arm')

        # Controller loop config
        self.fps = fps

        self.time_it = False
        self.global_time = time.time()
        self.latencies = []

        # Filter
        self.smooth = smooth
        if smooth:
            self.window_size = 20
            self.pos_filter = {RobotParams.LEFT: MovingAverageFilter(window_size=self.window_size), 
                               RobotParams.RIGHT: MovingAverageFilter(window_size=self.window_size)}
            self.vel_filter = {RobotParams.LEFT: MovingAverageFilter(window_size=self.window_size), 
                               RobotParams.RIGHT: MovingAverageFilter(window_size=self.window_size)}
            self.tau_filter = {RobotParams.LEFT: MovingAverageFilter(window_size=self.window_size), 
                               RobotParams.RIGHT: MovingAverageFilter(window_size=self.window_size)}


        # Controller loop thread handle
        self.control_thread_handle: threading.Thread = None
        self.stop_event: threading.Event = threading.Event()

    ## Helpers
    @staticmethod
    def _is_proximal(a: npt.NDArray, b:npt.NDArray, eps: float):
        assert a.shape == b.shape, 'a and b must have same dimension'
        return np.linalg.norm(a-b) < eps
    

    @staticmethod
    def compute_desired_torque(pino: PinocchioInterface,
                               target_joint_pos: npt.NDArray, 
                               cur_joint_pos: npt.NDArray, 
                               cur_joint_vel: npt.NDArray,
                               p_gains: npt.NDArray,
                               d_gains: npt.NDArray,
                               max_tau: npt.NDArray,
                               arm: str,
                               safety_coeff: float=0.9,
                               log: bool=False):
        
        cur_grav_tau = getGravityCompensation(pino, cur_joint_pos, cur_joint_vel)

        if arm == 'left':
            if GRIPPER[arm]:
                cur_grav_tau[1] *= 0.75 
                cur_grav_tau[2] *= 0.9
                cur_grav_tau[3] *= 0.9
                cur_grav_tau[4] *= 0.85
                cur_grav_tau[5] *= 0.35
            elif HAMMER:
                cur_grav_tau[1] *= 0.85 
                cur_grav_tau[2] *= 1.0
                cur_grav_tau[3] *= 0.7
                cur_grav_tau[4] *= 1.5
                cur_grav_tau[5] *= 1.5
            else:
                cur_grav_tau[1] *= 0.64
                cur_grav_tau[2] *= 0.6
                cur_grav_tau[3] *= 0.6
                cur_grav_tau[4] *= 0.85
                cur_grav_tau[5] *= 0.9
        
        elif arm == 'right':
            if GRIPPER[arm]:
                cur_grav_tau[1] *= 0.9
                cur_grav_tau[2] *= 0.95
                cur_grav_tau[3] *= 1.0

            else:
                cur_grav_tau[1] *= 0.72
                cur_grav_tau[2] *= 0.68
                cur_grav_tau[3] *= 0.68

            cur_grav_tau[4] *= 0.3
            cur_grav_tau[5] *= 0.3
        
        des_tau = p_gains * (target_joint_pos - cur_joint_pos) - d_gains * cur_joint_vel + cur_grav_tau
        des_tau = np.clip(des_tau, -safety_coeff*max_tau, safety_coeff*max_tau)

        if log:
            np.set_printoptions(precision=4, suppress=True)
            # print(f'[INFO| compute_desired_torque] current gravity compensation is {cur_grav_tau}')
            print(f'[INFO| compute_desired_torque] Excess torque is {des_tau - cur_grav_tau}')

        return des_tau, cur_grav_tau
    

    def initialize_single_arm(self, arm: str, joint_init_offset: npt.NDArray, urdf: str) -> \
                                Tuple[List[Tmotor], RMD.RMD, RMD.RMD, npt.NDArray, npt.NDArray, npt.NDArray]:
        
        try:
            # left: [1, 2, 3, 4], right: [11, 12, 13, 14]
            if arm == 'right':
                id1, id2, id3, id4 = CANIDs.R1, CANIDs.R2, CANIDs.R3, CANIDs.R4
                yaw_socket, pitch_socket, tmotor_socket = CANSockets.R_YAW, CANSockets.R_PITCH, CANSockets.R_TMOTOR
            elif arm == 'left':
                id1, id2, id3, id4 = CANIDs.L1, CANIDs.L2, CANIDs.L3, CANIDs.L4
                yaw_socket, pitch_socket, tmotor_socket = CANSockets.L_YAW, CANSockets.L_PITCH, CANSockets.L_TMOTOR
            else:
                raise ValueError('Do you have more than 2 arms?')
            
            bus0 = can.interface.Bus(bustype="socketcan", channel=yaw_socket, bitrate=1000000)
            time.sleep(0.1)
            joint5 = RMD.RMD(1, bus0)
            time.sleep(0.1)

            bus1 = can.interface.Bus(bustype="socketcan", channel=pitch_socket, bitrate=1000000)
            time.sleep(0.1)
            joint6 = RMD.RMD(1, bus1)
            time.sleep(0.1)
            
            joint1 = Tmotor(can_socket=tmotor_socket, motor_id=id1, motor_type='AK70_10_V1p1')
            joint2 = Tmotor(can_socket=tmotor_socket, motor_id=id2, motor_type='AK70_10_V1p1')
            joint3 = Tmotor(can_socket=tmotor_socket, motor_id=id3, motor_type='AK70_10_V1p1')
            joint4 = Tmotor(can_socket=tmotor_socket, motor_id=id4, motor_type='AK70_10_V1p1')

            tmotor_joint_list = [joint1, joint2, joint3, joint4]

            time.sleep(0.1)
        except:
            if NO_DEVICE_DEBUG:
                print('[DEBUG] Currently debugging the server without physical device')
                tmotor_joint_list = []
                joint5 = None
                joint6 = None
                pass

        self.can_idx+=3

        if self.initialization:

            print("GO TO INITIAL POSE")
            joint_info = enableMotors(tmotor_joint_list)
            print(joint_info)
            time.sleep(0.1)            

            joint5.set_zero(log=True)
            print("joint5 init")
            time.sleep(0.1)

            joint6.set_zero(log=True)
            print("joint6 init")
            time.sleep(0.1)

            print("REMOVE SUPPORT")
            time.sleep(0.1)

        else:
            print("GO TO URDF DEFAULT POSE")
            if not NO_DEVICE_DEBUG:
                joint_info = enableMotorsWithoutSetZero(tmotor_joint_list)
                print(joint_info)
        
        self.pino[arm] = PinocchioInterface(urdf, arm, self.dof)

        self.tmotor_joints[arm] = tmotor_joint_list
        if arm == 'left':
            self.rmd_joints[arm] = {JointNames.L_WRIST_YAW: joint5, JointNames.L_WRIST_PITCH: joint6}
        elif arm == 'right':
            self.rmd_joints[arm] = {JointNames.R_WRIST_YAW: joint5, JointNames.R_WRIST_PITCH: joint6}
        else:
            raise ValueError('Name of an arm should either be left or right')

        if not NO_DEVICE_DEBUG:
            self.cur_joint_pos[arm], self.cur_joint_vel[arm], self.cur_joint_tau[arm] \
                = self.zero_torque_whole_arm(arm, log=False)
        else:
            self.cur_joint_pos[arm], self.cur_joint_vel[arm], self.cur_joint_tau[arm] \
                = np.zeros(self.dof), np.zeros(self.dof), np.zeros(self.dof)
        
        self.p_gains[arm] = np.zeros(self.dof)
        self.d_gains[arm] = ControlParams.DEFAULT_D_GAIN

        self.target_joint_pos[arm] = self.cur_joint_pos[arm].copy()
        self.target_joint_vel[arm] = np.zeros(self.dof)

        return tmotor_joint_list, joint5, joint6


    def initialize_gripper(self, arm: str, **kwargs):
        if not NO_DEVICE_DEBUG:
            self.grippers[arm] = GripperControl(**kwargs)
            self.gripper_states[arm] = self.grippers[arm].read_pos()[0]
        else:
            self.grippers[arm] = None
            if arm == 'left':
                self.gripper_states[arm] = RobotParams.L_GRIPPER[0]
            else:
                self.gripper_states[arm] = RobotParams.R_GRIPPER[0]


        self.gripper_delay = kwargs.get('gripper_delay', 0.001)



    def robot_shutdown(self):
        # Join control thread
        self.stop_event.set()
        self.control_thread_handle.join()

        try:
            for arm in self.cur_joint_pos.keys():
                print(f'Sent zero torque to {arm}')
                self.zero_torque_whole_arm(arm)
        except Exception as e:
            pass

        for tmotor_joint_list in self.tmotor_joints.values():
            disableMotors(tmotor_joint_list)

        for rmd_joints in self.rmd_joints.values():
            for joint in rmd_joints.values():
                joint.motor_stop()
                joint.motor_shutdown()

        print('Motor should be shut down at this point')


    ## HW signals (outgoing)
    def torque_control_whole_arm(self, 
                                 arm: str, 
                                 desired_torques: npt.NDArray,
                                 log: bool=False, 
                                 parallel: bool=False) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        
        tmotor_joints_list = self.tmotor_joints[arm]
        rmd_joints_dict = self.rmd_joints[arm]
        joint_offset = self.joint_init_offset
        max_tau = self.max_tau
        joint_limit_min = self.joint_limit_min[arm]
        joint_limit_max = self.joint_limit_max[arm]

        rmd_joint_pos = []
        rmd_joint_vel = []
        rmd_joint_tau = []

        if parallel:
            def control_tmotor():
                return torque_control_Tmotor(self.M, tmotor_joints_list, desired_torques, 
                                            offset=joint_offset[:len(tmotor_joints_list)])

            def control_rmd(rmd: RMD.RMD, joint_idx: int):
                return rmd.torque_control_RMD(joint_idx-1, self.M, desired_torques[joint_idx-1], 
                                            offset=joint_offset[joint_idx-1], log=False)

            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit the Tmotor control task
                futures = [executor.submit(control_tmotor)]

                # Submit the RMD control tasks
                for id, rmd in rmd_joints_dict.items():
                    joint_idx = id if arm == 'left' else id - len(desired_torques)  # len(desired_torques) == DOF
                    futures.append(executor.submit(control_rmd, rmd, joint_idx))

                # Wait for all tasks to complete
                tmotor_joint_pos, tmotor_joint_vel, tmotor_joint_tau = futures[0].result()
                for future in futures[1:]:
                    pos, vel, tau = future.result()
                    rmd_joint_pos.append(pos)
                    rmd_joint_vel.append(vel)
                    rmd_joint_tau.append(tau)

        else:
            # Control Tmotors
            tmotor_joint_pos, tmotor_joint_vel, tmotor_joint_tau = torque_control_Tmotor(self.M, 
                                                                                         tmotor_joints_list, 
                                                                                         desired_torques, 
                                                                                         offset=joint_offset[:len(tmotor_joints_list)])
            

            # Control RMD motors
            for id, rmd in rmd_joints_dict.items():
                joint_idx = id if arm == 'left' else id - len(desired_torques)         # len(desired_torques) == DOF
                pos, vel, tau = rmd.torque_control_RMD(joint_idx-1, self.M, desired_torques[joint_idx-1], offset=joint_offset[joint_idx-1], log=False)
                rmd_joint_pos.append(pos)
                rmd_joint_vel.append(vel)
                rmd_joint_tau.append(tau)

        cur_joint_pos = np.concatenate((tmotor_joint_pos, rmd_joint_pos))
        cur_joint_vel = np.concatenate((tmotor_joint_vel, rmd_joint_vel))
        cur_joint_tau = np.concatenate((tmotor_joint_tau, rmd_joint_tau))

        self.cur_joint_pos[arm] = cur_joint_pos
        self.cur_joint_vel[arm] = cur_joint_vel
        self.cur_joint_tau[arm] = cur_joint_tau

        if np.any(np.abs(cur_joint_tau) > max_tau): 
            raise ValueError(f'[Error | IM2Server] max torque exceeded, Torque = {desired_torques}')

        if np.any(cur_joint_pos < joint_limit_min) or np.any(cur_joint_pos > joint_limit_max):
            raise ValueError(f'[Error | IM2Server] joint limit exceeded, joint_pos = {cur_joint_pos/np.pi*180}')
        
        return cur_joint_pos, cur_joint_vel, cur_joint_tau


    def pos_control_whole_arm(self, arm: str, 
                              desired_pos: npt.NDArray,
                              desired_vel: npt.NDArray,
                              ff_torque=None,
                              kps=ControlParams.POS_P_GAIN,
                              kds=ControlParams.POS_D_GAIN,
                              log: bool=False, 
                              parallel: bool=False) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        
        tmotor_joints_list = self.tmotor_joints[arm]
        rmd_joints_dict = self.rmd_joints[arm]
        joint_offset = self.joint_init_offset
        max_tau = self.max_tau
        joint_limit_min = self.joint_limit_min[arm]
        joint_limit_max = self.joint_limit_max[arm]

        rmd_joint_pos = []
        rmd_joint_vel = []
        rmd_joint_tau = []

        if parallel:
            def control_tmotor():
                return urdfJointPDControl(self.M,
                                          tmotor_joints_list, 
                                          desired_pos,
                                        #   desired_vel,
                                          np.zeros(ControlParams.DOF), 
                                          ff_torque=ff_torque, 
                                          kps=kps,
                                          kds=kds)

            def control_rmd(rmd: RMD.RMD, joint_idx: int):
                # return rmd.pos_control_degree(desired_pos[joint_idx-1]*180/np.pi, 100, log=False)
                return rmd.ff_pd_RMD(float(desired_pos[joint_idx-1]),
                                    #  float(desired_vel[joint_idx-1]),
                                     0,
                                     kps[joint_idx-1],
                                     kds[joint_idx-1],
                                     ff_torque[joint_idx-1],
                                     log=False)

            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit the Tmotor control task
                futures = [executor.submit(control_tmotor)]

                # Submit the RMD control tasks
                for id, rmd in rmd_joints_dict.items():
                    joint_idx = id if arm == 'left' else id - ControlParams.DOF
                    futures.append(executor.submit(control_rmd, rmd, joint_idx))

                # Wait for all tasks to complete
                tmotor_joint_pos, tmotor_joint_vel, tmotor_joint_tau = futures[0].result()
                for future in futures[1:]:
                    pos, vel, tau = future.result()
                    rmd_joint_pos.append(pos)
                    rmd_joint_vel.append(vel)
                    rmd_joint_tau.append(tau)

        else:
            # Control Tmotors
            tmotor_joint_pos, tmotor_joint_vel, tmotor_joint_tau = urdfJointPDControl(self.M,
                                                                                      tmotor_joints_list, 
                                                                                      desired_pos,
                                                                                      desired_vel, 
                                                                                      ff_torque=ff_torque, 
                                                                                      kps=kps,
                                                                                      kds=kds)
            
            # Control RMD motors
            for id, rmd in rmd_joints_dict.items():
                joint_idx = id if arm == 'left' else id - self.dof         # len(desired_torques) == DOF
                pos, vel, tau = rmd.ff_pd_RMD(float(desired_pos[joint_idx-1]),
                                              float(desired_vel[joint_idx-1]),
                                              kps[joint_idx-1],
                                              kds[joint_idx-1],
                                              ff_torque[joint_idx-1],
                                              log=False)
                rmd_joint_pos.append(pos)
                rmd_joint_vel.append(vel)
                rmd_joint_tau.append(tau)

        cur_joint_pos = np.concatenate((tmotor_joint_pos, rmd_joint_pos))
        cur_joint_vel = np.concatenate((tmotor_joint_vel, rmd_joint_vel))
        cur_joint_tau = np.concatenate((tmotor_joint_tau, rmd_joint_tau))

        self.cur_joint_pos[arm] = cur_joint_pos
        self.cur_joint_vel[arm] = cur_joint_vel
        self.cur_joint_tau[arm] = cur_joint_tau

        if np.any(cur_joint_pos < joint_limit_min) or np.any(cur_joint_pos > joint_limit_max):
            raise ValueError(f'[Error | IM2Server] joint limit exceeded, joint_pos = {cur_joint_pos/np.pi*180}')
        
        return cur_joint_pos, cur_joint_vel, cur_joint_tau


    def zero_torque_whole_arm(self, arm: str, log: bool=False) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        return self.torque_control_whole_arm(arm, np.zeros(self.dof), log=log)

    ## Control Loop
    def initiate_controller(self, log=False, parallel=False):
        def control_loop(log=log):
            step = 0
            while not self.stop_event.is_set():
                st = time.time()
                for arm in self.cur_joint_pos.keys():
                    # Checking proximity should be done by the controller algorithm
                    # if not IM2ControlServer._is_proximal(self.cur_joint_pos[arm], self.target_joint_pos[arm]):
                    des_tau, grav_tau = IM2ControlServer.compute_desired_torque(self.pino[arm], self.target_joint_pos[arm], 
                                                                                self.cur_joint_pos[arm], self.cur_joint_vel[arm],
                                                                                self.p_gains[arm], self.d_gains[arm],
                                                                                self.max_tau, arm, log=False)
                    # des_tau = 0.9*des_tau
                    self.cur_grav_tau[arm] = grav_tau
                    try:
                        if self.smooth:
                            filtered_des_tau = self.tau_filter[arm].filter(des_tau)
                            des_tau = filtered_des_tau

                        if NO_DEVICE_DEBUG:
                            self.manip.robot.setJointStates(arm, self.target_joint_pos[arm])
                        else:
                            self.torque_control_whole_arm(arm, des_tau, log=log, parallel=parallel)
                        if log and step%200==0:
                            np.set_printoptions(precision=3, suppress=True)
                            print(f'[DEBUG | control loop] msg from client | arm: {arm} \
                                \n | target: {180./np.pi*self.target_joint_pos[arm]} | current: {180./np.pi*self.cur_joint_pos[arm]} | desired: {des_tau} | grav: {grav_tau} | excess: {des_tau - grav_tau}')
                    except Exception as e:
                        if NO_DEVICE_DEBUG and log and step%200==0:
                            print(f'[DEBUG | no device debugging inside control loop] Current Joint Position: {self.cur_joint_pos[arm]} | Target Joint Position: {self.target_joint_pos[arm]}')
                        if not NO_DEVICE_DEBUG and step%10==0:
                            print('[ERROR | encountered error during operation. Compliant mode activated.]')
                        self.p_gains[arm] = np.zeros(self.dof)
                        self.d_gains[arm] = RobotParams.POS_D_GAIN
                        if SUPER_SAFE:
                            self.zero_torque_whole_arm(arm, log=log)

                duration = time.time() - st
                step = step + 1 if (step+1)%100!=0 else 0
                if self.time_it:
                    latency = time.time() - self.global_time
                    self.latencies.append(latency)
                    if len(self.latencies) > 1:
                        print(f'[TIMER] average latency: {latency*1000} ms')
                    self.global_time = time.time()
        self.control_thread_handle = threading.Thread(name='control loop', target=control_loop, daemon=False)
        self.control_thread_handle.start()



    def initiate_pos_controller(self, log=False, parallel=False):
        # Reset motors
        if not NO_DEVICE_DEBUG:
            for rmd_joints in self.rmd_joints.values():
                for joint in rmd_joints.values():
                    joint.motor_stop()
                    joint.motor_shutdown()

        for arm in self.cur_joint_pos.keys():
            self.p_gains[arm] = ControlParams.POS_P_GAIN
            self.d_gains[arm] = ControlParams.POS_D_GAIN

        def control_loop(log=log):
            step = 0
            while not self.stop_event.is_set():
                st = time.time()
                for arm in self.cur_joint_pos.keys():
                    # Checking proximity should be done by the controller algorithm
                    _, grav_tau = IM2ControlServer.compute_desired_torque(self.pino[arm],
                                                                          self.target_joint_pos[arm],
                                                                          self.cur_joint_pos[arm],
                                                                          self.cur_joint_vel[arm],
                                                                          np.zeros(self.dof),
                                                                          np.zeros(self.dof),
                                                                          self.max_tau,
                                                                          arm,
                                                                          log=False)
                    self.cur_grav_tau[arm] = grav_tau

                    try:
                        if self.smooth:
                            target_joint_pos = self.pos_filter[arm].filter(self.target_joint_pos[arm])
                            target_joint_vel = self.vel_filter[arm].filter(self.target_joint_vel[arm])
                        else:
                            target_joint_pos = self.target_joint_pos[arm]
                            target_joint_vel = self.target_joint_vel[arm]

                        if NO_DEVICE_DEBUG:
                            self.manip.robot.setJointStates(arm, target_joint_pos)
                            motor_tau = None
                        else:
                            _, _, motor_tau = self.pos_control_whole_arm(arm, target_joint_pos, target_joint_vel, ff_torque=grav_tau, kps=self.p_gains[arm], kds=self.d_gains[arm], log=False, parallel=parallel)
                        if log and step%5000==0 and arm == 'right':
                            step = 0
                            np.set_printoptions(precision=3, suppress=True)
                            print(f'[DEBUG | control loop] msg from client | arm: {arm} \
                                \n | target: {180./np.pi*target_joint_pos} | current: {180./np.pi*self.cur_joint_pos[arm]} | Kp: {self.p_gains[arm]} | Kd: {self.d_gains[arm]} | tau: {motor_tau}')
                    except Exception as e:
                        if NO_DEVICE_DEBUG and log and step%5000==0:
                            print(f'[DEBUG | no device debugging inside control loop] Current Joint Position: {self.cur_joint_pos[arm]} | Target Joint Position: {self.target_joint_pos[arm]}')
                        if not NO_DEVICE_DEBUG and step%1000==0:
                            print(f'[ERROR | {arm}-arm encountered error during operation. Compliant mode activated. Error={e}]')
                        self.p_gains[arm] = np.zeros(self.dof)
                        self.d_gains[arm] = RobotParams.SNATCH_D_GAIN*2
                        if SUPER_SAFE:
                            self.zero_torque_whole_arm(arm, log=log)

                duration = time.time() - st
                step = step + 1
                if self.time_it:
                    latency = time.time() - self.global_time
                    self.latencies.append(latency)
                    if len(self.latencies) > 1:
                        print(f'[TIMER] average latency: {latency*1000} ms')
                    self.global_time = time.time()
        
        self.control_thread_handle = threading.Thread(name='control loop', target=control_loop, daemon=False)
        self.control_thread_handle.start()


    ## GRPC functions
    def GetMaxTorques(self, request: Empty, context) -> NDArray:
        return NDArray(array=self.max_tau.tolist())
    

    def GetJointState(self, request: String, context):
        arm = request.value
        out_msg = JointState(current_position=self.cur_joint_pos[arm],
                             current_velocity=self.cur_joint_vel[arm],
                             current_torque=self.cur_joint_tau[arm],
                             current_gravity=self.cur_grav_tau[arm])

        return out_msg


    def GetJointInitOffset(self, request: Empty, context) -> NDArray:
        return NDArray(array=self.joint_init_offset.tolist())


    def GetEEPose(self, request: String, context) -> Pose:
        arm = request.value
        ee_pose: npt.NDArray = get_ee_pose(self.pino[arm], self.cur_joint_pos[arm], self.cur_joint_vel[arm])

        out_msg = Pose(position=ee_pose[:3].tolist(),
                       orientation=ee_pose[3:].tolist())
        
        return out_msg


    def GetJacobian(self, request: String, context) -> Jacobian:
        arm = request.value
        lin_j, ang_j = getJacobian(self.pino[arm], self.cur_joint_pos[arm], self.cur_joint_vel[arm])
        
        
        if LOG:
            np.set_printoptions(precision=3, suppress=True)
            print(f'[DEBUG | GetJacobian] | Linear Jacobian = {lin_j} | Angular Jacobian = {ang_j}')

        out_msg = Jacobian(linear_jacobian=lin_j.flatten().tolist(),
                           angular_jacobian=ang_j.flatten().tolist())

        return out_msg


    def TorqueControl(self, request: TorqueControlIn, context) -> Empty:
        arm: str = request.arm
        des_tau = np.array(request.desired_torques)

        self.torque_control_whole_arm(arm, des_tau, log=LOG, parallel=PARALLEL)

        return Empty()
    

    def PositionControl(self, request: PositionControlIn, context) -> Empty:
        arm: str = request.arm
        target_joint_pos: List[float] = request.desired_position
        target_joint_vel: List[float] = request.desired_velocity
        p_gains: List[float] = request.desired_p_gains
        d_gains: List[float] = request.desired_d_gains

        self.target_joint_pos[arm] = np.array(target_joint_pos)
        self.target_joint_vel[arm] = np.array(target_joint_vel)
        self.p_gains[arm] = np.array(p_gains)
        self.d_gains[arm] = np.array(d_gains)

        return Empty()
        

    def JointPDControl(self, request: JointPDControlIn, context) -> Empty:
        self.time_it = False
        arm: str = request.arm
        target_joint_pos: List[float] = request.target_joint_position
        p_gains: List[float] = request.p_gains
        d_gains: List[float] = request.d_gains

        self.target_joint_pos[arm] = np.array(target_joint_pos)
        self.p_gains[arm] = np.array(p_gains)
        self.d_gains[arm] = np.array(d_gains)

        return Empty()
    

    def Shutdown(self, request: Empty, context):
        result = True
        try:
            self.robot_shutdown()
        except:
            result = False

        return Boolean(value=result)


    def SetGripperState(self, request: GripperControlIn, context):
        arm: str = request.arm
        value: float = request.value
        if not NO_DEVICE_DEBUG:
            self.grippers[arm].move(value, self.gripper_delay)

        return Empty()


    def GetGripperState(self, request: String, context):
        arm: str = request.value
        gripper_state: float = self.gripper_states[arm]

        out_msg = Float(value=gripper_state)

        return out_msg


def serve(ip:str=ControlParams.IP, port:str=ControlParams.PORT, initialization: bool=True, mode: str='torque', smooth: bool=False):
    try:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        im2control_server = IM2ControlServer(initialization=initialization, smooth=smooth)
        if mode in ['torque']:
            im2control_server.initiate_controller(log=LOG, parallel=PARALLEL)
        elif mode in ['pos', 'position']:
            im2control_server.initiate_pos_controller(log=LOG, parallel=PARALLEL)
        else:
            raise NotImplementedError
        add_IM2ControlServicer_to_server(im2control_server, server)
        server.add_insecure_port(f"{ip}:{port}")
        server.start()
        print('IM2 controller server started, listening on ' + port)
        server.wait_for_termination()
    except:
        print('[Server] Terminated')
        im2control_server.robot_shutdown()



def get_params():
    parser = argparse.ArgumentParser(description='IM2Control server')
    parser.add_argument('--init', action='store_true', default=False, help='Set true for no-device debug')
    parser.add_argument('--ip', type=str, default='[::]')
    parser.add_argument('--port', type=str, default='50051')
    parser.add_argument('--mode', type=str, default='torque', help='[torque, pos, position]')
    parser.add_argument('--smooth', action='store_true', default=False, help='Moving average filter on the action')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig()
    params = get_params()
    serve(ip=params.ip, port=params.port, initialization=params.init, mode=params.mode, smooth=params.smooth)
