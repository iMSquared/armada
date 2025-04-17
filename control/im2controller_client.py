import sys
from pathlib import Path
BASEDIR = Path(__file__).parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.append(str(BASEDIR))

import time
import grpc
from control.im2control_pb2 import *
from control.im2control_pb2_grpc import IM2ControlStub
from typing import List, Tuple, Dict
import numpy.typing as npt

# Pinocchio
from simulation.utils.PinocchioInterface import *

# Control
from control import DefaultControllerValues as ControlParams

class IM2Client:
    def __init__(self,
                 ip: str=ControlParams.IP, 
                 port: str=ControlParams.PORT, 
                 dof: int=ControlParams.DOF):
        try: 
            self.channel = grpc.insecure_channel(f'{ip}:{port}')
            self.stub = IM2ControlStub(self.channel)
        except:
            print(f'IM2Control server connection failed at {ip}:{port}')
            exit()

        self.max_tau = None
        self.joint_init_offset = None
        self.dof = dof

        # max_tau, joint_init_offset
        self.get_robot_properties()


    ## Get robot properties
    def get_robot_properties(self):
        if self.max_tau is None:
            max_tau_msg: NDArray = self.stub.GetMaxTorques(Empty())
            self.max_tau = np.array(max_tau_msg.array)
        if self.joint_init_offset is None:
            joint_init_offset_msg: NDArray = self.stub.GetJointInitOffset(Empty())
            self.joint_init_offset = np.array(joint_init_offset_msg.array)

        return self.max_tau, self.joint_init_offset

    ## Send torque control command to the IM2 control server
    def torque_control(self, 
                       arm:str, 
                       desired_torques: npt.NDArray):

        assert arm in ['left', 'right'], 'arm must be either left or right'
        
        # Format input
        in_msg = TorqueControlIn(arm=arm, desired_torques=desired_torques.tolist())
        
        # Call the server
        self.stub.TorqueControl(in_msg)

    
    ## Send position control command to the IM2 control server
    def position_control(self, 
                         arm:str, 
                         desired_position: npt.NDArray, 
                         desired_velocity: npt.NDArray = np.zeros(ControlParams.DOF),
                         desired_p_gains: npt.NDArray = ControlParams.POS_P_GAIN,
                         desired_d_gains: npt.NDArray = ControlParams.POS_D_GAIN):

        assert arm in ['left', 'right'], 'arm must be either left or right'
        
        # Format input
        in_msg = PositionControlIn(arm=arm, 
                                   desired_position=desired_position, 
                                   desired_velocity=desired_velocity,
                                   desired_p_gains=desired_p_gains,
                                   desired_d_gains=desired_d_gains)
        
        # Call the server
        self.stub.PositionControl(in_msg)
    
    

    ## Get current state
    def get_current_states(self, arm:str) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        get_current_in_msg = String(value=arm)

        joint_state_msg: JointState = self.stub.GetJointState(get_current_in_msg)

        cur_pos = np.array(joint_state_msg.current_position)
        cur_vel = np.array(joint_state_msg.current_velocity)
        cur_tau = np.array(joint_state_msg.current_torque)
        cur_grav = np.array(joint_state_msg.current_gravity)

        return cur_pos, cur_vel, cur_tau, cur_grav
    

    def get_ee_pose(self, arm:str) -> Tuple[npt.NDArray, npt.NDArray]:
        in_msg = String(value=arm)
        out_msg: Pose = self.stub.GetEEPose(in_msg)

        pos = np.array(out_msg.position)
        orn = np.array(out_msg.orientation)

        return pos, orn
    

    def get_jacobian(self, arm:str) -> Tuple[npt.NDArray, npt.NDArray]:
        in_msg = String(value=arm)
        out_msg: Jacobian = self.stub.GetJacobian(in_msg)

        lin_j = np.array(out_msg.linear_jacobian).reshape((3, self.dof))
        ang_j = np.array(out_msg.angular_jacobian).reshape((3, self.dof))

        return lin_j, ang_j

    def joint_pd_control(self, arm:str, joint_target_pos: npt.NDArray, p_gains: npt.NDArray, d_gains: npt.NDArray):
        pd_joint_in_msg = JointPDControlIn(arm=arm, target_joint_position=joint_target_pos, p_gains=p_gains, d_gains=d_gains)
        dummy_future = self.stub.JointPDControl.future(pd_joint_in_msg)

        return dummy_future
    

    def set_gripper_state(self, arm: str, pos: float):
        in_msg = GripperControlIn(arm=arm, value=pos)
        dummy_future = self.stub.SetGripperState.future(in_msg)

        return dummy_future
    

    def get_gripper_state(self, arm: str) -> float:
        in_msg = String(value=arm)
        out_msg: Float = self.stub.GetGripperState(in_msg)

        return out_msg.value


    ## Close connection to the server
    def close_channel(self):
        self.channel.close()


def run():
    im2client = IM2Client(ip=ControlParams.IP, port=ControlParams.PORT)
    print('Connected to IM2control server')
    
    ######################################
    ## Do whatever you want to do below ##
    ######################################

    target_joint_pos = np.array([0, 0, 0, 0, 0, 0])
    p_gains = np.array([1.8, 2.2, 1.6, 1.6, 0.7, 0.7])
    d_gains = np.array([0.06, 0.08, 0.1, 0.1, 0.04, 0.04])

    im2client.close_channel()


if __name__ == '__main__':
    run()
