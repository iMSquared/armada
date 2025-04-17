# import pinocchio

import pinocchio as pin
import numpy as np
from scipy.spatial.transform import Rotation as R
from control import DefaultControllerValues as RobotParams

LEFT = RobotParams.LEFT
RIGHT = RobotParams.RIGHT
EE_NAMES = RobotParams.EE_NAMES

class PinocchioInterface:
    def __init__(self, 
                 urdf_file, 
                 arm, 
                 n_joint, 
                 control_freq=0.5):
        
        self.urdf_file_path = urdf_file
        self.arm = arm
        self.end_effector_name = EE_NAMES[self.arm]
        self.n_joint = n_joint
        self.control_freq = control_freq
        self.q = np.zeros((self.n_joint,))
        self.q_dot = np.zeros((self.n_joint,))
        self.q_ddot = np.zeros((self.n_joint,))
        self.end_effector_position = None
        self.end_effector_linear_velocity = np.zeros((3,))
        self.end_effector_angular_velocity = np.zeros((3,))
        self.end_effector_acceleration = None

        self.M_Matrix = np.zeros((self.n_joint, self.n_joint))
        self.C_Matrix = np.zeros((self.n_joint,))
        self.G_Matrix = np.zeros((self.n_joint,))
        self.J_Matrix = np.zeros((6, self.n_joint,))
        self.dJ_Matrix = np.zeros((6, self.n_joint,))
        self.Linear_Jacobian_M = np.zeros((3,self.n_joint))
        self.Angular_Jacobian_M = np.zeros((3,self.n_joint))
        self.Linear_Jacobian_dot_M = np.zeros((3,self.n_joint))
        self.Angular_Jacobian_dot_M = np.zeros((3,self.n_joint))
        self.Calculated_Torque = np.zeros((self.n_joint, 1))
        self.Target_Torque = []

        self.init_diff_flag = False
        self.D_x = None
        self.D_xold = None
        self.D_xold2 = None
        
        self._model = pin.buildModelFromUrdf(self.urdf_file_path)
        print('Pinocchio _model name: ' + self._model.name)
        self._data = self._model.createData()
        self.end_effector_id = self._model.getFrameId(self.end_effector_name)


    ######################################################## Set
    def SetRobotParameter(self, joint_position, joint_velocity):
        self.q = np.array(joint_position)
        self.q_dot = np.array(joint_velocity)
        self.q_ddot = self.Diff(self.q_dot)

        pin.forwardKinematics(self._model,self._data, self.q)
        pin.computeJointJacobians(self._model, self._data, self.q)
        pin.computeJointJacobiansTimeVariation(self._model, self._data, self.q, self.q_dot)

    def SetJacobian(self):
        self.J_Matrix = pin.getFrameJacobian(self._model, self._data, self.end_effector_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        self.dJ_Matrix = pin.getFrameJacobianTimeVariation(self._model, self._data, self.end_effector_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        self.Linear_Jacobian_M = self.J_Matrix[0:3, :]
        self.Angular_Jacobian_M = self.J_Matrix[3:6, :]
        self.Linear_Jacobian_dot_M = self.dJ_Matrix[0:3, :]
        self.Angular_Jacobian_dot_M = self.dJ_Matrix[3:6, :]

    # def get_link_jacobian_dot_times_qdot(self):
    #     frame_id = self._model.getFrameId(self.end_effector_name)

    #     pin.forwardKinematics(self._model, self._data, self.q, self.q_dot,
    #                           0*self.q_dot)
    #     jdot_qdot = pin.getFrameClassicalAcceleration(
    #         self._model, self._data, frame_id,
    #         pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    #     ret = np.zeros_like(jdot_qdot)
    #     ret[0:3] = jdot_qdot.angular
    #     ret[3:6] = jdot_qdot.linear

    #     print(f'ret[3:6] = {ret[0:3]}')
    

    def Setlinkvel(self):
        frame_id = self._model.getFrameId(self.end_effector_name)

        spatial_vel = pin.getFrameVelocity(
            self._model, self._data, frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        self.end_effector_angular_velocity = spatial_vel.angular
        self.end_effector_linear_velocity = spatial_vel.linear


    def SetGravity(self):
        pin.computeGeneralizedGravity(self._model, self._data, self.q)
        self.G_Matrix = self._data.g
        self.G_Matrix = self.G_Matrix.reshape((self.n_joint,1))

    def SetCoriolis(self):
        self.C_Matrix = pin.nonLinearEffects(self._model, self._data, self.q, self.q_dot)
        self.C_Matrix = self.C_Matrix.reshape((self.n_joint,1))
        self.C_Matrix = self.C_Matrix - self.GetGravity()
        # pin.computeCoriolisMatrix(self._model, self._data, self.q, self.q_dot)
        # self.C_Matrix@self.q_dot
    

    def SetInertia(self):
        self.M_Matrix = pin.crba(self._model, self._data, self.q)


    def SetKinematics(self): 
        pin.updateFramePlacement(self._model, self._data, self.end_effector_id)

        self.end_effector_state = self._data.oMf[self.end_effector_id] #if(9,1) 
        # print(f'self._data.oMf[self.end_effector_id] = {self._data.oMf[self.end_effector_id]}')
        self.end_effector_rotation = self.end_effector_state.rotation
        self.end_effector_position = self.end_effector_state.translation

        self.SetJacobian()
        self.Setlinkvel()
        # self.get_link_jacobian_dot_times_qdot()
        self.end_effector_acceleration = np.dot(self.Linear_Jacobian_dot_M, self.q_dot) + np.dot(self.Linear_Jacobian_M, self.q_ddot)


    def SetDynamics(self, pos_d, vel_d, acc_d, Kp, Kd):
        self.SetInertia()
        self.SetGravity()
        self.SetCoriolis()

        desired_pos = np.array(pos_d).reshape((3,1))
        desired_vel = np.array(vel_d).reshape((3,1))
        desired_acc = np.array(acc_d).reshape((3,1))

        Kp_gain = np.array([[Kp[0], 0, 0], [0, Kp[1], 0], [0, 0, Kp[2]]])
        Kd_gain = np.array([[Kd[0], 0, 0], [0, Kd[1], 0], [0, 0, Kd[2]]])

        self.end_effector_linear_velocity = self.end_effector_linear_velocity.reshape((3,1))

        err = desired_pos - self.end_effector_position
        err_d = desired_vel - self.end_effector_linear_velocity

        xddot_des = desired_acc + np.dot(Kp_gain, err) + np.dot(Kd_gain, err_d)

        qddot_des = np.dot(np.linalg.pinv(self.Linear_Jacobian_M, rcond=1e-3), xddot_des)

        self.Calculated_Torque = np.dot(self.M_Matrix, qddot_des) + self.C_Matrix + self.G_Matrix

        print(f'err = {err}')
        # print(f'xddot_des = {xddot_des}')
        # print(f'qddot_des = {qddot_des}')
        print(f'self.Calculated_Torque = {self.Calculated_Torque}')




        


    ######################################################## Numerical
    def Diff(self, x_):
        if(self.init_diff_flag == False):
            self.init_diff_flag = True
            self.D_xold2 = self.D_xold = x_

        self.D_x = x_
        xdot=(3*self.D_x-4*self.D_xold+self.D_xold2)*0.5*self.control_freq
        self.D_xold2 = self.D_xold
        self.D_xold = self.D_x
        return xdot


  


    ######################################################### Get
    def GetGravity(self):
        return self.G_Matrix

    def GetEEPos(self):
        return self.end_effector_position 

    def GetJacobian(self):
        return self.Linear_Jacobian_M, self.Angular_Jacobian_M

    def GetCorilois(self):
        return self.C_Matrix
    
    def GetInertia(self):
        return self.M_Matrix
    
    def GetDynamics(self):
        self.Calculated_Torque.reshape((self.n_joint))
        return self.Calculated_Torque
    
def getGravityCompensation(p:PinocchioInterface, current_joint_pos, current_joint_vel):
    p.SetRobotParameter(current_joint_pos, current_joint_vel)
    p.SetGravity()
    return p.GetGravity().squeeze()

def getJacobian(p:PinocchioInterface, current_joint_pos, current_joint_vel):
    p.SetRobotParameter(current_joint_pos, current_joint_vel)
    p.SetJacobian()
    return p.GetJacobian()

def get_delta_dof_pos(delta_pose, jacobian):
    lambda_val = 0.1
    jacobian_T = np.transpose(jacobian)
    lambda_matrix = (lambda_val ** 2) * np.eye(jacobian.shape[0])
    delta_dof_pos = jacobian_T @ np.linalg.inv(jacobian @ jacobian_T + lambda_matrix) @ delta_pose
    return delta_dof_pos

def get_ee_pose(p:PinocchioInterface, current_joint_pos, current_joint_vel):
    p.SetRobotParameter(current_joint_pos, current_joint_vel)
    p.SetKinematics()

    hand_pos = p.end_effector_position
    hand_rotm = p.end_effector_rotation
    hand_quat = R.from_matrix(hand_rotm).as_quat()

    return np.concatenate((hand_pos, hand_quat))

def transform_jacobian(jac_pin):

    jac_IG = np.zeros((6, 6))
    jac_IG[0] = -jac_pin[1]
    jac_IG[1] =  jac_pin[0]
    jac_IG[2] =  jac_pin[2]
    jac_IG[3] = -jac_pin[4]
    jac_IG[4] =  jac_pin[3]
    jac_IG[5] =  jac_pin[5]

    return jac_IG