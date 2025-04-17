import sys
from pathlib import Path
BASEDIR = Path(__file__).parent.parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.append(str(BASEDIR))

from typing import Tuple, List, Dict
import numpy.typing as npt
from pybullet_utils.bullet_client import BulletClient
from pathlib import Path
from enum import IntEnum
import pybullet_data
import numpy as np
import pybullet as p
import time

from simulation.bullet.robot import Robot, LEFT, RIGHT, BIMANUAL
from control import DefaultControllerValues as RobotParams

URDF_NAME = 'simulation/assets/urdf/RobotBimanualV7/urdf/Simplify_Robot_real_hammer.urdf'        # 'LinkageURDF3'
DEBUG = True
EE_NAME = {LEFT: 'link6', RIGHT: 'link12'}

class BulletEnvironment:
    def __init__(self, 
                 start_pos: npt.ArrayLike, 
                 start_orn: npt.ArrayLike,
                 joint_min: npt.ArrayLike,
                 joint_max: npt.ArrayLike,
                 robot_name: str=None,
                 arm: str='all',
                 ee_name: Dict[str, str]=EE_NAME,
                 scene_on: bool=False, 
                 debug: bool=False, 
                 log: bool=False, 
                 **kwargs):
        
        self.bc = self.load_simulator(**kwargs)
        self.robot = Robot(self.bc, 
                           start_pos, 
                           start_orn, 
                           urdf_name=robot_name, 
                           arm=arm, 
                           ee_name=ee_name, 
                           joint_min=joint_min, 
                           joint_max=joint_max, 
                           debug=debug, 
                           log=log)
        self.log = log

        if scene_on:
            self.scene = self.load_scene()


    def load_simulator(self, load_plane: bool=True, gravity: bool=False, gui=True, **kwargs):
        if gui:
            bc = BulletClient(connection_mode=p.GUI)
        else:
            bc = BulletClient(connection_mode=p.DIRECT)
        bc.setAdditionalSearchPath(pybullet_data.getDataPath())

        if load_plane:
            bc.loadURDF("plane.urdf")

        # Simulation setting
        bc.setRealTimeSimulation(False)
        if gravity:
            bc.setGravity(0, 0, -9.81)
        else:
            bc.setGravity(0, 0, 0)
        bc.stepSimulation()

        return bc
    

    def load_scene(self, table_height: float=0.5065):
        tableShape = (0.2, 0.4, table_height/2)
        tablePosition = (0.3, 0.0, table_height/2)
        boxColor = (np.array([170, 170, 170, 170]) / 255.0).tolist()
        tableVisualShapeId = self.bc.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=tableShape,
            rgbaColor=boxColor
        )
        tableCollisionShapeId = self.bc.createCollisionShape(
            shapeType=p.GEOM_BOX, 
            halfExtents=tableShape
        )
        tableId = self.bc.createMultiBody(
            baseMass=10,
            baseCollisionShapeIndex=tableCollisionShapeId,
            baseVisualShapeIndex=tableVisualShapeId,
            basePosition=tablePosition
        )

        objs = {'table': tableId}

        return objs

    def run_debug_loop(self, arm: str):
        init_EE_pos = self.robot.get_ee_pose(arm)
        uid = self.robot.uid
        ee = self.robot.ee
        link_id_dict = self.robot.link_id_dict
        joint_id_dict = self.robot.joint_id_dict
        dof = self.robot.dof
        mode = self.robot.mode
        
        thetas = []
        for param in self.robot.debugparams:
            thetas.append(self.bc.readUserDebugParameter(param))

        while True:
            self.bc.stepSimulation()

            if mode == 'jp':
                targetEEPosition = init_EE_pos + np.array(thetas)
                jointPoses = self.bc.calculateInverseKinematics(uid, link_id_dict[ee], targetEEPosition)
                for joint_id in range(dof-1):
                    self.bc.setJointMotorControl2(uid, joint_id, self.bc.POSITION_CONTROL, jointPoses[joint_id])

            elif mode == 'inv_dyn':
                targetEEPosition = init_EE_pos + np.array(thetas)
                cur_EE_state = self.bc.getLinkState(uid, link_id_dict[ee], computeLinkVelocity=1)
                cur_EE_pos = cur_EE_state[0]
                cur_EE_vel = cur_EE_state[6]
                dpose = targetEEPosition - cur_EE_pos

                mpos, mvel, mtorq = self.robot.getMotorJointStates()
                pos, vel, torq = self.robot.getJointStates()

                result = self.bc.getLinkState(uid, 
                                              link_id_dict[ee],
                                              computeLinkVelocity=1,
                                              computeForwardKinematics=1)
                link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result

                zero_vec = [0.0] * len(mpos)
                jac_t, jac_r = self.bc.calculateJacobian(uid, link_id_dict[ee], com_trn, mpos, zero_vec, zero_vec)
                jac_t = np.array(jac_t)
                mm = self.bc.calculateMassMatrix(uid, pos)
                
                def control_osc(j_eef, hand_vel, mm, dpose, kp):
                    kd = 2.0 * np.sqrt(kp)
                    mm_inv = np.linalg.inv(mm)
                    m_eef_inv = np.matmul(np.matmul(j_eef, mm_inv), np.transpose(j_eef))
                    m_eef = np.linalg.inv(m_eef_inv)
                    u = np.matmul(np.matmul(np.transpose(j_eef), m_eef), (
                        kp * dpose - kd * hand_vel)[..., None])
                    return u.reshape(-1)
                
                torques = control_osc(jac_t, cur_EE_vel, mm, dpose, kp=np.array([10000, 10000, 10000]))

                # print(torques)
                        
                for i in range(dof-1):
                    # sim.resetJointState(robotId, i, jointPoses[i])
                    self.bc.setJointMotorControl2(uid, i, self.bc.TORQUE_CONTROL, force=torques[i])
            
            elif mode == 'pos':
                joint_id_offset = 7
                for i in range(len(thetas)):
                    self.bc.resetJointState(uid, joint_id_dict[f'joint{i+joint_id_offset}'], targetValue=thetas[i])

                cur_joint_pos = []
                for i in range(len(thetas)):
                    link_name = f'link{i+joint_id_offset}'
                    joint_pos = self.bc.getJointState(uid, link_id_dict[link_name])[0]
                    cur_joint_pos.append(joint_pos)
                cur_joint_pos = np.array(cur_joint_pos)
                if self.log:
                    print(cur_joint_pos)
            
            elif mode == 'sinusoidal':

                x_range = self.robot.x_range
                y_range = self.robot.y_range
                z_range = self.robot.z_range
                freq_x = self.robot.freq_x
                freq_y = self.robot.freq_y
                freq_z = self.robot.freq_z
                sin_t = self.robot.sin_t

                x_traj = (x_range[1] - x_range[0]) * np.sin(freq_x * sin_t) / 2
                y_traj = (y_range[1] - y_range[0]) * np.sin(freq_y * sin_t) / 2
                z_traj = (z_range[1] - z_range[0]) * np.sin(freq_z * sin_t) / 2

                targetPosition = init_EE_pos + np.array([x_traj, y_traj, z_traj])
                jointPoses = self.bc.calculateInverseKinematics(uid, link_id_dict[ee], targetPosition)
                for joint_id in range(dof-1):
                    # sim.resetJointState(robotId, i, jointPoses[i])
                    self.bc.setJointMotorControl2(uid, joint_id, self.bc.POSITION_CONTROL, jointPoses[joint_id])

                sin_t += 1

            time.sleep(0.01)
            
    def shutdown(self):
        self.bc.disconnect()



def main():

    start_pos = [0, 0, 0]
    start_orn = [0, 0, 0]
    arm = 'left'

    env = BulletEnvironment(start_pos,
                            start_orn,
                            RobotParams.L_JOINT_LIMIT_MIN,
                            RobotParams.L_JOINT_LIMIT_MAX,
                            robot_name=URDF_NAME,
                            arm=arm,
                            ee_name=EE_NAME,
                            debug=False,
                            log=False)

    try:
        env.run_debug_loop(arm)
    except Exception as e:
        env.shutdown()


if __name__ == "__main__":
    main()