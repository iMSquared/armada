import sys
from pathlib import Path
BASEDIR = Path(__file__).parent.parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.append(str(BASEDIR))

from typing import Tuple, List, Dict
import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from tqdm import tqdm
from math import ceil

from simulation.bullet.bullet_sim import BulletEnvironment
from simulation.utils.PinocchioInterface import PinocchioInterface, getJacobian
from control import DefaultControllerValues as RobotParams

LEFT = RobotParams.LEFT
RIGHT = RobotParams.RIGHT


URDF_NAME = 'simulation/assets/urdf/RobotBimanualV5/urdf/Simplify_Robot_plate_gripper.urdf'        # 'LinkageURDF3'
DATA_DIR = 'experiment/bimanual_snatch/Bimanual_all_0933AM_November222024.pkl'
CONTROLLER = 'position'
JOINT_POS_LOWER_LIMIT = np.array([-90.,  -50., -180.,  -45., -110., -125.])*np.pi/180
JOINT_POS_UPPER_LIMIT = np.array([50.,  50., 50.,  75., 210. , 125.])*np.pi/180
JOINT_VEL_UPPER_LIMIT = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.1750, 2.1750])
JOINT_ID_OFFSET = 7
TASKSPACE = ((0.1, -0.4, 0.509), (0.5, 0.05, 0.7))
PLAYSPEED = 10
CONTROL_FPS = 200
POLICY_FPS = 200

class Replayer(BulletEnvironment):
    def __init__(self, 
                 start_pos: npt.ArrayLike, 
                 start_orn: npt.ArrayLike,
                 data_dir: str,
                 robot_name: str,  
                 arm: str=LEFT,
                 controller_type: str=CONTROLLER,
                 joint_pos_lower_limit: npt.NDArray=JOINT_POS_LOWER_LIMIT,
                 joint_pos_upper_limit: npt.NDArray=JOINT_POS_UPPER_LIMIT,
                 joint_vel_upper_limit: npt.NDArray=JOINT_VEL_UPPER_LIMIT,
                 jonit_id_offset: int=JOINT_ID_OFFSET,
                 speed: float=PLAYSPEED,
                 debug: bool = False, 
                 log: bool = False, **kwargs):
        super().__init__(start_pos, 
                         start_orn, 
                         joint_pos_lower_limit, 
                         joint_pos_upper_limit, 
                         robot_name=robot_name, 
                         arm=arm, 
                         scene_on=True, 
                         debug=debug, 
                         log=log, 
                         gravity=True)
        self.bc.resetDebugVisualizerCamera(cameraDistance=1.2,
                                           cameraYaw=51.2,
                                           cameraPitch=-39.4,
                                           cameraTargetPosition=(-0.018, -0.0214, 0.44))

        # Controller and robot configs
        self.controller = controller_type
        self.joint_pos_lower_limit = joint_pos_lower_limit
        self.joint_pos_upper_limit = joint_pos_upper_limit
        self.joint_vel_upper_limit = joint_vel_upper_limit
        self.joint_id_offset = jonit_id_offset

        # Play configs
        self.speed = speed

        with open(data_dir, 'rb') as f:
            self.real_data: Dict[str, List[npt.NDArray]] = pickle.load(f)


    def extract_policy_trajectory(self) -> List[npt.NDArray]:
        if 'act' in self.real_data:
            action_history = self.real_data['act']    # shape: (#timestep, 18)
            if isinstance(self.real_data['act'], list):
                action_history = np.stack(self.real_data['act'])
            if len(action_history.shape) > 2:
                action_history = action_history.squeeze()
        elif 'action' in self.real_data:
            action_history = self.real_data['action']
            if isinstance(self.real_data['action'], list):
                action_history = np.stack(self.real_data['action'])
            if len(action_history.shape) > 2:
                action_history = action_history.squeeze()
        else:
            raise KeyError('act or action must be in the real data')
        joint_traj = self.real_data['joint_pos']
        if isinstance(joint_traj, list):
            joint_traj = np.stack(joint_traj)
        if len(joint_traj.shape) > 2:
            joint_traj = joint_traj.squeeze()

        
        cur_joint_pos = joint_traj[0]

        step = ceil(len(joint_traj)/len(action_history))

        traj = []

        # FYI: actions are already transformed during execution
        for i, action in enumerate(action_history):
            if self.controller == "JP":
                raise NotImplementedError
            
            elif self.controller == "position":
                delta_joint_pos = action[:6]
                target_joint_pos = cur_joint_pos + delta_joint_pos

                dof_lower_safety_limit = (0.95*self.joint_pos_lower_limit + 0.05*self.joint_pos_upper_limit)
                dof_upper_safety_limit = (0.05*self.joint_pos_lower_limit + 0.95*self.joint_pos_upper_limit)

                # Input to the client
                target_joint_pos = np.clip(target_joint_pos, dof_lower_safety_limit, dof_upper_safety_limit)
                p_gains = action[6:12]
                d_gains = action[12:]

                traj.append(target_joint_pos)
                if step*(i+1) < len(joint_traj):
                    cur_joint_pos = joint_traj[step*(i+1)]
                else:
                    break

        return np.stack(traj), step
    

    def extract_real_trajectory(self) -> List[npt.NDArray]:
        if len(self.real_data['joint_pos']) > 1:
            self.real_data['joint_pos'] = np.stack(self.real_data['joint_pos']).squeeze()
        return self.real_data['joint_pos']
    
    
    def extract_sysID_trajectory(self) -> Tuple[npt.NDArray, npt.NDArray]:
        if len(self.real_data['joint_pos']) > 1:
            self.real_data['joint_pos'] = self.real_data['joint_pos'].squeeze()
        return self.real_data['joint_pos'], self.real_data['des_pos']

    def extract_hand_trajectory(self) -> Tuple[npt.NDArray, npt.NDArray]:
        assert 'obs' in self.real_data, 'observation data is required'
        if isinstance(self.real_data['obs'], list):
            self.real_data['obs'] = np.stack(self.real_data['obs'])

        if len(self.real_data['obs'].shape) > 2:
            self.real_data['obs'] = self.real_data['obs'].squeeze()

        hand_traj = np.stack([o[44:51] for o in self.real_data['obs']])

        return hand_traj
    
    def extract_mp_trajectory(self) -> Tuple[npt.NDArray, npt.NDArray]:
        assert 'trajectory' in self.real_data, 'No motion plan trajectory in the pickle file'
        if len(self.real_data['trajectory']) > 1:
            self.real_data['trajectory'] = self.real_data['trajectory'].squeeze()
        return self.real_data['trajectory']

    def extract_bi_trajectory(self) -> Tuple[npt.NDArray, npt.NDArray]:
        assert 'joint_pos' in self.real_data, 'No motion plan trajectory in the pickle file'
        if len(self.real_data['joint_pos']) > 1:
            self.real_data['joint_pos'] = self.real_data['joint_pos'].squeeze()
        return self.real_data['joint_pos']


    def will_play(self):

        valid = False
        while not valid:
            print('Do you want to replay? [y/n]')
            key = input()

            if len(key) > 1:
                key = key[0]

            if key.lower() in ['y', 'n']:
                valid = True
                return key == 'y'
    

    def replayer_prompt(self):

        valid = False
        while not valid:
            print('trajectory type? [real, act, sysid, hand, mp, bi]')
            key = input()

            if key.lower() in ['real', 'act', 'sysid', 'hand', 'mp', 'bi']:
                valid = True

        return key.lower()
    

    def replay_loop(self, arm: str, draw:bool=True, animate_sysID: bool=False):
        
        pallete = {'real': [0,0,1], 'act': [0,1,0], 'sysid': [0,0,1], 'hand': [0,1,1], 'mp': [1,0,0], 'bi': ([1,0,0], [0,0,1])}

        while True:
            traj_type = self.replayer_prompt()

            if traj_type in ['real']:
                joint_traj = self.extract_real_trajectory()
                t = 1/CONTROL_FPS
            elif traj_type in ['policy', 'act']:
                joint_traj, _ = self.extract_policy_trajectory()
                t = 1/POLICY_FPS
            elif traj_type in ['sysID']:
                joint_traj, predefined_traj = self.extract_sysID_trajectory()
                t = 1/POLICY_FPS
            elif traj_type in ['hand']:
                joint_traj = self.extract_hand_trajectory()
                t = 1/POLICY_FPS
            elif traj_type in ['mp']:
                joint_traj = self.extract_mp_trajectory()
                t = 1/POLICY_FPS
            elif traj_type in ['bi']:
                joint_traj = self.extract_bi_trajectory()
                t = 1/POLICY_FPS
            else:
                raise ValueError('Available options for traj_type: [real, policy, act]')

            
            if traj_type in ['sysID']:
                self.robot.setArmJointStates(arm, self.real_data['init_pos'])
                prev = self.robot.get_ee_pose(arm)    
                for q in tqdm(predefined_traj):
                    st = time.time()
                    self.robot.setArmJointStates(arm, q)
                    cur = self.robot.get_ee_pose(arm)
                    if draw:
                        self.bc.addUserDebugLine(prev[0], cur[0], lineColorRGB=[1,1,0], lineWidth=2)
                    prev = cur
                    if animate_sysID:
                        duration = time.time() - st
                        if duration < (t / self.speed):
                            time.sleep((t / self.speed) - duration)

            self.robot.setArmJointStates(arm, self.real_data['joint_pos'][0])
            prev = self.robot.get_ee_pose(arm)    
            
            if traj_type not in ['hand', 'bi']:
                for i, q in enumerate(tqdm(joint_traj)):
                    st = time.time()
                    self.robot.setArmJointStates(arm, q)
                    cur = self.robot.get_ee_pose(arm)
                    if draw:
                        self.bc.addUserDebugLine(prev[0], cur[0], lineColorRGB=pallete[traj_type], lineWidth=2)
                    prev = cur
                    duration = time.time() - st
                    if duration < (t / self.speed):
                        time.sleep((t / self.speed) - duration)
            
            if traj_type in ['bi']:
                for i, q in enumerate(tqdm(joint_traj)):
                    st = time.time()
                    self.robot.setArmJointStates(arm, q)
                    cur = self.robot.get_ee_pose(arm)
                    if draw:
                        for single_prev, single_cur, color in zip(prev, cur, pallete[traj_type]):
                            self.bc.addUserDebugLine(single_prev[0], single_cur[0], lineColorRGB=color, lineWidth=2)
                    prev = cur
                    duration = time.time() - st
                    if duration < (t / self.speed):
                        time.sleep((t / self.speed) - duration)

            
            else:
                prev = joint_traj[0][:3]
                for i, ee_pose in enumerate(tqdm(joint_traj)):
                    st = time.time()
                    cur = ee_pose[:3]
                    self.bc.addUserDebugLine(prev, cur, lineColorRGB=[0,1,1], lineWidth=2)
                    prev = cur
                    duration = time.time() - st
                    if duration < (t / self.speed):
                        time.sleep((t / self.speed) - duration)


    def graph(self):
        real_traj = self.extract_real_trajectory()
        policy_traj, step = self.extract_policy_trajectory()
        real_timestep = np.array(range(len(real_traj)))*0.01
        policy_timestep = real_timestep.copy()[::step]


        plt.figure(figsize=(20, 10))
        for i in range(self.robot.dof):
            # Real Trajectory
            plt.subplot(2, 3, i+1)
            plt.plot(real_timestep, real_traj[:,i], label=f'real {i+1}', color='r')
            plt.plot(policy_timestep, policy_traj[:,i], label=f'policy {i+1}', color='b')
            # plt.axhline(y=np.max(real_traj[:,i]), color='g', linestyle='--')
            # plt.axhline(y=np.min(real_traj[:,i]), color='g', linestyle='--')
            plt.legend()
            plt.title(f'Joint {i+1} position')
            plt.xlabel('time')
            plt.ylabel('pos')
            plt.grid(True)
        plt.show()

    
    def computeEEvel(self):
        left_urdf = 'simulation/assets/urdf/RobotBimanualV5/urdf/Simplify_Robot_plate_gripper_LEFT.urdf'
        right_urdf = 'simulation/assets/urdf/RobotBimanualV5/urdf/Simplify_Robot_plate_gripper_RIGHT.urdf'
        l_pino = PinocchioInterface(left_urdf, LEFT, RobotParams.DOF)
        r_pino = PinocchioInterface(right_urdf, RIGHT, RobotParams.DOF)
        l_pino.SetJacobian()
        r_pino.SetJacobian()

        l_theta = replayer.real_data['joint_pos'][-1, :RobotParams.DOF]
        r_theta = replayer.real_data['joint_pos'][-1, -RobotParams.DOF:]

        l_theta_dot = replayer.real_data['joint_vel'][-1, :RobotParams.DOF]
        r_theta_dot = replayer.real_data['joint_vel'][-1, -RobotParams.DOF:]

        l_J, _ = getJacobian(l_pino, l_theta, l_theta_dot)
        r_J, _ = getJacobian(r_pino, r_theta, r_theta_dot)

        v_l = l_J@l_theta_dot
        v_r = r_J@r_theta_dot

        print(f'EE vel from Jacobian: {v_l, v_r}')

        # Proof
        self.robot.setArmJointStates('all', self.real_data['joint_pos'][-2])
        l_ee_T_1, _ = self.robot.get_ee_pose(LEFT)
        r_ee_T_1, _ = self.robot.get_ee_pose(RIGHT)
        self.robot.setArmJointStates('all', self.real_data['joint_pos'][-1])
        l_ee_T, _ = self.robot.get_ee_pose(LEFT)
        r_ee_T, _ = self.robot.get_ee_pose(RIGHT)

        v_l_proof = (np.array(l_ee_T) - np.array(l_ee_T_1))*POLICY_FPS
        v_r_proof = (np.array(r_ee_T) - np.array(r_ee_T_1))*POLICY_FPS

        print(f'EE vel from sim: {v_l_proof, v_r_proof}')


        return v_l, v_r



if __name__ == '__main__':

    start_pos = [0, 0, 0]
    start_orn = [0, 0, 0]
    arm = 'all'

    if arm == 'all':
        joint_pos_lower_limit = np.concatenate((RobotParams.L_JOINT_LIMIT_MIN, RobotParams.GRIPPER_SIM_MIN, RobotParams.R_JOINT_LIMIT_MIN, RobotParams.GRIPPER_SIM_MIN))
        joint_pos_upper_limit = np.concatenate((RobotParams.L_JOINT_LIMIT_MAX, RobotParams.GRIPPER_SIM_MAX, RobotParams.R_JOINT_LIMIT_MAX, RobotParams.GRIPPER_SIM_MAX))
        joint_vel_upper_limit = RobotParams.JOINT_VEL_UPPER_LIMIT
    elif arm == LEFT:
        joint_pos_lower_limit = RobotParams.L_JOINT_LIMIT_MIN
        joint_pos_upper_limit = RobotParams.L_JOINT_LIMIT_MAX
        joint_vel_upper_limit = RobotParams.JOINT_VEL_UPPER_LIMIT
    elif arm == RIGHT:
        joint_pos_lower_limit = RobotParams.R_JOINT_LIMIT_MIN
        joint_pos_upper_limit = RobotParams.R_JOINT_LIMIT_MAX
        joint_vel_upper_limit = RobotParams.JOINT_VEL_UPPER_LIMIT

    replayer = Replayer(start_pos, 
                        start_orn, 
                        DATA_DIR,
                        robot_name=URDF_NAME, 
                        arm=arm,
                        joint_pos_lower_limit=joint_pos_lower_limit,
                        joint_pos_upper_limit=joint_pos_upper_limit,
                        joint_vel_upper_limit=joint_vel_upper_limit,
                        speed=PLAYSPEED, 
                        debug=False,
                        log=False)

    # replayer.graph()
    v_l, v_r = replayer.computeEEvel()
    v_l, v_r = np.linalg.norm(v_l), np.linalg.norm(v_r)
    m_l, m_r = 0.1214, 0.1093

    ke_l, ke_r = m_l*v_l*v_l, m_r*v_r*v_r
    
    
    try:
        replayer.replay_loop(arm)
    except Exception as e:
        replayer.shutdown()