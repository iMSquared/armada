#%% Import libraries and initialize global variable
import os, sys
from pathlib import Path
BASEDIR = Path(__file__).parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))

from numpy import typing as npt
from typing import List, Tuple, Callable, Dict
import time, datetime
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

# SysID simulation
from simulation.bullet.manipulation import ShadowingManipulation

# Experiment
from control.im2controller_client import IM2Client
from control import DefaultControllerValues as RobotParams
from experiment.utils.exp_utils import *

DEBUG = False
INITIALIZATION = False
SAVE_DIR = 'experiment/shadowing/'
DOF = RobotParams.DOF
FPS = 30

IP = '[::]'
PORT = '50051'

VIZ = False

def execute(manip: ShadowingManipulation, 
            im2client: IM2Client,
            data_dumper: Callable,
            fps: float=FPS,
            show: bool=False,
            save_dir: str=SAVE_DIR,
            **kwargs):
    
    # Logging
    current_time = datetime.datetime.now().strftime("%I%M%p_%B%d%Y")
    os.makedirs(save_dir, exist_ok=True)
    time_list = []
    des_pos_list = []
    joint_pos_list = []
    joint_vel_list = []
    joint_tau_list = []
    des_tau_list = []
    t_policy_list = []
    act_list = []

    empty_data = {'filename': 'Snatch_test',
                  'current_time': current_time,
                  'time_list': time_list,
                  'joint_pos_list': joint_pos_list,
                  'joint_vel_list': joint_vel_list,
                  'joint_tau_list': joint_tau_list,
                  'des_pos_list': des_pos_list,
                  'des_tau_list': des_tau_list,
                  't_policy_list': t_policy_list,
                  'act_list': act_list,
                  'obs_list': [],
                  'img_list': [],
                  'trajectory': [],
                  'save_dir': save_dir}
    
    
    traj, grasp = manip.extract_retargeted_trajectory()
    traj = traj[20:]
    grasp = grasp[20:]
    init = traj[0]
    init_g_state = grasp[0]

    move_to_init(manip, im2client, 'all', init, duration=2)
    im2client.set_gripper_state(RobotParams.LEFT, RobotParams.L_GRIPPER[int(init_g_state[0])])
    im2client.set_gripper_state(RobotParams.RIGHT, RobotParams.R_GRIPPER[int(init_g_state[1])])

    print('[Notice] Executing the shadowed motion')

    shadowed_data = shadow(manip, 
                           im2client,
                           traj,
                           grasp, 
                           data=empty_data, 
                           fps=fps,
                           show=show,
                           **kwargs)
        
    
    data_dumper(**shadowed_data)

    return True


def shadow(manip: ShadowingManipulation, 
           im2client: IM2Client,
           traj: npt.NDArray,
           grasp: npt.NDArray,
           data: Dict[str, Union[List, npt.NDArray, str]]=None,
           fps: float=FPS,
           show: bool=False,
           **kwargs):
    

    if len(traj) == 0:
        return False

    kps = RobotParams.SHADOW_P_GAIN
    kds = RobotParams.SHADOW_D_GAIN

    if show:
        manip.simulate_traj('all', traj, draw=True)

    try:
        pos_L, _, _, _ = im2client.get_current_states(RobotParams.LEFT)
        pos_R, _, _, _ = im2client.get_current_states(RobotParams.RIGHT)
        prev_joint_pos = np.concatenate((pos_L, pos_R))
        prev_g = grasp[0, :]

        for i, (q, g) in enumerate(zip(traj, grasp)):
            # Log time
            st = time.time()

            # Reconstruct action
            delta_joint_pos = q - prev_joint_pos
            act = np.concatenate((delta_joint_pos, kps, kds))
            
            # predict velocity
            target_joint_vel = delta_joint_pos*fps
            main_q = q[:RobotParams.DOF]
            sub_q = q[RobotParams.DOF:]
            main_v = target_joint_vel[:RobotParams.DOF]
            sub_v = target_joint_vel[RobotParams.DOF:]
            pos = []
            vel = []
            tau = []
            grav = []
            for single_q, single_v, single_arm in zip([main_q, sub_q],[main_v, sub_v],[RobotParams.LEFT, RobotParams.RIGHT]):
                im2client.position_control(single_arm, 
                                            single_q, 
                                            desired_p_gains=RobotParams.SHADOW_P_GAIN,
                                            desired_d_gains=RobotParams.SHADOW_D_GAIN,
                                            desired_velocity=single_v)
                single_pos, single_vel, single_tau, single_grav = im2client.get_current_states(single_arm)

                # Gripper
                if single_arm == RobotParams.LEFT:
                    if prev_g[0] != g[0]:
                        im2client.set_gripper_state(single_arm, RobotParams.L_GRIPPER[g[0]])

                if single_arm == RobotParams.RIGHT:
                    if prev_g[1] != g[1]:
                        im2client.set_gripper_state(single_arm, RobotParams.R_GRIPPER[g[1]])

                pos.append(single_pos)
                vel.append(single_vel)
                tau.append(single_tau)
                grav.append(single_grav)
            pos = np.concatenate(pos)
            vel = np.concatenate(vel)
            tau = np.concatenate(tau)
            grav = np.concatenate(grav)

            # Logging
            prev_joint_pos = q
            prev_g = g
            dur = time.time() - st

            # Logging
            if data is not None:
                data['time_list'].append(st)
                data['act_list'].append(act)
                data['des_pos_list'].append(q)
                data['joint_pos_list'].append(pos)
                data['joint_vel_list'].append(vel)
                data['joint_tau_list'].append(tau)
                data['des_tau_list'].append(tau-grav)
                data['t_policy_list'].append(dur)

            if dur < 1/fps:
                time.sleep(1/fps - dur)


    except Exception as e:
        print(f'[ERROR | Execute] {e}')
        pass

    return data


def shadow_no_robot(manip: ShadowingManipulation, 
                    fps: float=FPS,
                    **kwargs):
    
    traj, grasp = manip.extract_retargeted_trajectory()

    for q in tqdm(traj):
        st = time.time()

        manip.robot.setJointStates('all', q)


        dur = time.time() - st
        if dur < 1/fps:
            time.sleep(1/fps - dur)


#%% Driver function
def main():    
    joint_pos_lower_limit = np.array([-90.,  -90., -180.,  -45., -210., -125.])*math.pi/180.
    joint_pos_upper_limit = np.array([50.,  80., 80.,  75., 210. , 125.])*math.pi/180.
    joint_vel_upper_limit = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.1750, 2.1750])

    # arm_data_dir = '/home/im2/Project/WHAM/output/demo/wholebody_hand_zero/wham_output.pkl'
    # left_hand_data_dir = '/home/im2/Project/hamer/humanoid_out/wholebody_hand_zero'
    # right_hand_data_dir = '/home/im2/Project/hamer/humanoid_out/wholebody_hand_zero'
    # arm_data_dir = '/home/im2/Project/WHAM/output/demo/left_wholebody_zero/wham_output.pkl'
    # left_hand_data_dir = '/home/im2/Project/hamer/humanoid_out/left_wholebody_zero_left'
    # right_hand_data_dir = '/home/im2/Project/hamer/humanoid_out/left_wholebody_zero_right'
    # arm_data_dir = '/home/im2/Project/WHAM/output/demo/left_hand_wholebody/wham_output.pkl'
    # left_hand_data_dir = '/home/im2/Project/hamer/humanoid_out/left_hand_wholebody_left'
    # right_hand_data_dir = '/home/im2/Project/hamer/humanoid_out/left_hand_wholebody_right'
    # arm_data_dir = '/home/im2/Project/WHAM/output/demo/hand_down/wham_output.pkl'
    # left_hand_data_dir = '/home/im2/Project/hamer/humanoid_out/hand_down'
    # right_hand_data_dir = '/home/im2/Project/hamer/humanoid_out/hand_down'
    arm_data_dir = '/home/im2/Project/WHAM/output/demo/hand_over_3/wham_output.pkl'
    left_hand_data_dir = '/home/im2/Project/hamer/humanoid_out/hand_over_3_left'
    right_hand_data_dir = '/home/im2/Project/hamer/humanoid_out/hand_over_3_right'
    urdf_dir = 'simulation/assets/urdf/RobotBimanualV5/urdf/Simplify_Robot_plate_gripper.urdf'
    
    start_pos = [0, 0, 0]
    start_orn = [0, 0, 0]
    
    manip = ShadowingManipulation(start_pos, 
                                  start_orn,
                                  robot_name=urdf_dir, 
                                  joint_min=joint_pos_lower_limit, 
                                  joint_max=joint_pos_upper_limit, 
                                  joint_vel_upper_limit=joint_vel_upper_limit,
                                  debug=True)



    

    manip.load_arm(arm_data_dir)
    manip.load_hand(left_hand_data_dir, right_hand_data_dir)

    shadow_no_robot(manip)
    
    # im2client = IM2Client(ip=IP, port=PORT)  

    # exec_params = {'manip': manip,
    #                'im2client': im2client,
    #                'show': False,
    #                'data_dumper': dump_data}
        
    # execute(**exec_params)

    manip.close()


if __name__ == '__main__':
    main()

