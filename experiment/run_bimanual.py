#%% Import libraries and initialize global variable
import os, sys
from pathlib import Path
BASEDIR = Path(__file__).parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))
from glob import glob

from numpy import typing as npt
import pickle
from typing import List, Tuple, Callable
import time, datetime
import numpy as np
import torch
import argparse

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

# SysID simulation
from simulation.bullet.manipulation import Bimanipulation, URDF_NAME

# Experiment
from control.im2controller_client import IM2Client
from control import DefaultControllerValues as RobotParams
from experiment.utils.exp_utils import *

DEBUG = False
INITIALIZATION = False
SAVE_DIR = 'experiment/bimanual/'
DOF = 6
FPS = 200
RESOLUTION = 0.001

IP = '[::]'
PORT = '50051'

VIZ = False
EE =  True
LEFT = RobotParams.LEFT
RIGHT = RobotParams.RIGHT

EE_OPEN = {LEFT: 76, RIGHT: 142}
EE_CLOSE = {LEFT: 126, RIGHT: 192}

def execute(manip: Bimanipulation, 
            im2client: IM2Client,
            arm: str,
            init: npt.NDArray=None,
            term: npt.NDArray=None,
            data_dumper: Callable=None,
            fps: float=FPS,
            show: bool=False,
            save_dir: str=SAVE_DIR,
            gripper_action: str=None,
            **kwargs):
    
    # Logging
    time_list = []
    des_pos_list = []
    joint_pos_list = []
    joint_vel_list = []
    joint_tau_list = []
    des_tau_list = []
    t_policy_list = []
    act_list = []
    
    if init is None:
        init, _, _, _ = im2client.get_current_states(arm)
        # manip.robot.setArmJointStates(arm, pos)

    if term is None:
        term = init.copy()

    if arm in [LEFT, RIGHT]:
        traj = manip.motion_plan(arm, init, term)
    else:
        traj = np.array(manip.synced_motion_plan(init, term))

    if len(traj) == 0:
        return False

    if show:
        manip.simulate_traj(arm, traj, draw=True)

    try:
        if arm in [LEFT, RIGHT]:
            prev_joint_pos, cur_joint_vel, _, _ = im2client.get_current_states(arm)
        else:
            prev_l_ee_pos, _, _, _ = im2client.get_current_states(manip.main_arm)
            prev_r_ee_pos, _, _, _ = im2client.get_current_states(manip.sub_arm)
            prev_joint_pos = np.concatenate((prev_l_ee_pos, prev_r_ee_pos))
        
        for i, q in enumerate(traj):
            # Log time
            st = time.time()
            time_list.append(st)

            # Reconstruct action
            delta_joint_pos = q - prev_joint_pos
            act = np.concatenate((delta_joint_pos, RobotParams.POS_P_GAIN, RobotParams.POS_D_GAIN))
            act_list.append(act)

            # predict velocity
            target_joint_vel = delta_joint_pos*fps
            if arm in [LEFT, RIGHT]:
                im2client.position_control(arm, q, desired_velocity=target_joint_vel)
                pos, vel, tau, grav = im2client.get_current_states(arm)
            else:
                main_q = q[:RobotParams.DOF]
                sub_q = q[RobotParams.DOF:]
                main_v = target_joint_vel[:RobotParams.DOF]
                sub_v = target_joint_vel[RobotParams.DOF:]
                pos = []
                vel = []
                tau = []
                grav = []
                for single_q, single_v, single_arm in zip([main_q, sub_q],[main_v, sub_v],[manip.main_arm, manip.sub_arm]):
                    im2client.position_control(single_arm, single_q, desired_velocity=single_v)
                    single_pos, single_vel, single_tau, single_grav = im2client.get_current_states(single_arm)
                    pos.append(single_pos)
                    vel.append(single_vel)
                    tau.append(single_tau)
                    grav.append(single_grav)
                pos = np.concatenate(pos)
                vel = np.concatenate(vel)
                tau = np.concatenate(tau)
                grav = np.concatenate(grav)


            # Logging
            des_pos_list.append(q)
            joint_pos_list.append(pos)
            joint_vel_list.append(vel)
            joint_tau_list.append(tau)
            des_tau_list.append(tau - grav)
            prev_joint_pos = q

            dur = time.time() - st
            t_policy_list.append(dur)

            if dur < 1/fps:
                time.sleep(1/fps - dur)


        if arm in [LEFT, RIGHT]:
            if gripper_action == 'open':
                im2client.set_gripper_state(arm, EE_OPEN[arm])
            elif gripper_action == 'close':
                im2client.set_gripper_state(arm, EE_CLOSE[arm])
        else:
            for single_arm in [LEFT, RIGHT]:
                if gripper_action == 'open':
                    im2client.set_gripper_state(arm, EE_OPEN[single_arm])
                elif gripper_action == 'close':
                    im2client.set_gripper_state(arm, EE_CLOSE[single_arm])


        if kwargs.get('post_record', False):
            for i in range(int(fps*0.5)):
                # Log time
                st = time.time()
                time_list.append(st)

                # Reconstruct action
                delta_joint_pos = traj[-1] - prev_joint_pos
                act = np.concatenate((delta_joint_pos, RobotParams.POS_P_GAIN, RobotParams.POS_D_GAIN))
                act_list.append(act)

                # predict velocity
                pos, vel, tau, grav = im2client.get_current_states(arm)

                # Logging
                joint_pos_list.append(pos)
                joint_vel_list.append(vel)
                joint_tau_list.append(tau)

                dur = time.time() - st
                t_policy_list.append(dur)

                if dur < 1/fps:
                    time.sleep(1/fps - dur)

    except Exception as e:
        print(f'[ERROR | Execute] {e}')
        pass


    current_time = datetime.datetime.now().strftime("%I%M%p_%B%d%Y")
    os.makedirs(save_dir, exist_ok=True)

    params = {'filename': f'Bimanual_{arm}',
              'arm': arm,
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
              'trajectory': np.array(traj),
              'save_dir': save_dir}
    
    data_dumper(**params)

    return True


#%% Driver function
def main():    
    
    start_pos = [0, 0, 0]
    start_orn = [0, 0, 0]

    joint_pos_lower_limit = np.concatenate((RobotParams.L_JOINT_LIMIT_MIN, RobotParams.GRIPPER_SIM_MIN, RobotParams.R_JOINT_LIMIT_MIN, RobotParams.GRIPPER_SIM_MIN))
    joint_pos_upper_limit = np.concatenate((RobotParams.L_JOINT_LIMIT_MAX, RobotParams.GRIPPER_SIM_MAX, RobotParams.R_JOINT_LIMIT_MAX, RobotParams.GRIPPER_SIM_MAX))
    joint_vel_limit = RobotParams.JOINT_VEL_UPPER_LIMIT
    
    manip = Bimanipulation(URDF_NAME, 
                         start_pos, 
                         start_orn, 
                         main_arm=LEFT, 
                         control_freq=FPS,
                         resolution=RESOLUTION, 
                         joint_min=joint_pos_lower_limit, 
                         joint_max=joint_pos_upper_limit, 
                         joint_vel_upper_limit=joint_vel_limit,
                         jonit_id_offset=RobotParams.DOF_ID_OFFSET,
                         debug=False)

    im2client = IM2Client(ip=IP, port=PORT)
    if EE:
        for arm in [LEFT, RIGHT]:
            im2client.set_gripper_state(arm, EE_OPEN[arm])
    
    ## 0: test trajectory
    # left_init = np.array([-0.7, -0.0422,  0.1596, -0.0566,  -0.3,  -0.8])
    # left_term = np.array([-0.6, -0.0422,  0.1596, -0.0566,  -0.3,  -0.7])
    
    ## 1: planar grasp 
    left_init = np.array([ 0.2321, -0.1574, -0.1123,  0.0566, -0.0873, -1.6406])
    left_term = np.array([-0.9924, -0.3443, -0.1101,  0.0093, -0.2793, -0.8203])

    ## 2: twisted grasp
    # left_init = np.array([-0.8501, -0.5488,  0.9676, -0.3321, -1.7453, -1.0123])
    # left_term = np.array([-0.728 , -0.3416,  0.2146, -0.0273, -1.2217, -1.1345])
    # manip.set_hand_offset_orientation((0, -90, 0))
    # manip.set_hand_offset_position(0.1)

    ## 3: 90deg grasp
    # left_init = np.array([-0.0471,  0.0772, -0.2722,  0.268 , -0.2618, -0.384 ])
    # left_term = np.array([ 0.3096, -0.2245,  0.0971,  0.025 , -0.3665, -0.5061])
    # manip.set_hand_offset_orientation((0, 0, -90))
    # manip.set_hand_offset_position((0.07, 0.07, 0))

    # Compute subordinate initialization pose
    right_init = manip.get_sub_joint_pose(left_init)

    # Move to grasping mode
    l_0, _, _, _ = im2client.get_current_states(LEFT)
    r_0, _, _, _ = im2client.get_current_states(RIGHT)

    for q_0, q_init, arm in zip([l_0, r_0], [left_init, right_init], [LEFT, RIGHT]):
        exec_params = {'manip': manip,
                       'im2client': im2client,
                       'init': q_0,
                       'term': q_init,
                       'arm': arm,
                       'show': False,
                       'data_dumper': dump_data,
                       'gripper_action': 'open' if EE else None,
                       'post_record': False}
        execute(**exec_params)

    # Bimanual motion planning
    exec_params = {'manip': manip,
                   'im2client': im2client,
                   'init': left_init,
                   'term': left_term,
                   'arm': 'all',
                   'show': False,
                   'data_dumper': dump_data,
                   'gripper_action': 'open' if EE else None,
                   'post_record': False}
    execute(**exec_params)

    manip.close()


if __name__ == '__main__':
    main()

