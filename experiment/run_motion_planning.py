#%% Import libraries and initialize global variable
import os, sys
from pathlib import Path
BASEDIR = Path(__file__).parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))
from glob import glob

from numpy import typing as npt
from typing import List, Tuple, Callable, Dict
import time, datetime
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

# SysID simulation
from simulation.bullet.manipulation import Manipulation

# Experiment
from control.im2controller_client import IM2Client
from control import DefaultControllerValues as RobotParams
from experiment.utils.exp_utils import *
import matplotlib.pyplot as plt

DEBUG = False
INITIALIZATION = False
SAVE_DIR = 'experiment/motion_planning/'
DOF = RobotParams.DOF
FPS = 200
SNATCH_RESOLUTION = 0.015
MOVE_RESOLUTION = 0.005

IP = '[::]'
PORT = '50051'

VIZ = False
EE =  True
EE_OPEN, EE_CLOSE = RobotParams.L_GRIPPER
# EE_OPEN -= 10

def execute(manip: Manipulation, 
            im2client: IM2Client,
            arm: str,
            pick: npt.NDArray,
            release: npt.NDArray,
            data_dumper: Callable,
            fps: float=FPS,
            show: bool=False,
            save_dir: str=SAVE_DIR,
            gripper_action: str=None,
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
    
    
    snatched_data = snatch(manip, 
                           im2client, 
                           arm,
                           pick,
                           release,
                           data=empty_data, 
                           fps=fps,
                           show=show,
                           gripper_action=gripper_action,
                           **kwargs)
        
    # finished_data = slow_move(manip,
    #                           im2client,
    #                           arm,
    #                           release,
    #                           snatched_data,
    #                           fps,
    #                           show,
    #                           gripper_action='open',
    #                           **kwargs)
    
    # data_dumper(**finished_data)

    return True


def snatch(manip: Manipulation, 
           im2client: IM2Client,
           arm: str,
           pick,
           release,
           data: Dict[str, Union[List, npt.NDArray, str]]=None,
           fps: float=FPS,
           show: bool=False,
           **kwargs):
    
    manip.resolution = SNATCH_RESOLUTION
    init, _, _, _ = im2client.get_current_states(arm)
    pick_traj = manip.motion_plan(arm, init, pick)
    move_traj = manip.motion_plan(arm, pick, release)

    traj = np.concatenate((pick_traj, move_traj), axis=0)
    
    if len(traj) == 0:
        return False
    
    start_grasp = len(pick_traj) - 1 - kwargs.get('grasp_timing', 0)*fps
    grasped = False

    kps = RobotParams.SNATCH_P_GAIN
    kds = RobotParams.SNATCH_D_GAIN

    if show: 
        manip.simulate_traj(arm, traj, draw=True)

    try:
        prev_joint_pos, cur_joint_vel, _, _ = im2client.get_current_states(arm)
        for i, q in enumerate(traj):
            # Log time
            st = time.time()

            # Reconstruct action
            delta_joint_pos = q - prev_joint_pos
            act = np.concatenate((delta_joint_pos, kps, kds))
            
            # predict velocity
            target_joint_vel = delta_joint_pos*fps
            im2client.position_control(arm, q, 
                                       desired_velocity=target_joint_vel,
                                       desired_p_gains=kps,
                                       desired_d_gains=kds)
            pos, vel, tau, grav = im2client.get_current_states(arm)

            # Logging
            
            prev_joint_pos = q

            # Pre-start gripper
            if not grasped and i > start_grasp:
                im2client.set_gripper_state(arm, EE_CLOSE)
                grasped = True

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

        if grasped:
            im2client.set_gripper_state(arm, EE_OPEN)
            grasped = False

        # Slow down
        for i in range(int(fps*0.1)):
            # Log time
            st = time.time()

            # read pos
            pos, vel, tau, grav = im2client.get_current_states(arm)
            im2client.position_control(arm, pos, 
                                       desired_p_gains=np.zeros(RobotParams.DOF),
                                       desired_d_gains=RobotParams.SNATCH_D_GAIN*2)

            # Reconstruct action
            delta_joint_pos = pos - prev_joint_pos
            act = np.concatenate((delta_joint_pos, kps, kds))
            
            # Logging
            prev_joint_pos = pos

            dur = time.time() - st

            if data is not None:
                data['time_list'].append(st)
                data['act_list'].append(act)
                data['joint_pos_list'].append(pos)
                data['joint_vel_list'].append(vel)
                data['joint_tau_list'].append(tau)
                data['t_policy_list'].append(dur)

            if dur < 1/fps:
                time.sleep(1/fps - dur)

        if kwargs.get('post_record', False):
            for i in range(int(fps*0.5)):
                # Log time
                st = time.time()

                # Reconstruct action
                delta_joint_pos = traj[-1] - prev_joint_pos
                act = np.concatenate((delta_joint_pos, kps, kds))

                # predict velocity
                pos, vel, tau, grav = im2client.get_current_states(arm)


                dur = time.time() - st

                # Logging
                if data is not None:
                    data['time_list'].append(st)
                    data['act_list'].append(act)
                    data['joint_pos_list'].append(pos)
                    data['joint_vel_list'].append(vel)
                    data['joint_tau_list'].append(tau)
                    data['t_policy_list'].append(dur)

                if dur < 1/fps:
                    time.sleep(1/fps - dur)

    except Exception as e:
        print(f'[ERROR | Execute] {e}')
        pass

    return data


def slow_move(manip: Manipulation, 
              im2client: IM2Client,
              arm: str,
              term,
              data: Dict[str, Union[List, npt.NDArray, str]]=None,
              fps: float=FPS,
              show: bool=False,
              gripper_action: str=None,
              **kwargs):
    
    manip.resolution = MOVE_RESOLUTION

    init, _, _, _ = im2client.get_current_states(arm)
    traj = manip.motion_plan(arm, init, term)
    
    if len(traj) == 0:
        return False
    
    kps = RobotParams.SNATCH_P_GAIN*2
    kds = RobotParams.SNATCH_D_GAIN*2

    if show:
        manip.simulate_traj(arm, traj, draw=True)

    try:
        prev_joint_pos, cur_joint_vel, _, _ = im2client.get_current_states(arm)
        for i, q in enumerate(traj):
            # Log time
            st = time.time()

            # Reconstruct action
            delta_joint_pos = q - prev_joint_pos
            act = np.concatenate((delta_joint_pos, kps, kds))
            
            # predict velocity
            target_joint_vel = delta_joint_pos*fps
            im2client.position_control(arm, q, 
                                       desired_velocity=target_joint_vel,
                                       desired_p_gains=kps,
                                       desired_d_gains=kds)
            pos, vel, tau, grav = im2client.get_current_states(arm)
            
            prev_joint_pos = q

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

        # Slow down
        for i in range(int(fps*0.1)):
            # Log time
            st = time.time()

            # read pos
            pos, vel, tau, grav = im2client.get_current_states(arm)
            im2client.position_control(arm, pos, 
                                       desired_p_gains=np.zeros(RobotParams.DOF),
                                       desired_d_gains=RobotParams.SNATCH_D_GAIN*2)

            # Reconstruct action
            delta_joint_pos = pos - prev_joint_pos
            act = np.concatenate((delta_joint_pos, kps, kds))
            
            # Logging
            prev_joint_pos = pos

            dur = time.time() - st

            if data is not None:
                data['time_list'].append(st)
                data['act_list'].append(act)
                data['joint_pos_list'].append(pos)
                data['joint_vel_list'].append(vel)
                data['joint_tau_list'].append(tau)
                data['t_policy_list'].append(dur)

            if dur < 1/fps:
                time.sleep(1/fps - dur)

        # Gripper action
        if gripper_action == 'open':
            im2client.set_gripper_state(arm, EE_OPEN)
        elif gripper_action == 'close':
            im2client.set_gripper_state(arm, EE_CLOSE)

    except Exception as e:
        print(f'[ERROR | Execute] {e}')
        pass

    return data


#%% Driver function
def main():    
    urdf_dir = 'simulation/assets/urdf/RobotBimanualV5/urdf/Simplify_Robot_plate_gripper_LEFT.urdf'

    joint_pos_lower_limit = np.array([-90.,  -90., -180.,  -45., -210., -125.])*math.pi/180.
    joint_pos_upper_limit = np.array([50.,  80., 80.,  75., 210. , 125.])*math.pi/180.
    joint_vel_upper_limit = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.1750, 2.1750])
    
    start_pos = [0, 0, 0]
    start_orn = [0, 0, 0]

    arm = RobotParams.LEFT
    
    manip = Manipulation(start_pos, 
                         start_orn, 
                         'rrt',
                         robot_name=urdf_dir, 
                         arm=arm, 
                         control_freq=FPS,
                         resolution=MOVE_RESOLUTION, 
                         joint_min=joint_pos_lower_limit, 
                         joint_max=joint_pos_upper_limit, 
                         joint_vel_upper_limit=joint_vel_upper_limit,
                         debug=True)

    im2client = IM2Client(ip=IP, port=PORT)
    
    params = get_params()

    # Traj following
    # init = np.array([0, 0, 0, 0, 1.57, 1.2])
    # term = np.array([0.23, -0.597, -0.627, -0.047, 0.432, -0.689])

    # Grasping
    # init = np.array([-0.416 , -1.0397,  0.2668, -0.1513, -0.4538, -0.5061])
    # term = np.array([-0.7277, -0.1749, -0.3363,  0.5457, -0.2269,  0.1745])

    # Snatching
    init = np.array([-0.0803, -1.3907, -0.3111,  0.7593, -0.9774, -1.7279])
    # pick = np.array([-0.5587, -0.429 ,  0.0227,  0.0792, -0.0524, -0.8552])
    pick = np.array([-0.5781, -0.4412, -0.4191,  0.6258, -0.0902, -1.0435])
    # release = np.array([-1.2263, -0.0673, -0.2394,  0.2409, -0.0788, -0.6254])
    release = np.array([-1.0073, -0.0849, -0.3511,  0.3603, -0.0978, -0.3061])
    

    move_to_init(manip, im2client, arm, init, gripper=EE, gripper_angle=EE_OPEN)

    exec_params = {'manip': manip,
                   'im2client': im2client,
                   'arm': arm,
                   'show': False,
                   'pick': pick,
                   'release': release,
                   'data_dumper': dump_data,
                   'grasp': True,
                   'post_record': False,
                   'grasp_timing': 0.16}
        

    execute(**exec_params)

    manip.close()


if __name__ == '__main__':
    main()

