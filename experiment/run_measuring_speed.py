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
SAVE_DIR = 'experiment/speed/'
DOF = RobotParams.DOF
FPS = 200
SNATCH_RESOLUTION = 0.020 # original snatch is 0.015 (dangerously fast)
MOVE_RESOLUTION = 0.003 # original move is 0.005 (still fast)

IP = '[::]'
PORT = '50051'

VIZ = False
EE =  False

def generate_trajectory(init, mid, end, num_init_mid, num_mid_end):
    """
    Generate a trajectory with smooth speed transitions, sparse points near the midpoint, 
    and denser points near the endpoint.
    
    Args:
        init (np.array): Initial configuration of the robot (6-DoF).
        mid (np.array): Midpoint configuration of the robot (6-DoF).
        end (np.array): Endpoint configuration of the robot (6-DoF).
        num_init_mid (int): Number of waypoints between init and mid.
        num_mid_end (int): Number of waypoints between mid and end.
        
    Returns:
        np.array: Array of waypoints representing the trajectory.
    """
    # Generate spacing weights (dense near endpoints)
    init_mid_weights = np.linspace(0, 1, num_init_mid) # Quadratic spacing
    mid_end_weights = np.linspace(0, 1, num_mid_end)  # Reverse quadratic spacing

    # Interpolate waypoints between init and mid
    init_mid_waypoints = [
        init * (1 - w) + mid * w for w in init_mid_weights
    ]
    
    # Interpolate waypoints between mid and end
    mid_end_waypoints = [
        mid * (1 - w) + end * w for w in mid_end_weights
    ]
    
    return init_mid_waypoints, mid_end_waypoints


def execute(manip: Manipulation, 
            im2client: IM2Client,
            arm: str,
            mid: npt.NDArray,
            end: npt.NDArray,
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
    
    
    # snatched_data = snatch(manip, 
    #                        im2client, 
    #                        arm,
    #                        mid,
    #                        end,
    #                        data=empty_data, 
    #                        fps=fps,
    #                        show=show,
    #                        **kwargs)
        
    snatched_data = snatch_continuous(manip, 
                           im2client, 
                           arm,
                           mid,
                           end,
                           data=empty_data, 
                           fps=fps,
                           show=show,
                           **kwargs)
    
    data_dumper(**snatched_data)

    return True


def snatch(manip: Manipulation, 
           im2client: IM2Client,
           arm: str,
           mid,
           end,
           data: Dict[str, Union[List, npt.NDArray, str]]=None,
           fps: float=FPS,
           show: bool=False,
           **kwargs):
    
    manip.resolution = SNATCH_RESOLUTION
    init, _, _, _ = im2client.get_current_states(arm)
    traj = manip.motion_plan(arm, init, mid)
    
    if len(traj) == 0:
        return False

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
        for i in range(int(fps*0.3)):
            # Log time
            st = time.time()

            # read pos
            pos, vel, tau, grav = im2client.get_current_states(arm)
            im2client.position_control(arm, pos, 
                                       desired_p_gains=np.zeros(RobotParams.DOF),
                                       desired_d_gains=RobotParams.SNATCH_D_GAIN*3)

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

def snatch_continuous(manip: Manipulation, 
           im2client: IM2Client,
           arm: str,
           mid,
           end,
           data: Dict[str, Union[List, npt.NDArray, str]]=None,
           fps: float=FPS,
           show: bool=False,
           **kwargs):
    
    manip.resolution = SNATCH_RESOLUTION
    init, _, _, _ = im2client.get_current_states(arm)

    # len(traj) = 48 with resolution = 0.015
    # len(traj) = 30 with resolution = 0.024
    num_init_mid = 36
    num_mid_end = 100

    traj_init_mid, traj_mid_end = generate_trajectory(init, mid, end, num_init_mid, num_mid_end)

    if show: 
        # manip.simulate_traj(arm, traj_init_mid, draw=True)
        # manip.simulate_traj(arm, traj_mid_end, draw=True)
        print('test')

    try:
        prev_joint_pos, cur_joint_vel, _, _ = im2client.get_current_states(arm)
        for i, q in enumerate(traj_init_mid):
            # Log time
            st = time.time()

            # Reconstruct action
            delta_joint_pos = q - prev_joint_pos

            kps = RobotParams.SNATCH_P_GAIN
            kds = RobotParams.SNATCH_D_GAIN

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
        for i, q in enumerate(traj_mid_end):
            # Log time
            st = time.time()

            # Reconstruct action
            delta_joint_pos = q - prev_joint_pos

            kps = 0. * RobotParams.SNATCH_P_GAIN

            if i < 50:
                kds = 2.2 * i / 50. * RobotParams.SNATCH_D_GAIN
            else:
                kds = 2.2 * RobotParams.SNATCH_D_GAIN

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
                         'interpolate',
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
    init = np.array([-1.04719741, -1.04719735, -1.57079618, -0.43633217, -0.69813156, -0.69813156])
    # mid = np.array([-0.35257405, -0.352574  , -0.87617282,  0.25829119, -0.0035082, -0.0035082 ])
    mid = np.array([-0.20943954, -0.20943948, -0.73303831,  0.4014257 ,  0.13962631, 0.13962631])
    end = np.array([ 0.34906571,  0.34906576, -0.17453307,  0.95993095,  0.69813156, 0.69813156])

    # init = np.array([-1.04719741, -1.04719735, -1.57079618, -0.43633217, 0., 0.])
    # mid = np.array([-0.35257405, -0.352574  , -0.87617282,  0.25829119,  0., 0.])
    # end = np.array([ 0.34906571,  0.34906576, -0.17453307,  0.95993095,  0., 0.])

    move_to_init(manip, im2client, arm, init, gripper=EE)
    # manip.planner = 'interpolate'

    exec_params = {'manip': manip,
                   'im2client': im2client,
                   'arm': arm,
                   'show': True,
                   'mid': mid,
                   'end': end,
                   'data_dumper': dump_data,
                   'grasp': True,
                   'post_record': False}
        

    execute(**exec_params)

    manip.close()


if __name__ == '__main__':
    main()

