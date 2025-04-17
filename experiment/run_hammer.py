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

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

# SysID simulation
from simulation.bullet.manipulation import Manipulation

# Experiment
from control.im2controller_client import IM2Client
from control import DefaultControllerValues as RobotParams
from experiment.utils.exp_utils import *

DEBUG = False
INITIALIZATION = False
SAVE_DIR = 'experiment/hammering/'
DOF = RobotParams.DOF
FPS = 200
HAMMER_RESOLUTION = 0.02
MOVE_RESOLUTION = 0.001

IP = '[::]'
PORT = '50051'

VIZ = False

def execute(manip: Manipulation, 
            im2client: IM2Client,
            arm: str,
            term: npt.NDArray,
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

    empty_data = {'filename': 'hammer_test',
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
    
    
    hammered_data = hammer(manip, 
                           im2client, 
                           arm,
                           term, 
                           data=empty_data, 
                           fps=fps,
                           show=show,
                           **kwargs)
        
    
    data_dumper(**hammered_data)

    return True


def hammer(manip: Manipulation, 
           im2client: IM2Client,
           arm: str,
           term,
           data: Dict[str, Union[List, npt.NDArray, str]]=None,
           fps: float=FPS,
           show: bool=False,
           **kwargs):
    
    manip.resolution = HAMMER_RESOLUTION

    init, _, _, _ = im2client.get_current_states(arm)
    # traj = manip.motion_plan(arm, init, term, allow_uid_list=[manip.scene['table']])
    traj = manip.motion_plan(arm, init, term)
    
    if len(traj) == 0:
        return False

    kps = RobotParams.HAMMER_P_GAIN
    kds = RobotParams.HAMMER_D_GAIN

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



#%% Driver function
def main():    
    joint_pos_lower_limit = np.array([-90.,  -90., -180.,  -45., -210., -125.])*math.pi/180.
    joint_pos_upper_limit = np.array([50.,  80., 80.,  75., 210. , 125.])*math.pi/180.
    joint_vel_upper_limit = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.1750, 2.1750])

    joint_pos_lower_limit = RobotParams.R_JOINT_LIMIT_MIN
    joint_pos_upper_limit = RobotParams.R_JOINT_LIMIT_MAX
    
    start_pos = [0, 0, 0]
    start_orn = [0, 0, 0]

    arm = RobotParams.RIGHT
    
    manip = Manipulation(start_pos, 
                         start_orn, 
                         'rrt', 
                         arm=arm, 
                         control_freq=FPS,
                         resolution=MOVE_RESOLUTION,
                         tol=5e-3,
                         allow_selfcollision=True, 
                         joint_min=joint_pos_lower_limit, 
                         joint_max=joint_pos_upper_limit, 
                         joint_vel_upper_limit=joint_vel_upper_limit,
                         debug=True)

    im2client = IM2Client(ip=IP, port=PORT)
    
    # init = np.array([0.362, -0.294, -0.501, -0.245, 1.582, 1.102])
    # init = np.array([ -0.0479, 0.1589, 1.8717, 0.3698,  -1.6226,  -1.8462])
    # init = np.array([ 0.0624,  0.017 ,  1.412 ,  0.2256, -1.6035, -1.4094]) # toy hammer right
    # init = np.array([-0.0021, -0.0051, -0.1219, -0.104 ,  1.544 ,  0.9558])
    # term = np.array([-0.088, -0.156, 0.344, 0.118, 1.505, 0.390])
    # term = np.array([ 0.0273,  0.0341, -0.5808,  0.1051, -1.5204, -0.6857]) # toy hammer right
    # init = np.array([ 0.0124, -0.0475, -0.0299, -0.355 , -1.6232,  1.117 ])
    # init = np.array([ 0.0002, -0.0475, -1.5288, -0.1814, -1.5929, -0.7876])
    # term = np.array([-0.0822, -0.0402,  0.0677,  0.0521,  -1.6057,  1.6057])
    init = np.array([0.02  , 0.062 , 1.0176, 0.3649, 1.639 , 0.1345])
    term = np.array([ 0.0128,  0.0277, -0.1139, -0.0353,  1.5372, -1.2362])
    
    for i in range(100):

        move_to_init(manip, im2client, arm, init, duration=1.5)

        exec_params = {'manip': manip,
                       'im2client': im2client,
                       'arm': arm,
                       'show': False,
                       'term': term,
                       'data_dumper': dump_data,
                       'post_record': True}
            

        execute(**exec_params)

    manip.close()


if __name__ == '__main__':
    main()

