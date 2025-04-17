#%% Import libraries and initialize global variable
import os, sys
from pathlib import Path
BASEDIR = Path(__file__).parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

import numpy as np
from numpy import typing as npt
from typing import List, Tuple, Callable, Dict
import time, datetime

# Control
from control.im2controller_client import IM2Client
from control import DefaultControllerValues as RobotParams

# Utils
from experiment.utils.exp_utils import move_to_init
from imitation.utils.exp_utils import *
from imitation.utils.transforms import matrix_to_quaternion
# Retarget
from imitation.retargeter import Retargeter

# Log
from loguru import logger

SIM = True
COLLECT = False
SAVE_DIR = 'imitation/data/'
DOF = RobotParams.DOF
FPS = 30

IP = '[::]'
PORT = '50051'

TAG_OFFSET_FILE = 'imitation/perception/pose/runtime/apriltag_offset.pkl'
DEBUG = False

class ShadowRunner():

    def __init__(self, 
                 im2client: IM2Client, 
                 config: Dict,
                 fps: float=FPS,
                 simulate: bool=SIM,
                 calibrate_gripper: bool=False,
                 collect_data: bool=True,
                 no_robot: bool=False):
        
        self.im2client = im2client
        self.config = config

        # Data collection
        self.calibrate_gripper = calibrate_gripper
        self.collect_data = collect_data
        self.save_dir = config.get('save_dir', SAVE_DIR)

        # Body tracker
        self.retargeter: Retargeter = Retargeter(**config)
        self.body_cam: rs.pyrealsense2.pipeline = self.retargeter.body_camera

        # Object tracker
        if self.collect_data:
            obj_cam, object_tracker = build_obj_tracker(**config)        
            self.obj_cam: MultiRSCamera = obj_cam
            self.object_tracker: Perception = object_tracker
        else:
            self.obj_cam, self.object_tracker = None, None

        self.fps = fps

        # No-robot mode for debugging
        self.simulate = simulate
        self.no_robot = no_robot
        


    def initialize(self):
        
        logger.info('[Initialization] Trying to move the arm to the initial position. Please stay still')

        init, g = self.retargeter.shadow()

        if not self.no_robot:
            move_to_init(self.retargeter.manip, self.im2client, 'all', init, duration=1.5)
            # self.im2client.set_gripper_state(RobotParams.LEFT, RobotParams.L_GRIPPER[int(g[0])])
            # self.im2client.set_gripper_state(RobotParams.RIGHT, RobotParams.R_GRIPPER[int(g[1])])

        if self.simulate:
            self.retargeter.manip.robot.setJointStates('all', init)

        time.sleep(1)
        logger.info('[Initalization] Robot caught up your body pose. You are free to move')

        if self.calibrate_gripper:
            self.grasp_threshold = self.retargeter.calibrate_gripper()


    def read_observation(self) -> npt.NDArray:
        obj_img: Dict[str, npt.NDArray] = self.obj_cam()
        out = self.object_tracker(obj_img)
        obj_pose = out['obj_pose']
        matrix_to_quaternion(obj_pose)

        return matrix_to_quaternion(obj_pose)


    def execute(self):
        '''
        Loop through shadow pipeline to control robot real time
        '''

        data = make_empty_data(self.save_dir)

        try:
            kps = RobotParams.SHADOW_P_GAIN
            kds = RobotParams.SHADOW_D_GAIN

            # Make robot move to the current body position to preven huge swing at start
            self.initialize()

            if not self.no_robot:
                pos_L, _, _, _ = self.im2client.get_current_states(RobotParams.LEFT)
                pos_R, _, _, _ = self.im2client.get_current_states(RobotParams.RIGHT)
                prev_joint_pos = np.concatenate((pos_L, pos_R))
                prev_g = np.array((0, 0))


            if DEBUG:
                debug_data = {'pred_cam': [], 'wrist_pose': [], 'quat_traj': [], 'global_orient': []}

            while True:
                st = time.time()

                if self.collect_data:
                    obs = self.read_observation()
                else:
                    obs = None
                if not DEBUG:
                    q, g = self.retargeter.shadow2()
                else:
                    q, g, debug_data = self.retargeter.shadow(debug=DEBUG, data=debug_data)
                # logger.info(f'Robot joint angle: {q*180/np.pi}  Grasp: {g}')

                if self.simulate:
                    self.retargeter.manip.robot.setJointStates('all', q)
                    # if g[0]:
                    #     logger.info(f'LEFT gripper closed')
                    # if g[1]:
                    #     logger.info(f'RIGHT gripper closed')

                if self.no_robot:
                    dur = time.time() - st
                    if dur < 1/self.fps:
                        time.sleep(1/self.fps - dur)
                    logger.info(f'Cycle: {dur} s')
                    continue

                # Reconstruct action
                delta_joint_pos = q - prev_joint_pos
                act = np.concatenate((delta_joint_pos, kps, kds))
                
                # predict velocity
                target_joint_vel = delta_joint_pos*self.fps
                main_q = q[:RobotParams.DOF]
                sub_q = q[RobotParams.DOF:]
                main_v = target_joint_vel[:RobotParams.DOF]
                sub_v = target_joint_vel[RobotParams.DOF:]
                pos = []
                vel = []
                tau = []
                grav = []
                for single_q, single_v, single_arm in zip([main_q, sub_q],[main_v, sub_v],[RobotParams.LEFT, RobotParams.RIGHT]):
                    self.im2client.position_control(single_arm, 
                                                    single_q, 
                                                    desired_p_gains=kps,
                                                    desired_d_gains=kds,
                                                    desired_velocity=single_v)
                    single_pos, single_vel, single_tau, single_grav = self.im2client.get_current_states(single_arm)

                    # Gripper
                    # if single_arm == RobotParams.LEFT:
                    #     if prev_g[0] != g[0]:
                    #         self.im2client.set_gripper_state(single_arm, RobotParams.L_GRIPPER[g[0]])

                    # if single_arm == RobotParams.RIGHT:
                    #     if prev_g[1] != g[1]:
                    #         self.im2client.set_gripper_state(single_arm, RobotParams.R_GRIPPER[g[1]])

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
                    data['obs_list'].append(obs)
                    data['des_pos_list'].append(q)
                    data['joint_pos_list'].append(pos)
                    data['joint_vel_list'].append(vel)
                    data['joint_tau_list'].append(tau)
                    data['des_tau_list'].append(tau-grav)
                    data['t_policy_list'].append(dur)

                if dur < 1/self.fps:
                    time.sleep(1/self.fps - dur)

        except KeyboardInterrupt:
            if self.collect_data:
                filename = 'human_shadowing_data'
                timestamp = datetime.datetime.now().strftime("%I%M%p_%B%d%Y")
                data['filename'] = filename
                data['current_time'] = timestamp

                dump_data(**data)

            logger.info('[Shutdown] Trying to move the arm to the default position.')

            default_q = np.zeros((RobotParams.DOF*2,))

            if not self.no_robot:
                move_to_init(self.retargeter.manip, self.im2client, 'all', default_q, duration=1.5)
                # self.im2client.set_gripper_state(RobotParams.LEFT, RobotParams.L_GRIPPER[0])
                # self.im2client.set_gripper_state(RobotParams.RIGHT, RobotParams.R_GRIPPER[0])

            logger.info('[Shutdown] Robot moved back to the default position.')
            
        except UnboundLocalError:
            if DEBUG:
                logger.info('[DEBUG] Dumping debug data')
                with open('right_hand_debug_data.pkl', 'wb') as f:
                    pickle.dump(debug_data, f)




#%% Driver function
def main():

    args = get_params()
    NO_ROBOT = args.no_robot    
       
    config = {'width': 640,
              'height': 480,
              'cam_fps': 30,
              'gui': NO_ROBOT,
              'device': 'cuda:0',
              'grasp_threshold': 0.07,
              'visualize_ik': False,
              'cal_file': TAG_OFFSET_FILE}

    if NO_ROBOT:
        im2client = None
    else:
        im2client = IM2Client(ip=IP, port=PORT)
    runner = ShadowRunner(im2client, config, FPS, collect_data=COLLECT, calibrate_gripper=args.calibrate_gripper, no_robot=NO_ROBOT)
    
    # How can we stop robot moving abruptly at the beginning?
    runner.execute()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()

