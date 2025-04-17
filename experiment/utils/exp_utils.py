#%% Import libraries and initialize global variable
import os, sys
from pathlib import Path
BASEDIR = Path(__file__).parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))

from numpy import typing as npt
from typing import List, Tuple, Callable, Union
import time
import numpy as np
import pickle
import experiment.utils.camera as cutil
import torch
from types import SimpleNamespace
import cv2
import argparse

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

import torch.nn.functional as F
from models.keypoint_detector import Keypoint_Detect
from models.policy import *


# Real robot
from control.im2controller_client import IM2Client
from control import DefaultControllerValues as RobotParams

# Manipulation
from simulation.bullet.manipulation import Manipulation, Bimanipulation

# Global variables
DEBUG = False
INITIALIZATION = False
KEYPOINT_CKPT = 'simulation/data/weights/v3_left_keypoints.pth'
POLICY_CKPT =  'Card_ep_18400_rew_1073.5237.pth'
CONTROLLER = 'position' # 'JP'
SAVE_DIR = './'

CONTROL_FPS = 100
POLICY_FPS = 10

IP = '[::]'
PORT = '50051'
TASK = 'Card'


#%% Load keypoint detector
def load_keypoint_detector(checkpoint_dir: str=KEYPOINT_CKPT, debug=DEBUG):
    cam_pipeline = None
    cam_align = None
    if not debug:
        cam_pipeline, cam_align = cutil.camera_init()

    keypoint_detector = Keypoint_Detect(3, 8).to('cuda:0')
    keypoint_detector_ckpt = checkpoint_dir
    state = torch.load(keypoint_detector_ckpt)
    keypoint_detector.load_state_dict(state['model_state_dict'])
    keypoint_detector.eval()

    return cam_pipeline, cam_align, keypoint_detector


#%% Detect keypoint
def detect_keypoint(keypoint_detector: Keypoint_Detect, cam_pipeline, cam_align, image_size=(320, 240), debug=DEBUG):
    
    if debug:
        cur_keypoints = torch.tensor([[260., 110., 258.,  94., 238., 109., 236.,  93., 260., 110., 257.,  94.,
         237., 110., 235.,  94.]], device='cuda:0')
        
        return cur_keypoints, None
    
    cur_image = cutil.get_img(cam_pipeline, cam_align)

    new_size = image_size
    cur_image_resized = cv2.resize(cur_image, new_size, interpolation=cv2.INTER_AREA)

    image_torch = torch.tensor(np.transpose(cur_image_resized, (2, 0, 1)).copy(), dtype=torch.float, device='cuda:0')[None, ...] / 255.0

    results, _ = keypoint_detector(image_torch)
    results = F.softmax(results.reshape(-1, 240 * 320), dim=-1).reshape(-1, 8, 240, 320)
    cur_keypoints = (results.flatten(-2).argmax(-1))
    cur_keypoints = torch.stack([cur_keypoints%320, cur_keypoints//320],-1).to(torch.float).reshape(1, 16)[:]

    return cur_keypoints, cur_image_resized


#%% Load policy
def load_policy(checkpoint_dir: str=POLICY_CKPT):
    model = Model()
    policy_ckpt = checkpoint_dir
    controller_type = "position" # JP or position

    model.load(policy_ckpt)
    model.policy.eval()

    return model


#%% Initialize action and observation scale
def initialize_action_and_observation_scale(controller_type: str, 
                                            joint_pos_upper_limit: torch.Tensor, 
                                            joint_pos_lower_limit: torch.Tensor,
                                            joint_vel_upper_limit: torch.Tensor):
    ee_limits: dict = {
            "ee_position": SimpleNamespace(
                low=torch.tensor([-1, -1, 0], dtype=torch.float, device='cuda:0'),
                high=torch.tensor([1, 1, 1], dtype=torch.float, device='cuda:0')
            ),
            "ee_orientation": SimpleNamespace(
                low=-torch.ones(4, dtype=torch.float, device='cuda:0'),
                high=torch.ones(4, dtype=torch.float, device='cuda:0')
            )
        }

    object_limits: dict = {
        "2Dkeypoint": SimpleNamespace(
            low=torch.tensor([0, 0], dtype=torch.float, device='cuda:0'),
            high=torch.tensor([320, 240], dtype=torch.float, device='cuda:0') #TODO: make this to be changed by the config file
        )
    }

    obs_action_scale = SimpleNamespace(
                    low=torch.full((18,), -1, dtype=torch.float, device='cuda:0'),
                    high=torch.full((18,), 1, dtype=torch.float, device='cuda:0')
                )

    observation_scale = SimpleNamespace(low=None, high=None)
    observation_scale.low = torch.cat([
        joint_pos_lower_limit,
        -joint_vel_upper_limit,
        object_limits["2Dkeypoint"].low.repeat(8),
        object_limits["2Dkeypoint"].low.repeat(8),
        ee_limits["ee_position"].low,
        ee_limits["ee_orientation"].low,
        obs_action_scale.low
    ])
    observation_scale.high = torch.cat([
        joint_pos_upper_limit,
        joint_vel_upper_limit,
        object_limits["2Dkeypoint"].high.repeat(8),
        object_limits["2Dkeypoint"].high.repeat(8),
        ee_limits["ee_position"].high,
        ee_limits["ee_orientation"].high,
        obs_action_scale.high
    ])

    action_scale = SimpleNamespace(low=None, high=None)

    if controller_type == "JP":
        initial_residual_scale = [0.05, 0.08]
        position_scale = initial_residual_scale[0]
        orientation_scale = initial_residual_scale[1]

        residual_scale_low = [-position_scale]*3+[-orientation_scale]*3
        residual_scale_high = [position_scale]*3+[orientation_scale]*3

        action_scale.low = to_torch(
                            residual_scale_low+[0.1]*6+[0.0]*6, 
                            device='cuda:0'
                        )
        action_scale.high = to_torch(
                            residual_scale_high+[1.8, 2.2, 1.6, 1.6, 0.7, 0.7]+[0.06, 0.08, 0.1, 0.1, 0.04, 0.04], 
                            device='cuda:0'
                        )
        
    elif controller_type == "position":
        action_scale.low = to_torch(
                            # [-0.30, -0.35, -0.45, -0.40, -0.30, -0.35]+[0.9, 1.2, 0.95, 0.9, 0.3, 0.35]+[0.0]*6,
                            # [-0.36 , -0.42 , -0.54 , -0.48 , -0.36 , -0.42, 0.9, 1.2, 0.95, 0.9, 0.3, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [-0.3948222038857476, -0.46062590453337227, -0.5922333058286213, -0.5264296051809969, -0.3948222038857476, -0.46062590453337227]+[0.9, 1.2, 0.95, 0.9, 0.3, 0.35]+[0.0]*6,
                            device='cuda:0'
                        )
        action_scale.high = to_torch(
                            # [0.30, 0.35, 0.45, 0.40, 0.30, 0.35]+[1.8, 2.4, 1.9, 1.8, 0.6, 0.7] + [0.04, 0.06, 0.06, 0.06, 0.01, 0.01], 
                            # [0.36 , 0.42 , 0.54 , 0.48 , 0.36 , 0.42, 1.8, 2.4, 1.9, 1.8, 0.6, 0.7, 0.048, 0.072, 0.072, 0.072, 0.01, 0.01],
                            [0.3948222038857476, 0.46062590453337227, 0.5922333058286213, 0.5264296051809969, 0.3948222038857476, 0.46062590453337227] + [2.3689332233144853, 3.158577631085981, 2.259493518505171, 2.1405728070049, 0.7896444077714952, 0.9212518090667445] + [0.05264296051809967, 0.07896444077714952, 0.07896444077714952, 0.07896444077714952, 0.013160740129524917, 0.013160740129524917],
                            device='cuda:0'
                        )
    
    elif controller_type == "position_2":
        action_scale.low = to_torch(
                            # [-0.7286617 , -1.14922249, -1.16561084, -0.97154894, -0.32237087, -0.32237087]+[1.09649068, 1.46198757, 1.17398177, 1.12428516, 0.42882231, 0.42882231]+[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            # [-0.54, -0.84, -0.855, -0.72, -0.245, -0.245, 0.9, 1.2, 0.95, 0.9, 0.35, 0.35, 0, 0, 0, 0, 0, 0],
                            # [-0.66241973, -1.04474772, -1.05964622, -0.88322631, -0.29306443, -0.29306443]+[0.99680971, 1.32907961, 1.06725616, 1.02207742, 0.38983846, 0.38983846]+[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [-0.495, -0.726, -0.792, -0.704, -0.294, -0.221]+[0.9, 1.1, 0.8, 0.8, 0.358, 0.273]+[0.0, 0.0, 0.0, 0.0, 0.078, 0.0614],
                            device='cuda:0'
                        ) 
        action_scale.high = to_torch(
                            # [0.7286617 , 1.14922249, 1.16561084, 0.97154894, 0.32237087, 0.32237087]+[2.19298136, 2.92397514, 2.34796354, 2.24857031, 0.85764462, 0.85764462]+[0.03, 0.045, 0.45, 0.45, 0.01, 0.01],
                            # [0.54, 0.84, 0.855, 0.72, 0.245, 0.245, 1.8, 2.4, 1.9, 1.8, 0.7, 0.7, 0.03, 0.045, 0.45, 0.45, 0.01, 0.01],
                            # [0.66241973, 1.04474772, 1.05964622, 0.88322631, 0.29306443, 0.29306443]+[1.99361942, 2.65815922, 2.13451231, 2.04415483, 0.77967693, 0.77967693]+[0.03, 0.045, 0.45, 0.45, 0.01, 0.01],
                            [0.495, 0.726, 0.792, 0.704, 0.294, 0.221]+[1.8, 2.2, 1.6, 1.6, 0.862, 0.597] + [0.06, 0.08, 0.1, 0.1, 0.102, 0.0926],
                            device='cuda:0'
                        )
        
    return action_scale, observation_scale


#%% Dump experiment data
def dump_data(filename: str,
              current_time: float, 
              time_list: List[float],
              joint_pos_list: List[npt.NDArray],
              joint_vel_list: List[npt.NDArray],
              joint_tau_list: List[npt.NDArray],
              des_pos_list: List[npt.NDArray],
              t_policy_list: List[float],
              obs_list: List[npt.NDArray],
              act_list: List[npt.NDArray],
              img_list: List[npt.NDArray],
              des_tau_list: List[npt.NDArray],
              save_dir: str,
              **kwargs
              ):

    # overwrite the name
    os.makedirs(save_dir, exist_ok=True)

    with open(save_dir+f'{filename}_{current_time}.pkl', 'wb') as f:
            pickle.dump({'time':np.array(time_list), 
                         'joint_pos':np.array(joint_pos_list),
                         'joint_vel':np.array(joint_vel_list),
                         'joint_tau':np.array(joint_tau_list),
                         'des_pos': np.array(des_pos_list),
                         't_policy_list':np.array(t_policy_list),
                         'obs': np.array(obs_list), 
                         'act': np.array(act_list), 
                         'imgs': img_list, 
                         'des_tau_list':np.array(des_tau_list),
                         'trajectory': kwargs.get('trajectory', [])}, 
                         f)



def move(arm: str, im2client: IM2Client, traj: List[npt.NDArray],
         p_gains: npt.NDArray, d_gains: npt.NDArray, fps: float=POLICY_FPS):
    
    try:
        for q in traj:
            st = time.time()
            im2client.joint_pd_control(arm, q, p_gains, d_gains)
            dur = time.time() - st
            if dur < 1/fps:
                time.sleep(1/fps - dur)
    except Exception as e:
        return False
    
    return True


def get_target_size(task: str) -> Tuple[float, float, float]:
    if task in ['Card', 'card']:
        size = (0.05, 0.07, 0.005)
    elif task in ['Bump', 'bump']:
        size = (0.09, 0.09, 0.09)
    else:
        raise ValueError('Wrong task name')
    
    return size



def project_target_to_camera_coord(object_pose: torch.Tensor, size: Tuple[float, float, float]):
    goal_keypoints_3d = gen_keypoints(pose=object_pose, size=size)

    camera_matrix = torch.tensor([[192.8216,   0.0000, 160.0000],
                                [  0.0000, 209.8376, 120.0000],
                                [  0.0000,   0.0000,   1.0000]], device='cuda:0')

    translation_from_camera_to_object = torch.tensor([[ 1.0235e-16, -1.0000e+00,  8.3267e-17, -9.7665e-17],
                                                    [-9.3969e-01, -1.1970e-16, -3.4202e-01,  4.5309e-01],
                                                    [ 3.4202e-01, -2.7756e-17, -9.3969e-01,  9.5248e-01]], device='cuda:0')

    goal_keypoints_2d = compute_projected_points(translation_from_camera_to_object, goal_keypoints_3d, camera_matrix, 'cuda:0').reshape(1, 16)[:]

    return goal_keypoints_2d


def keyboard_prompt(prompt: str):

    key = input(prompt)

    if len(key) > 1:
        key = key[0]

    return key


def get_params(policy_ckpt: str=POLICY_CKPT,
               keypoint_ckpt: str=KEYPOINT_CKPT,
               ip: str=IP,
               port: str=PORT,
               task: str=TASK):
    parser = argparse.ArgumentParser(description='Run real robot experiment')
    parser.add_argument('--policy', default=policy_ckpt, help='policy checkpoint directory')
    parser.add_argument('--keypoint', default=keypoint_ckpt, help='keypoint detector checkpoint directory')
    parser.add_argument('--ip', type=str, default=ip)
    parser.add_argument('--port', type=str, default=port)
    parser.add_argument('--task', type=str, default=task, help='task to solve [Bump, Card]')

    return parser.parse_args()


def move_to_init(manip: Manipulation,
                 im2client: IM2Client,
                 arm: str, 
                 init: Union[npt.NDArray, Tuple[str]],
                 duration: float=2.0,
                 gripper: bool=False,
                 gripper_angle: Union[float, Tuple[float]]=None):
    if arm == 'all':
        arm = (RobotParams.LEFT, RobotParams.RIGHT)

        init_L = init[:RobotParams.DOF]
        init_R = init[-RobotParams.DOF:]

        init = (init_L, init_R)

    else:
        arm = (arm,)

        if isinstance(init, np.ndarray):
            init = (init,)

    manip.planner = 'interpolate'
    for a, q in zip(arm, init):
        cur, _, _, _ = im2client.get_current_states(a)
        traj = manip.motion_plan(arm, cur, q, duration=duration)
        for q in traj:
            st = time.time()
            im2client.position_control(a, q,
                                       desired_p_gains=RobotParams.POS_P_GAIN,
                                       desired_d_gains=RobotParams.POS_D_GAIN)
            dur = time.time() - st
            if dur < 1/manip.control_freq:
                time.sleep(1/manip.control_freq - dur)
    manip.planner = 'rrt'

    if gripper:
        assert gripper_angle is not None, 'Must provide initial gripper angle to be set'
        if not isinstance(gripper_angle, tuple):
            gripper_angle = (gripper_angle,)

        for a, angle in zip(arm, gripper_angle):
            im2client.set_gripper_state(a, angle)

            