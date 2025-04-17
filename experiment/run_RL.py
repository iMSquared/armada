#%% Import libraries and initialize global variable
import os, sys
from pathlib import Path
BASEDIR = Path(__file__).parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))

from numpy import typing as npt
from typing import List, Tuple, Callable
import datetime, time
import numpy as np
import copy
import torch
import argparse

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

# Experiment
from control.im2controller_client import IM2Client
from experiment.utils.exp_utils import *

# Global variables
DEBUG = False
INITIALIZATION = False

# KEYPOINT_CKPT = '241224_bump_keypoint.pth'
# POLICY_CKPT =  '241224_bump_NOdeadzone.pth'

KEYPOINT_CKPT = '/home/user/workspace/250103_dataset_new_view/result/25-01-03-17-22/epoch_80_8points.pth'
POLICY_CKPT =  '240106_bump_dr.pth'

# KEYPOINT_CKPT = 'simulation/data/weights/v3_left_keypoints.pth'
# POLICY_CKPT = 'simulation/data/weights/Card_ep_3800_rew_1000.3785.pth'
CONTROLLER = 'position' # 'JP'
SAVE_DIR = './experiment/result/'
DOF = 6
CONTROL_FPS = 200
POLICY_FPS = 10

TASK = 'Bump'


#%% execute policy
def execute(im2client: IM2Client,
            task: str, 
            arm: str, 
            policy_freq: float,
            keypoint_detector: Keypoint_Detect, 
            cam_pipeline, cam_align,
            goal_keypoints_2d: torch.Tensor, 
            base_posquat: npt.NDArray,
            action_scale, 
            observation_scale, 
            model: Model, 
            controller_type: str,
            joint_pos_upper_limit: torch.Tensor, 
            joint_pos_lower_limit: torch.Tensor,
            prev_action: npt.NDArray,
            data_dumper: Callable=dump_data,
            obj_keypoints: npt.NDArray=None,
            timeout: float=None,
            collision_fn: Callable=None,
            skip_done: bool=False,
            **kwargs
            ):
    

    time_list = []
    des_pos_list = []
    joint_pos_list = []
    joint_vel_list = []
    joint_tau_list = []
    t_policy_list = []
    obs_list = []
    img_list = []
    act_list = []
    des_tau_list = []

    max_tau = im2client.max_tau

    assert max_tau is not None, 'Max Torques are not loaded in the client'

    exec_start = time.time()

    emergency_flag = False

    pos_dof_lower_safety_limit = (0.97 * joint_pos_lower_limit + 0.03 * joint_pos_upper_limit).clone().detach().cpu().numpy()
    pos_dof_upper_safety_limit = (0.03 * joint_pos_lower_limit + 0.97 * joint_pos_upper_limit).clone().detach().cpu().numpy()

    action_dof_lower_safety_limit = (0.95*joint_pos_lower_limit + 0.05*joint_pos_upper_limit).clone().detach().cpu().numpy()
    action_dof_upper_safety_limit = (0.05*joint_pos_lower_limit + 0.95*joint_pos_upper_limit).clone().detach().cpu().numpy()

    prev_keypoints = None
    KEYPOINT_ERROR_FLAG = False

    try:
        while True:
            total_exec_time = time.time() - exec_start
            if timeout is not None and total_exec_time > timeout:
                break
            t_policy_start = time.time()

            cur_joint_pos, cur_joint_vel, cur_joint_tau, cur_grav_tau = im2client.get_current_states(arm)

            

            # Check if any joint position exceeds the safety limits
            if np.any(cur_joint_pos[4:] < pos_dof_lower_safety_limit[4:]) or np.any(cur_joint_pos[4:] > pos_dof_upper_safety_limit[4:]):
                print('current RMD joint position is out of limit: ', cur_joint_pos[4:])
                break

            joint_pos_list.append(cur_joint_pos)
            joint_vel_list.append(cur_joint_vel)
            joint_tau_list.append(cur_joint_tau)

            cur_joint_pos_torch = torch.tensor(cur_joint_pos, device='cuda:0')[None, ...]
            cur_joint_vel_torch = torch.tensor(cur_joint_vel, device='cuda:0')[None, ...]

            # cam_t = time.time()
            cur_keypoints, cur_image_resized = detect_keypoint(keypoint_detector, cam_pipeline, cam_align)

            # if prev_keypoints is None:
            #     prev_keypoints = cur_keypoints

            # keyp_diff = np.max(np.linalg.norm((cur_keypoints.clone().detach().cpu().numpy().reshape(8, 2)-prev_keypoints.clone().detach().cpu().numpy().reshape(8, 2)), axis=1))
            # threshold = 55.0

            # if keyp_diff > threshold:
            #     if KEYPOINT_ERROR_FLAG:
            #         print(f"consecutive keypoint error: {keyp_diff}")
            #         KEYPOINT_ERROR_FLAG = False
            #         continue

            #     else:
            #         print(f"keypoint anomaly detected: {keyp_diff}. previous keypoint is used")
            #         cur_keypoints = prev_keypoints
            #         KEYPOINT_ERROR_FLAG = True

            # else:
            #     KEYPOINT_ERROR_FLAG = False
            
            # prev_keypoints = cur_keypoints
            

            # print(keyp_diff)
            # print("camera time: ", time.time()-cam_t)

            # Overwrite cur_keypoints if object keypoint is given (this is for SysID)
            if obj_keypoints is not None:
                cur_keypoints = obj_keypoints

            if task in ['Card', 'card', 'Bump', 'bump']:
                if torch.mean(torch.sqrt((cur_keypoints[0]-goal_keypoints_2d[0])**2)) <= 5.0:
                    print('succeed')
                    break
            # elif task in ['Bump', 'bump']:
            #     reshaped_cur_keypoints = cur_keypoints.reshape((-1, 2))
            #     reshaped_goal_keypoints = goal_keypoints_2d.reshape((-1, 2))
            #     if torch.norm((torch.mean(reshaped_cur_keypoints, dim=0)-torch.mean(reshaped_goal_keypoints, dim=0))) <= 4.0:
            #         print('succeed')
            #         break
            # else:
            #     raise ValueError('Wrong task name')

            ee_rel_posquat = im2client.get_ee_pose(arm)
            ee_rel_posquat = np.concatenate(ee_rel_posquat)

            hand_pose = torch.tensor(transform_pose(base_posquat, ee_rel_posquat), dtype=torch.float, device='cuda:0')[None, ...]

            if hand_pose[0, 6] < 0:
                hand_pose[0, 3:7] = -hand_pose[0, 3:7]

            observation = torch.concat((cur_joint_pos_torch, cur_joint_vel_torch, cur_keypoints, goal_keypoints_2d, hand_pose, prev_action), axis=1).to(torch.float)
            scaled_obs = scale_transform(observation, 
                                         lower=observation_scale.low,
                                         upper=observation_scale.high)
            scaled_obs = torch.clamp(scaled_obs, -1.0, 1.0)
            action, _, _, _, _ = model.step(scaled_obs)
            action = torch.clamp(action, -1.0, 1.0)
            prev_action = action.clone()

            action_transformed_numpy = unscale_transform(action,
                                                         lower=action_scale.low,
                                                         upper=action_scale.high).clone().detach().cpu().numpy()[0]

            delta_joint_pos = action_transformed_numpy[:6]

            target_joint_pos = cur_joint_pos + delta_joint_pos

            

            # Input to the client
            target_joint_pos = np.clip(target_joint_pos, action_dof_lower_safety_limit, action_dof_upper_safety_limit)
            p_gains = action_transformed_numpy[6:12]
            d_gains = action_transformed_numpy[12:]
            
            # Collision check
            if collision_fn is not None:
                if collision_fn(target_joint_pos)[0]:
                    return False

            # Tuning d gain
            # d_gains[4:] = 1.2*d_gains[4:]
            # p_gains[5] *= 1.2
            # p_gains = np.zeros_like(p_gains)
            # d_gains = np.zeros_like(d_gains)
            # target_joint_pos[5] = 0

            t_policy_time = time.time() - t_policy_start
            # print('time: ', t_policy_time)
            t_policy_list.append(t_policy_time)
            act_list.append(action_transformed_numpy)
            obs_list.append(observation.clone().detach().cpu().numpy()[0])
            img_list.append(cur_image_resized)

            im2client.joint_pd_control(arm, target_joint_pos, p_gains=p_gains, d_gains=d_gains)

            # cur_joint_pos, cur_joint_vel, cur_joint_tau, cur_grav_tau = im2client.get_current_states(arm)
            # des_tau_list.append(cur_joint_tau - cur_grav_tau)

            if t_policy_time < (1.0 / policy_freq):
                time.sleep((1.0 / policy_freq) - t_policy_time)

    except KeyboardInterrupt:
        pass

    current_time = datetime.datetime.now().strftime("%I%M%p_%B%d%Y")
    params = {'filename': kwargs.get('dump_filename', 'real_policy_data'),
              'current_time': current_time,
              'time_list': time_list,
              'joint_pos_list': joint_pos_list,
              'joint_vel_list': joint_vel_list,
              'joint_tau_list': joint_tau_list,
              'des_pos_list': des_pos_list,
              't_policy_list': t_policy_list,
              'obs_list': obs_list,
              'act_list': act_list,
              'img_list': img_list,
              'des_tau_list': des_tau_list}
    params.update(kwargs)

    data_dumper(**params)
            
    print("file saved")

    des_joint_pos = copy.deepcopy(cur_joint_pos)

    if not skip_done:
        try:
            im2client.joint_pd_control(arm, des_joint_pos, p_gains=np.zeros_like(p_gains), d_gains=d_gains)
            cur_joint_pos, cur_joint_vel, cur_joint_tau, cur_grav_tau = im2client.get_current_states(arm)

            if np.any(np.abs(cur_joint_tau) > max_tau): 
                print("max torque exceeded")
                    
        except KeyboardInterrupt: pass

    return True


#%% Driver function
def main():
    controller_type = CONTROLLER
    arm = 'left'
    policy_freq = POLICY_FPS

    parsed_args = get_params(policy_ckpt=POLICY_CKPT, 
                             keypoint_ckpt=KEYPOINT_CKPT,
                             task=TASK)

    joint_pos_lower_limit = torch.tensor([-90.,  -50., -180.,  -45., -210., -120.], device='cuda:0')*math.pi/180.
    joint_pos_upper_limit = torch.tensor([50.,  50., 50.,  75., 210. , 120.], device='cuda:0')*math.pi/180.
    joint_vel_upper_limit = torch.tensor([10., 10., 10., 10., 15., 15.], device='cuda:0')

    # Load keypoint detector
    cam_pipeline, cam_align, keypoint_detector = load_keypoint_detector(checkpoint_dir=parsed_args.keypoint)

    # Load policy
    model = load_policy(parsed_args.policy)

    # Initialize action and observation scale
    action_scale, observation_scale = initialize_action_and_observation_scale(controller_type, 
                                                                              joint_pos_upper_limit, 
                                                                              joint_pos_lower_limit, 
                                                                              joint_vel_upper_limit)
    

    # Initialize goal pose
    if TASK == 'Bump':
        # object_goal_pose = torch.tensor([0.25,  0.29,  0.5515, 0.0, 0.0, 0.0, 1.0], device='cuda:0') # bump
        object_goal_pose = torch.tensor([0.25,  0.0,  0.5515, 0, 0, 0, 1], device='cuda:0') # bump 

        # object_goal_pose = torch.tensor([0.3,  0.3,  0.5515, 0.0, 0.0, 0.0, 1.0], device='cuda:0') # bump
        # object_init_pose = torch.tensor([0.3,  0.0,  0.5515, 0.0, 0.0, 0.0, 1.0], device='cuda:0') 


    # object_goal_pose = torch.tensor([0.4, 0.35, 0.509, 0.0, 0.0, 0.0, 1.0], device='cuda:0')
    goal_keypoints_2d = project_target_to_camera_coord(object_goal_pose, get_target_size(parsed_args.task))
    # init_keypoints_2d = project_target_to_camera_coord(object_init_pose, get_target_size(parsed_args.task))

    base_posquat = np.array([0., 0., 0., 0., 0., 0., 1.])

    prev_action = torch.zeros((1, 18), dtype=torch.float, device='cuda:0')

    im2client = IM2Client(ip=parsed_args.ip, port=parsed_args.port)

    # init_joint_mapping = PreCalculatedIK()
    # init_joint_pos = init_joint_mapping.pre_calculated_IK(object_initial_pose[:3][None])[0].clone().detach().cpu().numpy()
    # print("pre-calculated IK solution: ", init_joint_pos)

    cur_joint_pos, _, _, _ = im2client.get_current_states(arm)
    # contact_traj = interpolate_trajectory(cur_joint_pos, init_joint_pos, 1, 1/FPS)
    # if not move(arm, im2client, contact_traj, kps, kds, fps=FPS):
    #     raise ValueError('Failed to move according to contact policy')

    print("current_joint_position: ", cur_joint_pos)

    params = {'im2client': im2client,
              'task': parsed_args.task,
              'arm': arm,
              'policy_freq': policy_freq,
              'keypoint_detector': keypoint_detector, 
              'cam_pipeline': cam_pipeline,
              'cam_align': cam_align,
              'goal_keypoints_2d': goal_keypoints_2d,
              'base_posquat': base_posquat,
              'action_scale': action_scale,
              'observation_scale': observation_scale,
              'model': model,
              'controller_type': controller_type,
              'joint_pos_upper_limit': joint_pos_upper_limit,
              'joint_pos_lower_limit': joint_pos_lower_limit,
              'prev_action': prev_action,
              'save_dir': SAVE_DIR,
              'data_dumper': dump_data
    }

    # kps = np.array([1.8, 2.2, 1.6, 1.6, 0.7, 0.7])
    # kds = np.array([0.06, 0.08, 0.1, 0.1, 0.04, 0.04])
    # im2client.joint_pd_control('right', np.array([0, 0, 0, 0, 0, 0]), p_gains=kps, d_gains=kds)
    execute(**params)


if __name__ == '__main__':
    main()


