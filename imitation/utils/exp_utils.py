#%% Import libraries and initialize global variable
import os, sys
from pathlib import Path
BASEDIR = Path(__file__).parent.parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))


from numpy import typing as npt
from typing import List, Tuple, Callable, Union, Dict
import time, datetime
import numpy as np
import pickle
import torch
import argparse

# Real robot
from control.im2controller_client import IM2Client
from control import DefaultControllerValues as RobotParams

# Manipulation
from simulation.bullet.manipulation import ShadowingManipulation

# Camera
import pyrealsense2 as rs

# Perceptions
PERCEPTION_DIR = BASEDIR / 'imitation' / 'perception'
sys.path.insert(0, str(PERCEPTION_DIR / 'WHAM'))
sys.path.insert(0, str(PERCEPTION_DIR / 'hamer'))
sys.path.insert(0, str(PERCEPTION_DIR / 'pose'))
from yacs.config import CfgNode as CN
from imitation.rt_cfg import RuntimeConfig
from imitation.perception.pose.utils.multi_rs_camera import MultiRSCamera, CameraConfig
from imitation.perception.pose.multicam_april_tag_tracker import MulticamAprilTagTracker
from imitation.perception.WHAM.demo_live import LiveWHAMPipeline, compute_cam_intrinsics
from imitation.perception.hamer.demo_live import LiveHAMERPipeline, CACHE_DIR_HAMER, DEFAULT_CHECKPOINT
from imitation.perception.pose.perception import Perception


POLICY_FPS = 20
IP = '[::]'
PORT = '50051'
URDF_DIR = 'simulation/assets/urdf/RobotBimanualV5/urdf/Simplify_Robot_plate_gripper.urdf'

# Camera
HUMAN_CAM_SERIAL = '032622072662'

def get_cfg_defaults():
     # Configuration variable
    cfg = CN()

    cfg.TITLE = 'default'
    cfg.OUTPUT_DIR = 'results'
    cfg.EXP_NAME = 'default'
    cfg.DEVICE = 'cuda'
    cfg.DEBUG = False
    cfg.EVAL = True
    cfg.RESUME = False
    cfg.LOGDIR = ''
    cfg.NUM_WORKERS = 5
    cfg.SEED_VALUE = -1
    cfg.SUMMARY_ITER = 50
    cfg.MODEL_CONFIG = ''
    cfg.FLIP_EVAL = False
    cfg.MIN_FRAMES = 4

    cfg.TRAIN = CN()
    cfg.TRAIN.STAGE = 'stage1'
    cfg.TRAIN.DATASET_EVAL = '3dpw'
    cfg.TRAIN.CHECKPOINT = ''
    cfg.TRAIN.BATCH_SIZE = 64
    cfg.TRAIN.START_EPOCH = 0
    cfg.TRAIN.END_EPOCH = 999
    cfg.TRAIN.OPTIM = 'Adam'
    cfg.TRAIN.LR = 3e-4
    cfg.TRAIN.LR_FINETUNE = 5e-5
    cfg.TRAIN.LR_PATIENCE = 5
    cfg.TRAIN.LR_DECAY_RATIO = 0.1
    cfg.TRAIN.WD = 0.0
    cfg.TRAIN.MOMENTUM = 0.9
    cfg.TRAIN.MILESTONES = [50, 70]

    cfg.DATASET = CN()
    cfg.DATASET.SEQLEN = 81
    cfg.DATASET.RATIO = [1.0, 0, 0, 0, 0]

    cfg.MODEL = CN()
    cfg.MODEL.BACKBONE = 'vit'

    cfg.LOSS = CN()
    cfg.LOSS.SHAPE_LOSS_WEIGHT = 0.001
    cfg.LOSS.JOINT2D_LOSS_WEIGHT = 5.
    cfg.LOSS.JOINT3D_LOSS_WEIGHT = 5.
    cfg.LOSS.VERTS3D_LOSS_WEIGHT = 1.
    cfg.LOSS.POSE_LOSS_WEIGHT = 1.
    cfg.LOSS.CASCADED_LOSS_WEIGHT = 0.0
    cfg.LOSS.CONTACT_LOSS_WEIGHT = 0.04
    cfg.LOSS.ROOT_VEL_LOSS_WEIGHT = 0.001
    cfg.LOSS.ROOT_POSE_LOSS_WEIGHT = 0.4
    cfg.LOSS.SLIDING_LOSS_WEIGHT = 0.5
    cfg.LOSS.CAMERA_LOSS_WEIGHT = 0.04
    cfg.LOSS.LOSS_WEIGHT = 60.
    cfg.LOSS.CAMERA_LOSS_SKIP_EPOCH = 5

    return cfg


def make_empty_data(save_dir) -> Dict[str, Union[str, List]]:
   
    os.makedirs(save_dir, exist_ok=True)
    time_list = []
    des_pos_list = []
    joint_pos_list = []
    joint_vel_list = []
    joint_tau_list = []
    des_tau_list = []
    t_policy_list = []
    act_list = []

    empty_data = {'filename': None,
                  'current_time': None,
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
    
    return empty_data


# Dump experiment data
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


def keyboard_prompt(prompt: str):

    key = input(prompt)

    if len(key) > 1:
        key = key[0]

    return key


def get_params(ip: str=IP, port: str=PORT):
    parser = argparse.ArgumentParser(description='Run real robot experiment')
    parser.add_argument('--save_dir', default=None, help='Data saving directory')
    parser.add_argument('--ip', type=str, default=ip)
    parser.add_argument('--port', type=str, default=port)
    parser.add_argument('--calibrate_gripper', action='store_true', default=False, help='calibrate gipper close threshold')
    parser.add_argument('--no_robot', action='store_true', default=False, help='no robot mode for debugging')

    return parser.parse_args()



def load_multi_rs_camera(rt_cfg: RuntimeConfig) -> MultiRSCamera:
    cam_cfg = [CameraConfig(device_id=cam_id,
                            img_height=480,
                            img_width=640,
                            exposure=None,
                            fps=60)
                            for cam_id in rt_cfg.cam_ids]
    
    multi_cfg = MultiRSCamera.Config(cams=cam_cfg)
    

    return MultiRSCamera(multi_cfg, start=True)


def load_body_cam(width: int=640, height: int=480, cam_fps: int=30, **kwargs):
    # BodyCamera
    cam_serial = kwargs.get('body_cam_serial', HUMAN_CAM_SERIAL)
    body_cam = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(cam_serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, cam_fps)
    body_cam.start(cfg)

    return body_cam


def build_body_tracker(**kwargs) -> Tuple[rs.pyrealsense2.pipeline,
                                     torch.nn.Module, 
                                     torch.nn.Module]:
    
    # Body camera
    body_cam = load_body_cam(**kwargs)
    width = kwargs.get('width', 640)
    height = kwargs.get('height', 480)
    
    # Models
    device = kwargs.get('device', 'cuda:0')
    res = torch.tensor([width, height]).float()
    intrinsics = compute_cam_intrinsics(res)

    # Configurations for perception algorithms
    wham_cfg = get_cfg_defaults()
    wham_cfg.merge_from_file('imitation/perception/WHAM/configs/yamls/demo.yaml')

    body_pose_tracker = LiveWHAMPipeline(LiveWHAMPipeline.Config(cfg_node=wham_cfg,
                                                                 calib=None,
                                                                 fps=kwargs.get('cam_fps', 30),
                                                                 run_smplify=False,
                                                                 flip=False,
                                                                 min_frames=wham_cfg.MIN_FRAMES),
                                                                 intrinsics,
                                                                 device)
    
    hand_pose_tracker = None
    # hand_pose_tracker = LiveHAMERPipeline(LiveHAMERPipeline.Config(cache_dir=CACHE_DIR_HAMER,
    #                                                                checkpoint=DEFAULT_CHECKPOINT,
    #                                                                body_detector='regnety',
    #                                                                rescale_factor=2.0,
    #                                                                batch_size=1))
    

    
    
    return body_cam, body_pose_tracker, hand_pose_tracker


def build_obj_tracker(**kwargs):
    # Runtime configuration and object camera 
    rt_cfg = RuntimeConfig()
    obj_cam = load_multi_rs_camera(rt_cfg)
    # Wait for camera to turn on
    time.sleep(0.5)
    with open(rt_cfg.extrinsics_file, 'rb') as fp:
        extrinsics = pickle.load(fp)
    T_ext = np.stack([extrinsics[d] for d in obj_cam.devices], axis=0).astype(np.float32)
    
    # Object tracker
    object_tracker = Perception(Perception.Config(pose_type='tag',
                                                  tag=MulticamAprilTagTracker.Config(cal_file=kwargs.get('cal_file', None)),
                                                  skip_seg=True,
                                                  device='cuda:0'))
    object_tracker.setup(obj_cam, T_ext)

    return obj_cam, object_tracker


def build_manip(urdf_dir=URDF_DIR, **kwargs) -> ShadowingManipulation:
        joint_pos_lower_limit = np.concatenate((RobotParams.L_JOINT_LIMIT_MIN, RobotParams.GRIPPER_SIM_MIN, RobotParams.R_JOINT_LIMIT_MIN, RobotParams.GRIPPER_SIM_MIN))
        joint_pos_upper_limit = np.concatenate((RobotParams.L_JOINT_LIMIT_MAX, RobotParams.GRIPPER_SIM_MAX, RobotParams.R_JOINT_LIMIT_MAX, RobotParams.GRIPPER_SIM_MAX))
        joint_vel_upper_limit = RobotParams.JOINT_VEL_UPPER_LIMIT

        robot_name = urdf_dir
        
        start_pos = [0, 0, 0]
        start_orn = [0, 0, 0]
        
        manip = ShadowingManipulation(start_pos, 
                                      start_orn,
                                      robot_name=robot_name, 
                                      joint_min=joint_pos_lower_limit, 
                                      joint_max=joint_pos_upper_limit, 
                                      joint_vel_upper_limit=joint_vel_upper_limit,
                                      debug=False,
                                      **kwargs)
        
        return manip

