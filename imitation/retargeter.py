#%% Import libraries and initialize global variable
import os, sys
from pathlib import Path
BASEDIR = Path(__file__).parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

from numpy import typing as npt
from typing import List, Tuple, Callable, Dict, Iterable, Deque
import numpy as np
import cv2
from collections import deque

# Control
from control import DefaultControllerValues as RobotParams

# Utils
from imitation.utils.exp_utils import *
from imitation.utils.mano_utils import *
from imitation.utils.smpl_utils import *
from imitation.utils.pink_utils import *
from imitation.utils.transforms import *

# Inference
import threading

# Log
from loguru import logger

DOF = RobotParams.DOF
FPS = 30

IP = '[::]'
PORT = '50051'

VIZ = False

# @ray.remote
def model_inference(pipe: Callable, x: torch.Tensor, output_list: Union[List, Dict], index: int):
    output_list[index] = pipe(x)
    # return pipe(x)


class Retargeter():
    def __init__(self, 
                 grasp_threshold:float=0.025,
                 **kwargs):
        
        self.device: str = kwargs.get('device', 'cuda:0')
        self.manip: ShadowingManipulation = build_manip(**kwargs)
        body_cam, wham, hamer = build_body_tracker(**kwargs)
        self.ik_solver: OptBasedIK = build_ik(**kwargs)

        self.body_pose_tracker: LiveWHAMPipeline  = wham
        self.hand_pose_tracker: LiveHAMERPipeline = hamer

        self.body_camera: rs.pyrealsense2.pipeline = body_cam

        # Inference settings
        self.min_frames: int = self.body_pose_tracker.cfg.min_frames
        self.tracker_initialized: bool = False      # WHAM outputs body pose after 4 frames

        # Grasp
        self.grasp_threshold = grasp_threshold

        # Placeholder for current pose readings
        self.curr_body_pose: Dict[str, torch.Tensor] = None
        self.curr_right_hand_pose: Dict[str, torch.Tensor] = None
        self.curr_left_hand_pose: Dict[str, torch.Tensor] = None

        # Initialize buffer for past quaternions
        self.prev_quaternions: Dict[str, Deque[torch.Tensor]] = dict()
        self.quaternion_window_size = 50

        # Initialize tracker
        for i in range(self.min_frames):
            img = self.read_image()
            body_pose, hand_pose = self.get_human_poses(img)

        # if 'pose' in body_pose and len(hand_pose) > 1:
        if 'pose' in body_pose:
            self.tracker_initialized = True


    def read_image(self, show: bool=False):
        '''
        Read image from camera 
        '''

        frame = self.body_camera.wait_for_frames()
        color_frame = frame.get_color_frame()

        # If no frame is available, error should have occured at this point
        flag = True
        color_image = np.asanyarray(color_frame.get_data())

        if show:
            cv2.imshow('rgb', color_image)
            if cv2.waitKey(1) == ord('q'):
                raise KeyboardInterrupt

        return color_image


    def get_human_poses(self, image: npt.NDArray):
        # pipes = [self.body_pose_tracker, self.hand_pose_tracker]
        pipes = [self.body_pose_tracker]

        ## Multithreading
        threads=[]
        outputs = [None]*len(pipes)
        for i, pipe in enumerate(pipes):
            t = threading.Thread(target=model_inference, args=(pipe, image, outputs, i))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        body_pose = outputs[0]
        # hand_pose = outputs[1]


        # if hand_pose is None or len(hand_pose) < 2:
        #     raise NoHandPredictionException

        hand_pose = None
        
        return body_pose[0], hand_pose
    

    def extract_shadowed_arm_joint(self, body_pose: List[Dict[str, torch.Tensor]]=None) -> Dict[str, torch.Tensor]:
        self.curr_body_pose = body_pose
        smpl_pose_traj = body_pose['pose']

        # Pelvis (base)
        pelvis_traj = extract_pelvis_traj(smpl_pose_traj)

        # Shoulder
        left_shoulder_traj, left_shoulder_local_traj, left_base_traj = extract_shoulder_traj(smpl_pose_traj, 'left')
        right_shoulder_traj, right_shoulder_local_traj, right_base_traj = extract_shoulder_traj(smpl_pose_traj, 'right')

        # Elbow
        left_elbow_traj, left_elbow_local_traj, _ = extract_elbow_traj(smpl_pose_traj, 'left', left_shoulder_traj)
        right_elbow_traj, right_elbow_local_traj, _ = extract_elbow_traj(smpl_pose_traj, 'right', right_shoulder_traj)

        # Wrist
        left_wrist_traj, left_wrist_local_traj, _ = extract_wrist_traj(smpl_pose_traj, 'left', left_elbow_traj)
        right_wrist_traj, right_wrist_local_traj, _ = extract_wrist_traj(smpl_pose_traj, 'right', right_elbow_traj)

        ret = {
            'pelvis_traj': pelvis_traj,
            'left_shoulder_traj': left_shoulder_traj,
            'right_shoulder_traj': right_shoulder_traj,
            'left_elbow_traj': left_elbow_local_traj,
            'right_elbow_traj': right_elbow_local_traj,
            'left_elbow_global_traj': left_elbow_traj,
            'right_elbow_global_traj': right_elbow_traj,
            'left_wrist_traj': left_wrist_local_traj,
            'right_wrist_traj': right_wrist_local_traj,
            'left_wrist_global_traj': left_wrist_traj,
            'right_wrist_global_traj': right_wrist_traj,
        }

        return ret
    

    def extract_shadowed_hand_joint(self, hand_pose: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for res in hand_pose:
            if res['right']:
                self.curr_right_hand_pose = res
            else:
                self.curr_left_hand_pose = res


        left_hand_global_traj = extract_hand_global_traj(self.curr_left_hand_pose['global_orient'])
        right_hand_global_traj = extract_hand_global_traj(self.curr_right_hand_pose['global_orient'])

        ret = {
            'right_hand_global_traj': right_hand_global_traj,
            'right_hand_keypoints_3d': self.curr_right_hand_pose['pred_keypoints_3d'],
            'left_hand_global_traj': left_hand_global_traj,
            'left_hand_keypoints_3d': self.curr_left_hand_pose['pred_keypoints_3d'],
        }

        return ret
    

    def retarget_elbow(self, 
                       robot_joint_angles: torch.Tensor, 
                       arm_joint_angles: Dict[str, torch.Tensor]) -> torch.Tensor:
       
        left_elbow_rotation_matrix = quaternion_to_matrix(arm_joint_angles['left_elbow_traj'])
        right_elbow_rotation_matrix = quaternion_to_matrix(arm_joint_angles['right_elbow_traj'])

        left_elbow_traj = extract_principal_rotation(left_elbow_rotation_matrix, axis='z', device=self.device)
        right_elbow_traj = extract_principal_rotation(right_elbow_rotation_matrix, axis='z', device=self.device)

        robot_joint_angles[:, 3] = -left_elbow_traj
        robot_joint_angles[:, RobotParams.DOF+3] = right_elbow_traj

        robot_joint_angles[:, 3] += RobotParams.L_JOINT_SMPL_OFFSET[3]
        robot_joint_angles[:, RobotParams.DOF+3] += RobotParams.R_JOINT_SMPL_OFFSET[3]

        return robot_joint_angles
            
    
    def retarget_shoulder(self, 
                          robot_joint_angles: torch.Tensor, 
                          arm_joint_angles: Dict[str, torch.Tensor]) -> torch.Tensor:
        offset = torch.tensor(np.concatenate((RobotParams.L_JOINT_SMPL_OFFSET, RobotParams.R_JOINT_SMPL_OFFSET)), device=self.device)

        l_theta, l_a = extract_rotation_from_quaternion(arm_joint_angles['left_shoulder_traj'])
        r_theta, r_a = extract_rotation_from_quaternion(arm_joint_angles['right_shoulder_traj'])

        l_quat_traj = l_theta.view(-1,1)*l_a
        r_quat_traj = r_theta.view(-1,1)*r_a

        # logger.info(f'right joint angles: {r_quat_traj*180/torch.pi}')
        # logger.info(f'left joint angles: {l_quat_traj*180/torch.pi}')

        # Adapt
        robot_joint_angles[:, 0] = -l_quat_traj[:,1]
        robot_joint_angles[:, 1] = l_quat_traj[:,2]
        robot_joint_angles[:, 2] = l_quat_traj[:,0]

        robot_joint_angles[:, RobotParams.DOF] = -r_quat_traj[:,1]
        robot_joint_angles[:, RobotParams.DOF+1] = -r_quat_traj[:,2]
        robot_joint_angles[:, RobotParams.DOF+2] = -r_quat_traj[:,0]

        robot_joint_angles[:, :3] += offset[:3]
        robot_joint_angles[:, RobotParams.DOF:RobotParams.DOF+3] += offset[RobotParams.DOF:RobotParams.DOF+3]

        robot_joint_angles[:, RobotParams.DOF]*=-1
        robot_joint_angles[:, RobotParams.DOF+2]*=-1
        robot_joint_angles[:, 0:3]*=-1


        return robot_joint_angles
    

    def retarget_hand(self, 
                      robot_joint_angles: torch.Tensor, 
                      arm_joint_angles: Dict[str, torch.Tensor], 
                      hand_joint_angles: Dict[str, torch.Tensor]) -> torch.Tensor:
        
        

        # Compute base rotations.
        cam_traj = arm_joint_angles['pelvis_traj']
        l_wrist_traj = mirror_quaternion(arm_joint_angles['left_wrist_global_traj'], 'yz')
        r_wrist_traj = arm_joint_angles['right_wrist_global_traj']
        l_hand_traj = hand_joint_angles['left_hand_global_traj']
        r_hand_traj = hand_joint_angles['right_hand_global_traj']

        l_hand_traj_pelvis = quaternion_multiply(l_hand_traj, quaternion_invert(cam_traj)).squeeze(0)
        r_hand_traj_pelvis = quaternion_multiply(r_hand_traj, quaternion_invert(cam_traj)).squeeze(0)

        l_hand_local_traj = quaternion_multiply(l_hand_traj_pelvis, quaternion_invert(l_wrist_traj))
        r_hand_local_traj = quaternion_multiply(r_hand_traj_pelvis, quaternion_invert(r_wrist_traj))

        # Pad past quaternions for sign consistency
        quats = {'l_hand_local_traj': l_hand_local_traj, 
                 'r_hand_local_traj': r_hand_local_traj}
        padded_quaternions = self.pad_quaternions(quats)

        l_hand_local_quat_traj = ensure_quaternion_consistency(padded_quaternions['l_hand_local_traj'], device=self.device)[-1].unsqueeze(0)
        r_hand_local_quat_traj = ensure_quaternion_consistency(padded_quaternions['r_hand_local_traj'], device=self.device)[-1].unsqueeze(0)

        # l_hand_local_quat_traj = l_hand_local_traj if l_hand_local_traj[0, 0] > 0 else -l_hand_local_traj
        # r_hand_local_quat_traj = r_hand_local_traj if r_hand_local_traj[0, 0] > 0 else -r_hand_local_traj

        # l_quat_traj = quaternion_to_axis_angle(l_hand_local_quat_traj)
        # r_quat_traj = quaternion_to_axis_angle(r_hand_local_quat_traj)

        quats = {'l_hand_local_traj': l_hand_local_quat_traj, 
                 'r_hand_local_traj': r_hand_local_quat_traj}
        
        aas, quats = self.sign_corrected_quat2aa(quats)
        # sign = '+' if quats['r_hand_local_traj'][0, 0] > 0 else '-'
        # logger.info(f'Sign of right hand quat = {sign}')

        l_traj = aas['l_hand_local_traj']
        r_traj = aas['r_hand_local_traj']

        self.store_quaternions(quats)

        # logger.info(f'[Retargeter] WHAM prediction: {quaternion_to_axis_angle(l_wrist_traj_pre)*180/np.pi}')

        # Adapt
        robot_joint_angles[:, 4] = l_traj[:,0]
        robot_joint_angles[:, 5] = RobotParams.L_JOINT_SMPL_OFFSET[5]
        robot_joint_angles[:, RobotParams.DOF+4] = -r_traj[:,0]
        robot_joint_angles[:, RobotParams.DOF+5] = RobotParams.R_JOINT_SMPL_OFFSET[5]

        return robot_joint_angles
    


    def offset_hand(self, robot_joint_angles: torch.Tensor) -> torch.Tensor:
        robot_joint_angles[:, 5] = RobotParams.L_JOINT_SMPL_OFFSET[5]
        robot_joint_angles[:, RobotParams.DOF+5] = RobotParams.R_JOINT_SMPL_OFFSET[5]

        return robot_joint_angles
    

    def sign_corrected_quat2aa(self, data: Dict[str, torch.Tensor], index: int=0, threshold: float=60):
        aa = dict()
        quat = dict()
        for k, v in data.items():
            prev_aa = quaternion_to_axis_angle(self.prev_quaternions[k][-1])
            curr_aa = quaternion_to_axis_angle(v)

            if torch.abs(prev_aa-curr_aa)[0, index] > threshold/180*torch.pi:
                aa[k] = quaternion_to_axis_angle(-v)
                quat[k] = -v.clone()
            else:
                aa[k] = curr_aa
                quat[k] = v.clone()

        return aa, quat

    


    def store_quaternions(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for k, v in data.items():
            self.prev_quaternions[k].append(v)
    

    def pad_quaternions(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        ret = dict()
        for k, v in data.items():
            if k not in self.prev_quaternions:
                self.prev_quaternions[k] = deque(maxlen=self.quaternion_window_size)
                for _ in range(self.quaternion_window_size):
                    self.prev_quaternions[k].append(v)

            ret[k] = torch.cat((*self.prev_quaternions[k], v))

        return ret
    
    def extract_grasp(self, threshold: float=0.025):

        right_keypoints = self.curr_right_hand_pose['pred_keypoints_3d']
        right_grasp = extract_grasp_from_keypoints(right_keypoints, threshold=threshold)

        left_keypoints = self.curr_left_hand_pose['pred_keypoints_3d']
        left_grasp = extract_grasp_from_keypoints(left_keypoints, threshold=threshold)


        return torch.cat((left_grasp, right_grasp))
    


    def find_finger_distances(self) -> torch.Tensor:
        left_keypoints = self.curr_left_hand_pose['pred_keypoints_3d']
        right_keypoints = self.curr_right_hand_pose['pred_keypoints_3d']
        left_finger_dist = extract_finger_dist_from_keypoints(left_keypoints)
        right_finger_dist = extract_finger_dist_from_keypoints(right_keypoints)

        return torch.cat((left_finger_dist, right_finger_dist))


    def calibrate_gripper(self) -> float:

        # Prompt
        logger.info('Start calibrating. Open your finger grippers and hold for a second')
        time.sleep(5)

        assert self.tracker_initialized, 'Body trackers must stream ouptut before retargeting. At least 4 frames must be past.'
        while True:      
            try:
                # Perception
                image = self.read_image(show=False)
                _, hand_pose = self.get_human_poses(image)
                self.extract_shadowed_hand_joint(hand_pose)
                break
            except NoHandPredictionException:
                logger.warning('We are missing pose for at least one hand. We will use the last prediction instead.')
                continue

        # Compute finger distance. Minimum open finger distance will be used.
        open_dist = self.find_finger_distances().min()

        # Prompt
        logger.info('Now close your finger grippers and hold for a second')
        time.sleep(5)

        while True:      
            try:
                # Perception
                image = self.read_image(show=False)
                _, hand_pose = self.get_human_poses(image)
                self.extract_shadowed_hand_joint(hand_pose)
                break
            except NoHandPredictionException:
                logger.warning('We are missing pose for at least one hand. We will use the last prediction instead.')
                continue

        # Compute finger distance. Maximum close finger distance will be used.
        close_dist = self.find_finger_distances().max()

        threshold = float((open_dist + close_dist)/2)

        self.grasp_threshold = threshold
        logger.info(f'Gripper is calibrated. New threshold is {threshold}')

        return threshold
        

    def shadow(self, **kwargs):
        '''
        Read image. Run perception, then project joint angle to actuators.
        '''

        # Initialize robot joint target
        robot_joint_angles = torch.zeros((1, RobotParams.DOF*2), device=self.device)

        # Read image from camera
        image = self.read_image(show=False)

        # Run body pose predictors
        assert self.tracker_initialized, 'Body trackers must stream ouptut before retargeting. At least 4 frames must be past.'
        try:
            body_pose, hand_pose = self.get_human_poses(image)
        except NoHandPredictionException:
            logger.warning('We are missing pose for at least one hand. We will use the last prediction instead.')

        # from matplotlib import pyplot as plt
        # arr = self.body_pose_tracker.network.pred_kp3d.detach().cpu().numpy().squeeze()
        # arr = arr.squeeze()
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(arr[-1,:,0], arr[-1,:,1], arr[-1,:,2])
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.show()

        # Extract values and retarget
        arm_joint_angles: Dict[str, torch.Tensor] = self.extract_shadowed_arm_joint(body_pose)
        # hand_joint_angles: Dict[str, torch.Tensor] = self.extract_shadowed_hand_joint(hand_pose)
        # grasp = self.extract_grasp(threshold=self.grasp_threshold)
        self.retarget_shoulder(robot_joint_angles, arm_joint_angles)
        self.retarget_elbow(robot_joint_angles, arm_joint_angles)
        # self.retarget_hand(robot_joint_angles, arm_joint_angles, hand_joint_angles)
        self.offset_hand(robot_joint_angles)

        # return robot_joint_angles.squeeze().detach().cpu().numpy(), grasp.detach().cpu().numpy()
            
        return robot_joint_angles.squeeze().detach().cpu().numpy(), None
    

    def optimize_ik(self, robot_joint_angles: torch.Tensor):
        # Optmization IK
        q = robot_joint_angles.squeeze().detach().cpu().numpy()
        self.manip.robot.setJointStates('all', q)
        left_elbow_trans, _ = self.manip.robot.get_link_state('link5')
        left_wrist_trans, _ = self.manip.robot.get_link_state('link6')
        right_elbow_trans, _ = self.manip.robot.get_link_state('link11')
        right_wrist_trans, _ = self.manip.robot.get_link_state('link12')

        left_elbow_target = IKtarget('left_elbow', translation=left_elbow_trans)
        left_wrist_target = IKtarget('left_wrist', translation=left_wrist_trans)
        right_elbow_target = IKtarget('right_elbow', translation=right_elbow_trans)
        right_wrist_target = IKtarget('right_wrist', translation=right_wrist_trans)

        opt_q = self.ik_solver.ik([left_elbow_target,
                                   left_wrist_target,
                                   right_elbow_target,
                                   right_wrist_target])
        
        return torch.tensor(opt_q, device=self.device).unsqueeze(0)

    def shadow2(self, **kwargs):
        '''
        Read image. Run perception, then project joint angle to actuators.
        '''

        # Initialize robot joint target
        robot_joint_angles = torch.zeros((1, RobotParams.DOF*2), device=self.device)

        # Read image from camera
        image = self.read_image(show=False)

        # Run body pose predictors
        assert self.tracker_initialized, 'Body trackers must stream ouptut before retargeting. At least 4 frames must be past.'
        try:
            body_pose, hand_pose = self.get_human_poses(image)
        except NoHandPredictionException:
            logger.warning('We are missing pose for at least one hand. We will use the last prediction instead.')

        arm_joint_angles: Dict[str, torch.Tensor] = self.extract_shadowed_arm_joint(body_pose)
        # hand_joint_angles: Dict[str, torch.Tensor] = self.extract_shadowed_hand_joint(hand_pose)
        # grasp = self.extract_grasp(threshold=self.grasp_threshold)


        # JH: I think problem is still here: we still directly using desired joint values from different morphology
        #     I think the better way to solve this is to 
        # 1. extract the hand pose w.r.t base of SMPL given its whole joint config
        # 2. Solve IK with that hand pose first
        # 3. Optimize IK solution with the desired corresponding joint values as the soft constraint.

        self.retarget_shoulder(robot_joint_angles, arm_joint_angles)
        self.retarget_elbow(robot_joint_angles, arm_joint_angles)

        robot_joint_angles = self.optimize_ik(robot_joint_angles)

        self.offset_hand(robot_joint_angles)
            
        return robot_joint_angles.squeeze().detach().cpu().numpy(), None