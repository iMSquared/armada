import sys
from pathlib import Path
BASEDIR = Path(__file__).parent.parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.append(str(BASEDIR))

from typing import Tuple, List, Dict, Callable
import numpy.typing as npt
import numpy as np
import re, time, pickle, joblib
from matplotlib import pyplot as plt

from functools import partial
from simulation.bullet.bullet_sim import BulletEnvironment
from simulation.bullet.robot import LEFT, RIGHT
from simulation.bullet.utils.capture_env import *
from simulation.bullet.utils.collision import ContactBasedCollision, BimanualCollision, LinkPair
from simulation.bullet.utils.kinematics import Kinematics
from simulation.bullet.utils.common import interpolate_trajectory, interpolate_vectors, draw_axes
from simulation.bullet.motion_planners.rrt_connect import birrt
from simulation.bullet.utils.bimanual import BimanualHandPoseGenerator
from imitation.utils.smpl_utils import *
from imitation.utils.mano_utils import *
from imitation.utils.transforms import *

from control import DefaultControllerValues as RobotParams

IP = '[::]'
PORT = '50051'
FPS = 200.0
JOINT_POS_LOWER_LIMIT = np.concatenate((RobotParams.L_JOINT_LIMIT_MIN, RobotParams.GRIPPER_SIM_MIN, RobotParams.R_JOINT_LIMIT_MIN, RobotParams.GRIPPER_SIM_MIN))
JOINT_POS_UPPER_LIMIT = np.concatenate((RobotParams.L_JOINT_LIMIT_MAX, RobotParams.GRIPPER_SIM_MAX, RobotParams.R_JOINT_LIMIT_MAX, RobotParams.GRIPPER_SIM_MAX))
JOINT_VEL_UPPER_LIMIT = RobotParams.JOINT_VEL_UPPER_LIMIT
JOINT_ID_OFFSET = RobotParams.DOF_ID_OFFSET
PALETTE = {'left': (0,0,1), 'right': (1,0,0)}

class Manipulation(BulletEnvironment):
    def __init__(self,
                 start_pos: npt.ArrayLike, 
                 start_orn: npt.ArrayLike,
                 planner: str,
                 arm: str='all',
                 robot_name: str=None,
                 control_freq: float=FPS,
                 resolution: float=0.1,
                 joint_min: npt.NDArray=JOINT_POS_LOWER_LIMIT,
                 joint_max: npt.NDArray=JOINT_POS_UPPER_LIMIT,
                 joint_vel_upper_limit: npt.NDArray=JOINT_VEL_UPPER_LIMIT,
                 joint_id_offset: int=JOINT_ID_OFFSET,
                 trials: int = 5,
                 debug: bool = False,
                 pallete: Dict[str, Tuple[float]] = PALETTE, 
                 log: bool = False, 
                 **kwargs):


        self.arm = arm         
        super().__init__(start_pos, start_orn, robot_name=robot_name, arm=arm, joint_min=joint_min, joint_max=joint_max, debug=debug, log=log, gravity=True, **kwargs)
        self.bc.resetDebugVisualizerCamera(cameraDistance=1.2,
                                           cameraYaw=51.2,
                                           cameraPitch=-39.4,
                                           cameraTargetPosition=(-0.018, -0.0214, 0.44))

        self.control_freq = control_freq
        self.joint_vel_upper_limit = joint_vel_upper_limit
        self.joint_id_offset = joint_id_offset

        self.pallete = pallete

        # planner
        assert planner in ['interpolate', 'taskspace', 'rrt'], 'only support [interpolate, taskspace, rrt]'
        self.planner = planner


        # Kinematics (forward and inverse kinematics)
        self.kinematics = Kinematics(False, self.bc, self.robot)

        # Default RRT callbacks
        self.resolution = resolution
        self.tol = kwargs.get('tol', 5e-5)
        if kwargs.get('allow_selfcollision', False):
            self_pair = LinkPair(body_id_a=self.robot.uid,
                                 link_id_a=None,
                                 body_id_b=self.robot.uid,
                                 link_id_b=None)
            self.allowlist = [self_pair]

        else:
            self.allowlist = []
        distance_fn, sample_fn, extend_fn, collsion_fn = self._get_default_functions()
        self.distance_fn          = distance_fn
        self.sample_fn            = sample_fn
        self.extend_fn            = extend_fn
        self.default_collision_fn = collsion_fn

        # RRT
        self.trials = trials

    ## Initialization helpers
    def _get_joint_ids(self):
        return [v for k,v in self.robot.joint_id_dict.items() if 'joint' in k]
    

    def _get_joint_limits(self):
        return [self.robot.joint_min, self.robot.joint_max]


    ## Interpolation
    def _interpolate_in_taskspace(self,
                                  arm: str,
                                  init: npt.NDArray,
                                  term: npt.NDArray,
                                  duration: float,
                                  control_dt: float):

        ee_traj = interpolate_trajectory(init, term, duration, control_dt)
        traj = [self.kinematics.inverse(arm, ee_pose[:3], ee_pose[4:]) for ee_pose in ee_traj]
        
        if None in traj:
            raise ValueError('No IK solution among the traj')
        
        return traj


    ## Motion planning helpers
    def _get_default_functions(self) -> Tuple[Callable, Callable, Callable, Callable]:
        """Init default sample, extend, distance, collision function

        Returns:
            distance_fn
            sample_fn
            extend_fn
            collision_fn
        """
        def distance_fn(q0: np.ndarray, q1: np.ndarray):
            return np.linalg.norm(np.subtract(q1, q0))
        
        def sample_fn(arm):
            indices = np.array([0,1,2,3,4,5]) if arm == 'left' else np.array([6,7,8,9,10,11])
            return np.random.uniform(self.robot.joint_min[indices], self.robot.joint_max[indices])
        # def sample_fn(debug=False, return_xyz=False):
        #     """RRT sampling function in cartesian space"""
        #     pos = random_sample_array_from_config(self.sample_space_center, self.sample_space_half_ranges)
        #     orn_e_toppick = [-3.1416, 0, -1.57]
        #     orn_e_yaw_rot = [0, 0, random.uniform(*self.sample_space_yaw_range)]
        #     _, orn_q = self.bc.multiplyTransforms(
        #         [0, 0, 0], self.bc.getQuaternionFromEuler(orn_e_yaw_rot),
        #         [0, 0, 0], self.bc.getQuaternionFromEuler(orn_e_toppick),)
        #     _, dst = self.solve_ik_numerical(pos, orn_q)
        #     if debug:
        #         self.bc.addUserDebugPoints([pos], [(1, 0, 0)], pointSize=3)
        #     if return_xyz:
        #         return pos, dst
        #     else:
        #         return dst

        def difference_fn(q0: np.ndarray, q1: np.ndarray):
            dtheta = (q1 - q0) % (2*np.pi)
            dtheta[dtheta>np.pi] -= 2*np.pi

            return dtheta
        
        def extend_fn(q1, q2):
            dq = difference_fn(q1, q2)
            n = int(np.max(np.abs(dq[:-1]))/self.resolution)+1

            return q1 + np.linspace(0, 1, num=n)[:, None] * dq
        
        if hasattr(self, 'scene'):
            touch_pair_robot_a = LinkPair(
                body_id_a=self.robot.uid,
                link_id_a=None,
                body_id_b=self.scene['table'],
                link_id_b=None)
            touch_pair_robot_b = LinkPair(
                body_id_a=self.scene['table'],
                link_id_a=None,
                body_id_b=self.robot.uid,
                link_id_b=None)
            
            touchlist = [touch_pair_robot_a, touch_pair_robot_b]
        else:
            touchlist = []
        
        default_collision_fn = ContactBasedCollision(
            bc           = self.bc,
            robot_id     = self.robot.uid,
            joint_ids    = self._get_joint_ids(),
            attachlist   = [],
            allowlist    = self.allowlist,
            touchlist    = touchlist,
            joint_limits = [self.robot.joint_min, self.robot.joint_max],
            tol          = {self.robot.uid: self.tol},
            touch_tol    = 0.005,
            return_contact = False)
        
        return distance_fn, sample_fn, extend_fn, default_collision_fn


    def _get_attach_pair(self, grasp_uid: Union[int, List[int]]) -> List[LinkPair]:
        # Attached object will be moved together when searching the path
        attachlist = []
        if isinstance(grasp_uid, int):
            attach_pair = LinkPair(
                body_id_a=self.robot.uid,
                link_id_a=self.robot.link_id_dict[self.robot.ee],
                body_id_b=grasp_uid,
                link_id_b=-1)
            attachlist.append(attach_pair)
        
        return attachlist
    

    def _get_allow_pair(self, grasp_uid: Union[int, List[int]]) -> List[LinkPair]:
        # Allow pair is not commutative
        allowlist = []
        if isinstance(grasp_uid, int):
            allow_pair_grasp_a = LinkPair(
                body_id_a=self.robot.uid,
                link_id_a=None,
                body_id_b=grasp_uid,
                link_id_b=None)
            allow_pair_grasp_b = LinkPair(
                body_id_a=grasp_uid,
                link_id_a=None,
                body_id_b=self.robot.uid,
                link_id_b=None)
        
            allowlist.append(allow_pair_grasp_a)
            allowlist.append(allow_pair_grasp_b)
        
        return allowlist
    
        
    def _define_grasp_collision_fn(self, grasp_uid: Union[int, List[int]], 
                                         allow_uid_list: List[int], 
                                         debug=False) -> ContactBasedCollision:
        """ This function returns the collision function that allows the grasped object to
        1. collide with the robot body
        2. be attached at robot finger
        3. touch with the cabinet.
        
        Args:
            grasp_uid: This object will be move together with the end effector.
            allow_uid_list: Uids of object to allow the contact.
        """
        
        attachlist: List[LinkPair] = self._get_attach_pair(grasp_uid)
        allowlist: List[LinkPair] = self._get_allow_pair(grasp_uid)
        for uid in allow_uid_list:
            allowlist += self._get_allow_pair(uid)
        
        # Compose collision fn
        collision_fn = ContactBasedCollision(
            bc           = self.bc,
            robot_id     = self.robot.uid,
            joint_ids    = self._get_joint_ids(),
            allowlist    = allowlist,
            attachlist   = attachlist,
            touchlist = [],
            joint_limits = self._get_joint_limits(),
            tol          = {},
            touch_tol    = 0.005)   #TODO: config

        return collision_fn


    def motion_plan(self,
                    arm: str,
                    init: npt.NDArray,
                    term: npt.NDArray,
                    holding_obj_uid: Union[int, None] = None, 
                    allow_uid_list: List[int] = [],
                    duration:float=None) -> List[npt.NDArray]:
        
        control_dt = 1/self.control_freq

        if self.planner == 'interpolate':
            trajectory = interpolate_trajectory(init, term, duration, control_dt)
        elif self.planner == 'taskspace':
            trajectory = self._interpolate_in_taskspace(arm, init, term, duration, control_dt)
        elif self.planner == 'rrt':
            with dream(self.bc, self.robot):
                # Moving with grapsed object
                if holding_obj_uid is not None or len(allow_uid_list) > 0:
                    collision_fn = self._define_grasp_collision_fn(holding_obj_uid, allow_uid_list, debug=False)
                    # Get RRT path using constraints
                    trajectory = birrt(init,
                                       term,
                                       self.distance_fn,
                                       partial(self.sample_fn, arm=arm),
                                       self.extend_fn,
                                       partial(collision_fn, arm=arm),
                                       restarts=self.trials)
                # Moving without grasped object
                else:
                    # Get RRT path using default constraints
                    trajectory = birrt(init,
                                       term,
                                       self.distance_fn,
                                       partial(self.sample_fn, arm=arm),
                                       self.extend_fn,
                                       partial(self.default_collision_fn, arm=arm),
                                       restarts=self.trials)
        else:
            raise NotImplementedError('To be implemented')

        return trajectory
    

    def motion_plan_with_gains(self,
                               arm: str,
                               init: npt.NDArray,
                               term: npt.NDArray,
                               holding_obj_uid: Union[int, None] = None, 
                               allow_uid_list: List[int] = [],
                               duration:float=None,
                               approach_duration: float=None,
                               init_p_gains: npt.NDArray=None,
                               init_d_gains: npt.NDArray=None,
                               term_p_gains: npt.NDArray=None,
                               term_d_gains: npt.NDArray=None) -> Tuple[List[npt.NDArray]]:
        
        '''
        approach_duration: proportion (0~1) of trajectory dedicated for approaching. After this point, gains are increased till terminal gains value.
        '''
        trajectory = self.motion_plan(arm, init, term, holding_obj_uid, allow_uid_list, duration)
        idx = int(len(trajectory)*approach_duration)

        # compute approach gains
        approach_p_gains = np.stack([init_p_gains]*idx)
        approach_d_gains = np.stack([init_d_gains]*idx)

        # interpolate gains
        stopping_p_gains = interpolate_vectors(init_p_gains, term_p_gains, len(trajectory)-idx)
        stopping_d_gains = interpolate_vectors(init_d_gains, term_d_gains, len(trajectory)-idx)

        p_gains = np.concatenate((approach_p_gains, stopping_p_gains), axis=0)
        d_gains = np.concatenate((approach_d_gains, stopping_d_gains), axis=0)

        return trajectory, p_gains, d_gains


    def _generate_joint_traj(self, arm: str, approach: bool=True, planner: str='interpolate'):
        assert self.ee_waypoints is not None, 'Must provide waypoint files'

        # self.ee_traj contains the list of waypoints.
        waypoints = []

        if planner != 'taskspace':
            for waypoint in self.ee_waypoints:
                joint_waypoint = self.kinematics.inverse(arm, waypoint[:3], waypoint[4:])
                if joint_waypoint is None:
                    raise ValueError(f'No IK solution at {waypoint}')
                waypoints.append(np.array(joint_waypoint))
        else:
            waypoints = self.ee_waypoints

        traj = []
            
        for i in range(1, len(waypoints)):
            init = waypoints[i-1]
            term = waypoints[i]
            motion = self.motion_plan(arm, init, term, 1.5, (1.0/FPS), planner=planner)
            traj += motion

        # Compute apporach traj
        if approach:
            if planner == 'taskspace':
                init = np.concatenate(self.robot.get_ee_pose(arm))
            else:
                init = np.array(self.robot.getJointStates(arm)[0])
            term = np.array(waypoints[0])
            motion = self.motion_plan(arm, init, term, 1.5, (1.0/FPS), planner=planner)
            traj = motion+traj

        return traj

    
    def simulate_traj(self, arm: str, traj: List[npt.NDArray], draw: bool=True):
        if arm != 'all':
            prev_ee_pos, _ = self.robot.get_ee_pose(arm)
        else:
            (prev_l_ee_pos, _), (prev_r_ee_pos, _) = self.robot.get_ee_pose(arm) 

        for i, q in enumerate(traj):
            st = time.time()
            self.robot.setJointStates(arm, q)

            if arm != 'all':
                cur_ee_pos, _ = self.robot.get_ee_pose(arm)
            else:
                (cur_l_ee_pos, _), (cur_r_ee_pos, _) = self.robot.get_ee_pose(arm)
            
            # Draw
            if draw and i > 0:
                if arm != 'all':
                    self.bc.addUserDebugLine(prev_ee_pos, cur_ee_pos,
                                            lineColorRGB=self.pallete[arm],
                                            lineWidth=2)
                else:
                    self.bc.addUserDebugLine(prev_l_ee_pos, cur_l_ee_pos,
                                            lineColorRGB=self.pallete[LEFT],
                                            lineWidth=2)
                    self.bc.addUserDebugLine(prev_r_ee_pos, cur_r_ee_pos,
                                            lineColorRGB=self.pallete[RIGHT],
                                            lineWidth=2)
                    self.bc.addUserDebugLine(cur_l_ee_pos, cur_r_ee_pos, 
                                             lineColorRGB=(0,0,0),
                                             lineWidth=2,
                                             lifeTime=0.1)
                

            # Duration
            dur = time.time() - st
            if dur < (1.0/self.control_freq):
                time.sleep((1.0/self.control_freq)-dur)
            if arm != 'all':
                prev_ee_pos = cur_ee_pos
            else:
                prev_l_ee_pos = cur_l_ee_pos
                prev_r_ee_pos = cur_r_ee_pos


    def close(self):
        self.bc.disconnect()


    def load_scene(self, table_height: float=0.5065, put_cabinet: bool=False):
        if put_cabinet:
            # Table
            tableShape = (0.2, 0.4, table_height/2)
            tablePosition = (0.3, -0.4, table_height/2)
            boxColor = (np.array([170, 170, 170, 170]) / 255.0).tolist()

            tableVisualShapeId = self.bc.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=tableShape,
                rgbaColor=boxColor
            )
            tableCollisionShapeId = self.bc.createCollisionShape(
                shapeType=p.GEOM_BOX, 
                halfExtents=tableShape
            )
            tableId = self.bc.createMultiBody(
                baseMass=10,
                baseCollisionShapeIndex=tableCollisionShapeId,
                baseVisualShapeIndex=tableVisualShapeId,
                basePosition=tablePosition
            )

            # Cabinet
            cabinetURDF = "simulation/assets/urdf/cabinet/cabinet.urdf"
            cabinetPosition = (0.6, 0.2, 0)
            cabinetOrientation = self.bc.getQuaternionFromEuler((0, 0, 0))
            cabinetId = self.bc.loadURDF(cabinetURDF, cabinetPosition, cabinetOrientation, useFixedBase=1, globalScaling=10)

            objs = {'table': tableId, 'cabinet': cabinetId}
        
        else:
            objs = super().load_scene()

        return objs


class Bimanipulation(Manipulation):
    def __init__(self, 
                 start_pos: npt.ArrayLike, 
                 start_orn: npt.ArrayLike,
                 main_arm: str,
                 robot_name: str=None,
                 control_freq: float=FPS,
                 resolution: float=0.1,
                 joint_min: npt.NDArray=JOINT_POS_LOWER_LIMIT,
                 joint_max: npt.NDArray=JOINT_POS_UPPER_LIMIT,
                 joint_vel_upper_limit: npt.NDArray=JOINT_VEL_UPPER_LIMIT,
                 joint_id_offset: int=JOINT_ID_OFFSET,
                 offset_position: float=0.1,
                 offset_orientation: Tuple[float]=(0,0,0),
                 trials: int = 5,
                 debug: bool = False,
                 pallete: Dict[str, Tuple[float]] = PALETTE, 
                 log: bool = False, **kwargs):
        
        super().__init__(start_pos, 
                         start_orn, 
                         'rrt',
                         robot_name=robot_name,
                         arm='all', 
                         control_freq=control_freq, 
                         resolution=resolution, 
                         joint_min=joint_min, 
                         joint_max=joint_max, 
                         joint_vel_upper_limit=joint_vel_upper_limit, 
                         jonit_id_offset=joint_id_offset, 
                         trials=trials, 
                         debug=debug, 
                         pallete=pallete, 
                         log=log, 
                         **kwargs)
        
        assert main_arm in [LEFT, RIGHT], 'Arm should be either left or right'
        self.main_arm = main_arm
        self.sub_arm = LEFT if main_arm == RIGHT else RIGHT

        self.hand_pose_gen = BimanualHandPoseGenerator(main_arm, 
                                                       bc=self.bc,
                                                       robot=self.robot)
        
        self.offset_position = offset_position
        self.offset_orientation = offset_orientation
        
        self.default_bicollision_fn = self._define_bicollision_fn(self, None, [])
        self.synced_sample_fn = self._define_synced_sample_fn()

    
    def get_sub_joint_pose(self, main_q: npt.NDArray, ik_iter:int=500) -> npt.NDArray:
        T_mw = self.kinematics.forward(self.main_arm, main_q)
        T_sw = self.hand_pose_gen(T_mw=T_mw,
                                    offset_position=self.offset_position,
                                    offset_orientation=self.offset_orientation, 
                                    debug=False)
        sub_q = self.kinematics.inverse(self.hand_pose_gen.sub_hand, *T_sw, ik_iter)[RobotParams.DOF:]

        return sub_q
    

    def set_hand_offset_position(self, position: Union[npt.ArrayLike, float]):
        self.offset_position = position
        self.hand_pose_gen.position = position


    def set_hand_offset_orientation(self, orientation: npt.ArrayLike):
        self.offset_orientation = orientation
        self.hand_pose_gen.orientation = orientation


    def _define_synced_sample_fn(self):
        def sample_fn():
            indices = np.array(range(RobotParams.DOF)) if self.main_arm == 'left' else np.array(range(RobotParams.DOF, RobotParams.DOF*2))
            main_q = np.random.uniform(self.robot.joint_min[indices], self.robot.joint_max[indices])
            sub_q = self.get_sub_joint_pose(main_q)

            return np.concatenate((main_q, sub_q))
        
        return sample_fn
    

    def _define_bicollision_fn(self, 
                               grasp_uid: Union[int, List[int]], 
                               allow_uid_list: List[int], 
                               debug=False) -> BimanualCollision:
        """ This function returns the collision function that allows the grasped object to
        1. collide with the robot body
        2. be attached at robot finger
        3. touch with the cabinet.
        
        Args:
            grasp_uid: This object will be move together with the end effector.
            allow_uid_list: Uids of object to allow the contact.
        """
        if grasp_uid is not None:
            attachlist: List[LinkPair] = self._get_attach_pair(grasp_uid)
            allowlist: List[LinkPair] = self._get_allow_pair(grasp_uid)
        else:
            attachlist = []
            allowlist =  []

        table_id = self.scene['table'] if hasattr(self, 'scene') else None

        touch_pair_robot_a = LinkPair(
            body_id_a=self.robot.uid,
            link_id_a=None,
            body_id_b=table_id,
            link_id_b=None)
        touch_pair_robot_b = LinkPair(
            body_id_a=table_id,
            link_id_a=None,
            body_id_b=self.robot.uid,
            link_id_b=None)
        
        # Compose collision fn
        collision_fn = BimanualCollision(
            bc           = self.bc,
            robot_id     = self.robot.uid,
            joint_ids    = self._get_joint_ids(),
            main_arm     = self.main_arm,
            allowlist    = allowlist,
            attachlist   = attachlist,
            touchlist    = [touch_pair_robot_a, touch_pair_robot_b],
            joint_limits = self._get_joint_limits(),
            tol          = {},
            touch_tol    = 0.005)   #TODO: config

        return collision_fn
    

    def synced_motion_plan(self,
                           init_main: npt.NDArray,
                           term_main: npt.NDArray,
                           holding_obj_uid: Union[int, None] = None, 
                           allow_uid_list: List[int] = [],
                           duration:float=None) -> List[npt.NDArray]:
        
        init_sub = self.get_sub_joint_pose(init_main)
        term_sub = self.get_sub_joint_pose(term_main)

        init = np.concatenate((init_main, init_sub))
        term = np.concatenate((term_main, term_sub))
        
        control_dt = 1/self.control_freq

        if self.planner == 'interpolate':
            trajectory = interpolate_trajectory(init, term, duration, control_dt)
        elif self.planner == 'rrt':
            with dream(self.bc, self.robot):
                # Moving with grapsed object
                if holding_obj_uid is not None:
                    collision_fn = self._define_bicollision_fn(holding_obj_uid, allow_uid_list, debug=False)
                    # Get RRT path using constraints
                    trajectory, contacts = birrt(
                        init,
                        term,
                        self.distance_fn,
                        self.synced_sample_fn,
                        self.extend_fn,
                        collision_fn,
                        max_solutions=1,
                        restarts=self.trials)
                # Moving without grasped object
                else:
                    # Get RRT path using default constraints
                    trajectory = birrt(
                        init,
                        term,
                        self.distance_fn,
                        self.synced_sample_fn,
                        self.extend_fn,
                        self.default_bicollision_fn,
                        restarts=self.trials)
        else:
            raise NotImplementedError('To be implemented')

        return trajectory
    

    def mirrored_motion_plan(self,
                           init_main: npt.NDArray,
                           term_main: npt.NDArray,
                           duration:float=None) -> List[npt.NDArray]:
        
        
        
        control_dt = 1/self.control_freq

        if self.planner == 'interpolate':
            assert duration is not None, 'Please specify the duration for interpolation'
            main_traj = interpolate_trajectory(init_main, term_main, duration, control_dt)
        elif self.planner == 'rrt':
            main_traj = self.motion_plan(self.main_arm, init_main, term_main)
        else:
            raise NotImplementedError('To be implemented')
        
        if main_traj is None:
            return []
        
        main_traj = np.array(main_traj)
        sub_traj = self.from_left_to_right(main_traj)

        return np.concatenate((main_traj, sub_traj), axis=1)


    def from_left_to_right(self, left_q: npt.NDArray):
        
        right_q = left_q.copy()
        right_q *= -1

        return right_q



class ShadowingManipulation(Manipulation):
    def __init__(self, 
                 start_pos: npt.NDArray, 
                 start_orn: npt.NDArray, 
                 robot_name = None, 
                 arm_data_dir = None,
                 hand_data_dirs = None,
                 **kwargs):
        
        super().__init__(start_pos, 
                         start_orn, 
                         planner='rrt', 
                         arm='all', 
                         robot_name=robot_name,
                         scene_on=False,
                         **kwargs)
        
        if arm_data_dir is not None:
            self.load_arm(arm_data_dir)
        if hand_data_dirs is not None:
            self.load_hand(*hand_data_dirs)
    


    def load_arm(self, data_dir: Union[str, None]):

        if data_dir is None:
            self.arm_data = None
            return None
        
        with open(data_dir, 'rb') as f:
            data = joblib.load(f)

        self.arm_data: List[Dict[str, torch.Tensor]] = data

        return data
    

    def load_hand(self, left_data_dir: Union[str, None], right_data_dir: Union[str, None]):

        self.hand_data: Dict[str, Dict[str, List[torch.Tensor]]]  = {RobotParams.LEFT: None, RobotParams.RIGHT: None}

        for arm, data_dir in zip((RobotParams.LEFT, RobotParams.RIGHT), (left_data_dir, right_data_dir)):
            if data_dir is None:
                continue

            pickle_dir = Path(data_dir)

            output = {'hand_pose': [], 'global_orient': [], 'pred_keypoints_3d': []}

            pickle_files = list(pickle_dir.glob("*_0.pkl"))
            pickle_files.sort(key=lambda x: int(re.search(r'frame_(\d+)_0.pkl', str(x)).group(1)))

            for pickle_file in pickle_files:
                with open(pickle_file, 'rb') as f:
                    data: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]] = pickle.load(f)
                    output['global_orient'].append(data['pred_mano_params']['global_orient'][0].detach().cpu())
                    output['hand_pose'].append(data['pred_mano_params']['hand_pose'][0].detach().cpu())
                    output['pred_keypoints_3d'].append(data['pred_keypoints_3d'][0].detach().cpu())

            output['global_orient'] = torch.stack(output['global_orient']).squeeze()
            output['hand_pose'] = torch.stack(output['hand_pose']).squeeze()
            output['pred_keypoints_3d'] = torch.stack(output['pred_keypoints_3d']).squeeze()

            self.hand_data[arm] = output
        
        return self.hand_data
    

    def extract_retargeted_trajectory(self) -> List[npt.NDArray]:

        # Arm 
        shadowed_arm_trajectory = self.extract_shadowed_joint_trajectory()
        
        traj_length = shadowed_arm_trajectory['left_shoulder_traj'].shape[0]
        traj = np.zeros((traj_length, RobotParams.DOF*2))

        traj = self.retarget_shoulder(traj, shadowed_arm_trajectory)
        traj = self.retarget_elbow(traj, shadowed_arm_trajectory)
        # traj = self.retarget_wrist(traj, shadowed_joint_trajectory)

        # Hand
        if self.hand_data is None:
            return traj, None

        shadowed_hand_trajectory = self.extract_shadowed_hand_trajectory()

        traj, grasp = self.retarget_hand(traj, shadowed_arm_trajectory, shadowed_hand_trajectory)

        return traj, grasp
    

    def extract_shadowed_joint_trajectory(self, arm_data: List[Dict[str, npt.NDArray]]=None) -> Dict[str, torch.Tensor]:

        if arm_data is None:
            assert self.arm_data is not None, 'Must load SMPL joint trajectory'
            arm_data = self.arm_data

        smpl_pose_traj = arm_data[0]['pose']

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
    

    def extract_shadowed_hand_trajectory(self) -> Dict[str, torch.Tensor]:

        assert self.hand_data is not None, 'Must load MANO joint trajectory'

        left_hand_global_traj = extract_hand_global_traj(self.hand_data['left']['global_orient'])
        left_finger_traj = extract_finger_traj(self.hand_data['left']['hand_pose'])
        right_hand_global_traj = extract_hand_global_traj(self.hand_data['right']['global_orient'])
        right_finger_traj = extract_finger_traj(self.hand_data['right']['hand_pose'])

        ret = {
            'right_hand_global_traj': right_hand_global_traj,
            'right_fingers_traj': right_finger_traj,
            'right_hand_keypoints_3d': self.hand_data['right']['pred_keypoints_3d'],
            'left_hand_global_traj': left_hand_global_traj,
            'left_fingers_traj': left_finger_traj,
            'left_hand_keypoints_3d': self.hand_data['left']['pred_keypoints_3d']
        }

        return ret
    

    def retarget_elbow(self, traj: npt.NDArray, shadowed_joint_trajectory: Dict[str, torch.Tensor]) -> npt.NDArray:
       
        left_elbow_rotation_matrix = quaternion_to_matrix(shadowed_joint_trajectory['left_elbow_traj'])
        right_elbow_rotation_matrix = quaternion_to_matrix(shadowed_joint_trajectory['right_elbow_traj'])

        left_elbow_traj = self._extract_principal_rotation(left_elbow_rotation_matrix, axis='z')
        right_elbow_traj = self._extract_principal_rotation(right_elbow_rotation_matrix, axis='z')

        traj[:, 3] = left_elbow_traj
        traj[:, RobotParams.DOF+3] = right_elbow_traj

        traj[:, 3] += RobotParams.L_JOINT_SMPL_OFFSET[3]
        traj[:, 3] *= -1
        traj[:, RobotParams.DOF+3] += RobotParams.R_JOINT_SMPL_OFFSET[3]

        return traj
            
    
    def retarget_shoulder(self, traj: npt.NDArray, shadowed_joint_trajectory: Dict[str, torch.Tensor]) -> npt.NDArray:
        offset = np.concatenate((RobotParams.L_JOINT_SMPL_OFFSET, RobotParams.R_JOINT_SMPL_OFFSET))

        l_theta, l_a = self._extract_rotation_from_quaternion(shadowed_joint_trajectory['left_shoulder_traj'])
        r_theta, r_a = self._extract_rotation_from_quaternion(shadowed_joint_trajectory['right_shoulder_traj'])

        l_quat_traj = l_theta.view(-1,1)*l_a
        r_quat_traj = r_theta.view(-1,1)*r_a

        traj[:, 0] = -l_quat_traj[:,1]
        traj[:, 1] = l_quat_traj[:,2]
        traj[:, 2] = l_quat_traj[:,0]

        traj[:, RobotParams.DOF] = -r_quat_traj[:,1]
        traj[:, RobotParams.DOF+1] = -r_quat_traj[:,2]
        traj[:, RobotParams.DOF+2] = -r_quat_traj[:,0]

        # Adapt
        traj[:, :3] += offset[:3]
        traj[:, RobotParams.DOF:RobotParams.DOF+3] += offset[RobotParams.DOF:RobotParams.DOF+3]

        traj[:, RobotParams.DOF]*=-1
        traj[:, RobotParams.DOF+2]*=-1
        traj[:, 0:3]*=-1


        return traj
    

    def retarget_hand(self, 
                      traj: npt.NDArray, 
                      shadowed_arm_trajectory: Dict[str, torch.Tensor], 
                      shadowed_hand_trajectory: torch.Tensor) -> npt.NDArray:

        # Compute base rotations.
        # l_wrist_traj = shadowed_arm_trajectory['left_wrist_global_traj']
        l_wrist_traj = mirror_quaternion(shadowed_arm_trajectory['left_wrist_global_traj'], 'yz')
        r_wrist_traj = shadowed_arm_trajectory['right_wrist_global_traj']

        l_hand_traj = shadowed_hand_trajectory['left_hand_global_traj']
        # l_hand_traj = mirror_quaternion(l_hand_traj, 'yz')
        r_hand_traj = shadowed_hand_trajectory['right_hand_global_traj']
      
        l_hand_local_traj = quaternion_multiply(l_hand_traj, quaternion_invert(l_wrist_traj))
        r_hand_local_traj = quaternion_multiply(r_hand_traj, quaternion_invert(r_wrist_traj))

        # l_hand_local_quat_traj = gaussian_smooth(ensure_quaternion_consistency(l_hand_local_traj).view(1, -1, 4)).squeeze()
        l_hand_local_quat_traj = ensure_quaternion_consistency(l_hand_local_traj)
        # l_hand_local_quat_traj  = mirror_quaternion(l_hand_local_quat_traj, 'yz')
        l_theta, l_a = self._extract_rotation_from_quaternion(l_hand_local_quat_traj)
        # r_hand_local_quat_traj = gaussian_smooth(ensure_quaternion_consistency(r_hand_local_traj).view(1, -1, 4)).squeeze()
        r_hand_local_quat_traj = ensure_quaternion_consistency(r_hand_local_traj)
        r_theta, r_a = self._extract_rotation_from_quaternion(r_hand_local_quat_traj)

        l_quat_traj = l_theta.view(-1,1)*l_a
        r_quat_traj = r_theta.view(-1,1)*r_a


        traj[:, 4] = -l_quat_traj[:,0]+np.pi
        traj[:, 5] = -l_quat_traj[:,1]
        traj[:, RobotParams.DOF+4] = -r_quat_traj[:,0]+np.pi
        traj[:, RobotParams.DOF+5] = -r_quat_traj[:,1]

        l_grasp = self._extract_grasp_from_keypoints(shadowed_hand_trajectory['left_hand_keypoints_3d'], threshold=0.06)
        r_grasp = self._extract_grasp_from_keypoints(shadowed_hand_trajectory['right_hand_keypoints_3d'], threshold=0.06)
        grasp = torch.stack((l_grasp, r_grasp), axis=1)
        # grasp = r_grasp


        # Adapt
        # traj = torch.zeros((len(grasp), RobotParams.DOF*2))
        assert len(traj) == len(grasp), "The length of arm trajectory and hand trajectory must match"

        return traj, grasp.numpy()
    

    def retarget_wrist(self, traj: npt.NDArray, shadowed_joint_trajectory: Dict[str, torch.Tensor]) -> npt.NDArray:
        offset = np.concatenate((RobotParams.L_JOINT_SMPL_OFFSET, RobotParams.R_JOINT_SMPL_OFFSET))

        l_theta, l_a = self._extract_rotation_from_quaternion(shadowed_joint_trajectory['left_wrist_traj'])
        r_theta, r_a = self._extract_rotation_from_quaternion(shadowed_joint_trajectory['right_wrist_traj'])

        l_quat_traj = l_theta.view(-1,1)*l_a
        r_quat_traj = r_theta.view(-1,1)*r_a

        traj[:, 4] = l_quat_traj[:,0]
        traj[:, 5] = l_quat_traj[:,1]

        traj[:, RobotParams.DOF+4] = r_quat_traj[:,0]
        traj[:, RobotParams.DOF+5] = r_quat_traj[:,1]

        # Adapt
        traj[:, 4:RobotParams.DOF] += offset[4:RobotParams.DOF]
        traj[:, RobotParams.DOF+4:] += offset[RobotParams.DOF+4:]

        return traj


    def _extract_principal_rotation(self, joint_trajectory: torch.Tensor, axis: str) -> torch.Tensor:
        traj_length = joint_trajectory.shape[0]
        if axis.lower() == 'x':
            axis_vector = torch.tensor([1.0, 0.0, 0.0]).repeat(traj_length, 1)
        elif axis.lower() == 'y':
            axis_vector = torch.tensor([0.0, 1.0, 0.0]).repeat(traj_length, 1)
        elif axis.lower() == 'z':
            axis_vector = torch.tensor([0.0, 0.0, 1.0]).repeat(traj_length, 1)
        else:
            raise ValueError('Choose from [x, y, z]')

        rot_component = torch.bmm(joint_trajectory, axis_vector.view(-1, 3, 1))
        projected_rot = torch.bmm(rot_component.view(-1, 1, 3), axis_vector.view(-1, 3, 1))
        angle = torch.arccos(torch.clip(projected_rot, -1.0, 1.0)).squeeze()

        return angle


    def _extract_rotation_from_quaternion(self, quaternion_trajectory: torch.Tensor):
        
        # w = quaternion_trajectory[:, 0]
        # x = quaternion_trajectory[:, 1]
        # y = quaternion_trajectory[:, 2]
        # z = quaternion_trajectory[:, 3]

        theta = 2*torch.arccos(quaternion_trajectory[:, 0])

        axis_norm = torch.linalg.norm(quaternion_trajectory[:, 1:], axis=1)
        theta[axis_norm<1e-6] = 0.0

        a = (1/torch.sqrt(1 - quaternion_trajectory[:, 0]**2)).view(-1, 1)*quaternion_trajectory[:, 1:]

        return theta, a
    

    def _extract_grasp_from_keypoints(self, hand_keypoints_3d: torch.Tensor, threshold: float=0.025):
        finger_dist = hand_keypoints_3d[:, MANOIndex.J14, :] - hand_keypoints_3d[:, MANOIndex.J24, :]
        finger_dist = torch.linalg.norm(finger_dist, axis=1)

        smoothed_finger_dist = gaussian_smooth(finger_dist.view(1, -1, 1), window_size=30, sigma=3.0).squeeze()

        grasp = smoothed_finger_dist < threshold
        
        return grasp
    

    def draw_quat(self, quaternion: torch.Tensor):
        theta, a = self._extract_rotation_from_quaternion(quaternion)

        # l_quat_traj = l_theta.view(-1,1)*l_a
        quat_traj = theta.view(-1,1)*a

        plt.plot(quat_traj[:,0]*180/np.pi, label='x')
        plt.plot(quat_traj[:,1]*180/np.pi, label='y')
        plt.plot(quat_traj[:,2]*180/np.pi, label='z')
        plt.legend()

        plt.show()

    def draw_axis_angle(self, aa: torch.Tensor):

        plt.plot(aa[:,0]*180/np.pi, label='x')
        plt.plot(aa[:,1]*180/np.pi, label='y')
        plt.plot(aa[:,2]*180/np.pi, label='z')
        plt.legend()

        plt.show()



def main():
    start_pos = [0, 0, 0]
    start_orn = [0, 0, 0]

    ### Single arm
    
    # manip = Manipulation(URDF_NAME, 
    #                      start_pos, 
    #                      start_orn, 
    #                      'rrt', 
    #                      arm=LEFT, 
    #                      control_freq=FPS,
    #                      resolution=0.01, 
    #                      joint_min=JOINT_POS_LOWER_LIMIT, 
    #                      joint_max=JOINT_POS_UPPER_LIMIT, 
    #                      joint_vel_upper_limit=JOINT_VEL_UPPER_LIMIT,
    #                      jonit_id_offset=JOINT_ID_OFFSET,
    #                      debug=True)
    
    # manip = Bimanipulation(URDF_NAME, 
    #                      start_pos, 
    #                      start_orn, 
    #                      main_arm=LEFT, 
    #                      control_freq=FPS,
    #                      resolution=0.01, 
    #                      joint_min=JOINT_POS_LOWER_LIMIT, 
    #                      joint_max=JOINT_POS_UPPER_LIMIT, 
    #                      joint_vel_upper_limit=JOINT_VEL_UPPER_LIMIT,
    #                      jonit_id_offset=JOINT_ID_OFFSET,
    #                      debug=True)
    
    # pos, _, _ = manip.robot.getJointStates(LEFT)
    
    # left_init = np.array([-0.7, -0.0422,  0.1596, -0.0566,  -0.3,  -0.8])
    # left_term = np.array([-0.6, -0.0422,  0.1596, -0.0566,  -0.3,  -0.7])
    # right_init = np.array([-0.305 ,  0.2584,  0.244 , -0.4763,  0.2967,  0.4014])

    # manip.robot.setArmJointStates(LEFT, left_init)
    # manip.robot.setArmJointStates(RIGHT, right_init)

    # left_traj = manip.motion_plan(LEFT, left_init, left_term)
    # right_traj = manip.motion_plan(RIGHT, right_init, right_term)

    # manip.simulate_traj(LEFT, traj[:,:6], draw=True)
    # manip.simulate_traj(RIGHT, traj[:,6:], draw=True)


    ### Bimanual
    manip = Bimanipulation(start_pos, 
                           start_orn, 
                           main_arm=LEFT, 
                           control_freq=FPS,
                           resolution=0.015, 
                           joint_min=JOINT_POS_LOWER_LIMIT, 
                           joint_max=JOINT_POS_UPPER_LIMIT, 
                           joint_vel_upper_limit=JOINT_VEL_UPPER_LIMIT,
                           joint_id_offset=JOINT_ID_OFFSET,
                           debug=True)
        
    ## 0: test trajectory
    # left_init = np.array([-0.7, -0.0422,  0.1596, -0.0566,  -0.3,  -0.8])
    # left_term = np.array([-0.6, -0.0422,  0.1596, -0.0566,  -0.3,  -0.7])
    
    ## 1: planar grasp 
    # left_init = np.array([ 0.2321, -0.1574, -0.1123,  0.0566, -0.0873, -1.6406])
    # left_term = np.array([-0.9924, -0.3443, -0.1101,  0.0093, -0.2793, -0.8203])

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

    # Compute Constrained right arm joint pose
    # right_init = manip.get_sub_joint_pose(left_init)

    ## 3: Bimanual snatch
    left_init = np.zeros(RobotParams.DOF)
    # left_pick = np.array([-0.334 , -0.0353, -0.2096,  0.3378, -0.0175,  0.2618])
    # left_release = np.array([-0.2417, -0.2497, -1.1164,  0.3504,  0.    ,  0.4189])
    # left_pick = np.array([-0.5232, -0.0521, -0.3767,  0.4393,  0.4189,  0.576 ])
    # left_release = np.array([-0.4358, -0.0525, -1.1492,  0.3225, -0.1396,  0.5061])
    left_pick = np.array([-0.4435,  0.0921, -0.6235,  0.8242,  0.4189,  0.4189])
    left_release = np.array([-0.4263,  0.0826, -1.2438,  0.6033,  0.5411,  0.3491])
    right_init = manip.from_left_to_right(left_init)

    # Move to grasping mode
    l_0, _, _ = manip.robot.getJointStates(LEFT)
    r_0, _, _ = manip.robot.getJointStates(RIGHT)

    left_traj = manip.motion_plan(LEFT, np.array(l_0), left_init)
    right_traj = manip.motion_plan(RIGHT, np.array(r_0), right_init)

    manip.simulate_traj(LEFT, left_traj, draw=True)
    manip.simulate_traj(RIGHT, right_traj, draw=True)
    
    p.removeAllUserDebugItems()

    # Bimanual motion planning
    # manip.robot.setArmJointStates(LEFT, left_init)
    # manip.robot.setArmJointStates(RIGHT, right_init)
    # traj = np.array(manip.synced_motion_plan(left_init, left_term))
    pick_traj = np.array(manip.mirrored_motion_plan(left_init, left_pick))
    release_traj = np.array(manip.mirrored_motion_plan(left_pick, left_release))
    traj = np.concatenate((pick_traj, release_traj), axis=0)

    manip.simulate_traj('all', traj, draw=True)

    manip.close()


if __name__ == '__main__':
    main()

    
