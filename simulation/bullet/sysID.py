import sys
from pathlib import Path
BASEDIR = Path(__file__).parent.parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.append(str(BASEDIR))

from typing import Tuple, List, Dict
import numpy.typing as npt
import numpy as np
import time
import pickle
from simulation.bullet.bullet_sim import BulletEnvironment
from simulation.bullet.robot import LEFT, RIGHT, BIMANUAL
from simulation.bullet.utils.capture_env import *
from simulation.bullet.utils.collision import ContactBasedCollision
from simulation.bullet.utils.kinematics import Kinematics
from simulation.bullet.utils.common import interpolate_trajectory

from control.im2controller_client import IM2Client
import argparse

URDF_NAME = 'RobotBimanualV5'        # 'LinkageURDF3'
DATA_DIR = 'communication/sysID/waypoints/'
CONTROLLER = 'position'
IP = '[::]'
PORT = '50051'
FPS = 10.0
JOINT_POS_LOWER_LIMIT = np.array([-80.,  -40., -170.,  -35., -100., -115.])*np.pi/180
JOINT_POS_UPPER_LIMIT = np.array([40., 40., 40.,  65., 200. , 115.])*np.pi/180
JOINT_VEL_UPPER_LIMIT = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.1750, 2.1750])
JOINT_ID_OFFSET = 7
# TASKSPACE = ((0.1, -0.4, 0.509), (0.5, -0.03, 0.7))           # Right
TASKSPACE = ((0.1, 0.1, 0.509), (0.5, 0.4, 0.7))         # Left
KPS = np.array([1.8, 2.2, 1.6, 1.6, 0.7, 0.7])
KDS = np.array([0.06, 0.08, 0.1, 0.1, 0.04, 0.04])
PLAYSPEED = 2

class SysIDTrajectoryGenerator(BulletEnvironment):
    def __init__(self, robot_name: str,
                 start_pos: npt.ArrayLike, 
                 start_orn: npt.ArrayLike,
                 arm: str=LEFT,
                 controller_type: str=CONTROLLER,
                 control_freq: float=FPS,
                 joint_pos_lower_limit: npt.NDArray=JOINT_POS_LOWER_LIMIT,
                 joint_pos_upper_limit: npt.NDArray=JOINT_POS_UPPER_LIMIT,
                 joint_vel_upper_limit: npt.NDArray=JOINT_VEL_UPPER_LIMIT,
                 jonit_id_offset: int=JOINT_ID_OFFSET,
                 taskspace: npt.ArrayLike=TASKSPACE,
                 waypoints_file: str=None,
                 debug: bool = False, 
                 log: bool = False, **kwargs):
        super().__init__(robot_name, start_pos, start_orn, arm=arm, scene_on=True, debug=debug, log=log, gravity=True)
        self.bc.resetDebugVisualizerCamera(cameraDistance=1.2,
                                           cameraYaw=51.2,
                                           cameraPitch=-39.4,
                                           cameraTargetPosition=(-0.018, -0.0214, 0.44))

        # Controller and robot configs
        self.controller = controller_type
        self.control_freq = control_freq
        self.joint_pos_lower_limit = joint_pos_lower_limit
        self.joint_pos_upper_limit = joint_pos_upper_limit
        self.joint_vel_upper_limit = joint_vel_upper_limit
        self.joint_id_offset = jonit_id_offset

        # Taskspace
        self.taskspace = np.array(taskspace)

        # Kinematics (forward and inverse kinematics)
        self.kinematics = Kinematics(False, self.bc, self.robot)

        # Collision
        self.collision_fn = ContactBasedCollision(self.bc, 
                                                  self.robot.uid, 
                                                  joint_ids=list(self.robot.joint_id_dict.values()),
                                                  allowlist=[],
                                                  attachlist=[],
                                                  touchlist=[],
                                                  joint_limits=[self.joint_pos_lower_limit, self.joint_pos_upper_limit])

        # Save directory for generated trajectory
        self.data_dir = DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)

        # Trajectory file
        self.ee_waypoints = None
        if waypoints_file is not None:
            self.ee_waypoints: npt.NDArray = np.load(waypoints_file)


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

        
    def motion_plan(self,
                    arm: str,
                    init: npt.NDArray,
                    term: npt.NDArray,
                    duration:float,
                    control_dt:float,
                    planner: str='interpolate') -> List[npt.NDArray]:
        
        if planner == 'interpolate':
            traj = interpolate_trajectory(init, term, duration, control_dt)
        elif planner == 'taskspace':
            traj = self._interpolate_in_taskspace(arm, init, term, duration, control_dt)
        else:
            raise NotImplementedError('To be implemented')

        return traj


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
    

    def _generate_gains_traj(self, 
                             traj: List[npt.NDArray],
                             method: str='constant',
                             default_kps: npt.NDArray = KPS,
                             default_kds: npt.NDArray = KDS) -> Tuple[List[npt.NDArray], List[npt.NDArray]]:
        
        if method == 'constant':
            p_gains = [default_kps for _ in traj]
            d_gains = [default_kds for _ in traj]
        else:
            raise NotImplementedError
        
        return p_gains, d_gains


    def generate(self, arm: str, approach: bool=True, planner: str='interpolate', gains: str='constant'):

        joint_traj = self._generate_joint_traj(arm, approach, planner)
        p_gains, d_gains = self._generate_gains_traj(joint_traj, gains)

        return joint_traj, p_gains, d_gains

    
    def simulate_traj(self, arm: str, traj: List[npt.NDArray], draw: bool=True):
        prev_ee_pos = self.robot.get_ee_pose(arm)[0]
        for q in traj:
            st = time.time()
            self.robot.setArmJointStates(arm, q)
            cur_ee_pos = self.robot.get_ee_pose(arm)[0]
            # Draw
            if draw:
                self.bc.addUserDebugLine(prev_ee_pos, cur_ee_pos,
                                         lineColorRGB=[0, 0, 1],
                                         lineWidth=2)

            # Duration
            dur = time.time() - st
            if dur < (1.0/self.control_freq):
                time.sleep((1.0/self.control_freq)-dur)
            prev_ee_pos = cur_ee_pos


    def sync(self, client: IM2Client, arm: str, log: bool=False):
        try:
            while True:
                pos, vel, tau, grav = client.get_current_states(arm)
                self.robot.setArmJointStates(arm, pos)
                print(f'[INFO|SYNC] Joint Position: {pos} | EE_pose: {self.robot.get_ee_pose(arm)}')
        except KeyboardInterrupt:
            pass


    def measure(self, client: IM2Client, arm: str, trajectory_index: int):

        ee_pose_list = []

        while True:
            print('Do you want to measure? [y/n]')
            go = self._keyboard_propmt()
            if not go:
                break

            pos, _, _, _ = client.get_current_states(arm)
            ee_pose = self.kinematics.forward(arm, pos)

            ee_pose_list.append(np.concatenate(ee_pose))

        print('Save the EE waypoints? [y/n]')
        go = self._keyboard_propmt()
        if go:
            data = np.array(ee_pose_list)
            print(f'saved /sysID_traj{trajectory_index}.npy')
            np.save(self.data_dir+f'/sysID_traj{trajectory_index}.npy', data)


    def _keyboard_propmt(self):

        valid = False
        while not valid:
            key = input()

            if len(key) > 1:
                key = key[0]

            if key.lower() in ['y', 'n']:
                valid = True
                return key == 'y'
            

    def sample_obj_pose(self, grid: Tuple[float]=(3,3), draw: bool=False):

        goals = []
        l, w, h = np.abs(self.taskspace[1] - self.taskspace[0])       # l -> x, w -> y, h -> z

        grid_l = float(l/grid[0])
        grid_w = float(w/grid[1])

        x_0 = grid_l/2 + self.taskspace[0, 0]
        y_0 = grid_w/2 + self.taskspace[0, 1]

        for i in range(grid[0]):
            for j in range(grid[1]):
                x = x_0 + i*grid_l
                y = y_0 + j*grid_w

                goals.append((x, y, float(self.taskspace[0,2]), 0, 0, 0, 1))
                if draw:
                    self.bc.addUserDebugPoints([(x, y, float(self.taskspace[0,2]))], [[1,0,0]], 5)

        return goals


    def sample_init_joint(self, arm: str, num: int=10):
        
        joint_pos = []

        # Random sample
        while len(joint_pos) < num:
            q = np.random.random((self.robot.dof,))
            q *= (self.joint_pos_upper_limit - self.joint_pos_lower_limit)
            q += self.joint_pos_lower_limit

            ee_pos = np.array(self.kinematics.forward(arm, q)[0])

            # Within taskspace
            if np.any(ee_pos < self.taskspace[0]) or np.any(ee_pos > self.taskspace[1]):
                continue

            if self.collision_fn(q)[0]:
                continue

            joint_pos.append(q)
        
            self.bc.addUserDebugPoints([ee_pos], [[0,0,1]], 5)

        return joint_pos


    def close(self):
        self.bc.disconnect()


def get_params():
    parser = argparse.ArgumentParser(description='IM2Control server')
    parser.add_argument('--mode', required=True, help='[measure, traj_gen, sync]')
    parser.add_argument('--arm', default='left', help='[left, right]')
    parser.add_argument('--traj_id', type=int, default=0, help='For naming waypoint')
    parser.add_argument('--waypoints', type=str, help='Waypoint file name')

    return parser.parse_args()


def main():
    params = get_params()

    start_pos = [0, 0, 0]
    start_orn = [0, 0, 0]

    arm = params.arm

    waypoints = None
    if params.waypoints is not None:
        waypoints = DATA_DIR + f'/{params.waypoints}'
    traj_gen = SysIDTrajectoryGenerator(URDF_NAME, start_pos, start_orn,
                                        arm=arm, 
                                        waypoints_file=waypoints, 
                                        debug=False, 
                                        log=False)
    
    # init_joint_pos = traj_gen.sample_init_joint('right')
    # init_ee = [traj_gen.kinematics.forward(q)[0] for q in init_joint_pos]
    # traj_gen.bc.addUserDebugPoints(init_ee, [[0,0,1]]*10, 6)

    if params.mode == 'measure':
        client = IM2Client(ip=IP, port=PORT, dof=traj_gen.robot.dof)
        traj_gen.measure(client, arm, params.traj_id)

    elif params.mode == 'traj_gen':
        traj_gen.ee_waypoints= np.load(waypoints)
        traj, _, _ = traj_gen.generate(arm, approach=True, planner='taskspace')
    
    elif params.mode == 'sync':
        client = IM2Client(ip=IP, port=PORT, dof=traj_gen.robot.dof)
        traj_gen.sync(client, arm, log=False)
    
    elif params.mode == 'simulate':
        traj_gen.ee_waypoints= np.load(waypoints)
        traj, _, _ = traj_gen.generate(arm, approach=True, planner='taskspace')
        traj_gen.simulate_traj(arm, traj)
    else:
        raise ValueError


    


if __name__ == '__main__':
    main()



    
