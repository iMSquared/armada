
import sys
from pathlib import Path
BASEDIR = Path(__file__).parent.parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))

from typing import List, Dict, Tuple, Union, Iterable
import numpy as np
import numpy.typing as npt
import meshcat_shapes
import qpsolvers
from loop_rate_limiters import RateLimiter
from dataclasses import dataclass

# ARMADA
from control import DefaultControllerValues as RobotParams


# Pinocchio and Pink
import pink
from pink import solve_ik
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
from pink.visualization import start_meshcat_visualizer
from pink.tasks import FrameTask, PostureTask
from pink.tasks.task import Task
from pink.barriers import PositionBarrier

URDF_DIR = Path('simulation/assets/urdf/RobotBimanualV5/urdf/Simplify_Robot_plate_gripper.urdf')
FPS = 200

@dataclass
class IKtarget:
    '''
    name(str): must match the task name
    translation(NDArray): translation target w.r.t world-frame (robot root)
    orientation(NDArray): orientaiton target
    '''
    name: str       
    translation: npt.NDArray = None
    orientation: npt.NDArray = None


def build_ik(urdf_dir=URDF_DIR, **kwargs):
    visualize = kwargs.get('visualize_ik', False)
    ik_solver = OptBasedIK(urdf_dir, 100, visualize=visualize)

    ik_solver.setup_elbow_wrist_tracking(RobotParams.LEFT, 
                                         elbow_position_cost=10.0,
                                         elbow_orientation_cost=1.0,
                                         wrist_position_cost=10.0,
                                         wrist_orientation_cost=1.0)
    ik_solver.setup_elbow_wrist_tracking(RobotParams.RIGHT, 
                                         elbow_position_cost=10.0,
                                         elbow_orientation_cost=1.0,
                                         wrist_position_cost=10.0,
                                         wrist_orientation_cost=1.0)
    ik_solver.setup_posture(cost=0.5)
    
    ik_solver.set_target_from_configuration()

    ik_solver.set_viewer()

    return ik_solver


class OptBasedIK():
    def __init__(self,
                 urdf_dir: Union[Path, str]=URDF_DIR,
                 fps: float=FPS,
                 visualize: bool=False):
        
        # Set up pink
        if isinstance(urdf_dir, str):
            urdf_dir = Path(urdf_dir)
        model_path = urdf_dir.parent.parent.parent

        model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_dir,
                                                                       package_dirs=model_path,
                                                                       root_joint=None)

        self.robot: RobotWrapper = RobotWrapper(model,
                                                collision_model=collision_model,
                                                visual_model=visual_model)
        self.configuration: pink.Configuration = pink.Configuration(self.robot.model, self.robot.data, self.robot.q0)

        self.viz: MeshcatVisualizer = start_meshcat_visualizer(self.robot) if visualize else None
        self.tasks: Dict[str, Union[FrameTask, PostureTask]] = dict()

        # Solver
        solver = qpsolvers.available_solvers[0]
        if 'osqp' in qpsolvers.available_solvers:
            solver = 'osqp'
        self.solver = solver

        # Time
        self.rate = RateLimiter(frequency=100.0)
        self.dt = self.rate.period


    def setup_elbow_wrist_tracking(self,
                                   arm: str,
                                   elbow_position_cost: Union[float, Iterable[float]],
                                   elbow_orientation_cost: Union[float, Iterable[float]],
                                   wrist_position_cost: Union[float, Iterable[float]],
                                   wrist_orientation_cost: Union[float, Iterable[float]]):
        
        assert arm in [RobotParams.RIGHT, RobotParams.LEFT], 'Either right or left arm. No third arm.'

        if arm == RobotParams.RIGHT:
            elbow_link = 'link11'
            wrist_link = 'link12'
        else:
            elbow_link = 'link5'
            wrist_link = 'link6'

        elbow_task = FrameTask(elbow_link,
                               position_cost=elbow_position_cost,
                               orientation_cost=elbow_orientation_cost)
        
        wrist_task = FrameTask(wrist_link,
                               position_cost=wrist_position_cost,
                               orientation_cost=wrist_orientation_cost)     

          

        self.tasks[f'{arm}_elbow'] = elbow_task
        self.tasks[f'{arm}_wrist'] = wrist_task
        


    def setup_posture(self, cost: npt.NDArray):
        posture_task = PostureTask(cost=cost)
        posture_task.set_target(np.array([0, 0, -1.5708, 0, 0, 0, 0, 0, 1.5708, 0, 0, 0]))
        # posture_task.set_target(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        self.tasks['posture'] = posture_task

    def set_target_from_configuration(self):
        '''
        call after all the tasks are given
        '''

        for task in self.tasks.values():
            if not isinstance(task, PostureTask):
                task.set_target_from_configuration(self.configuration)
        
        if self.viz is not None:
            self.viz.display(self.configuration.q)


    def set_viewer(self):
        if self.viz is None:
            return
        viewer = self.viz.viewer
        for k in self.tasks.keys():
            meshcat_shapes.frame(viewer[f'{k}_target'], opacity=0.5)
            meshcat_shapes.frame(viewer[f'{k}_actual'], opacity=1.0)


    def ik(self, targets: List[IKtarget]):

        for target in targets:
            target_handle= self.tasks[target.name].transform_target_to_world
            if target.translation is not None:
                target_handle.translation[0] = target.translation[0]
                target_handle.translation[1] = target.translation[1]
                target_handle.translation[2] = target.translation[2]            

            if self.viz is not None:
                self.viz.viewer[f'{target.name}_target'].set_transform(target_handle.np)
                self.viz.viewer[f'{target.name}_actual'].set_transform(self.configuration.get_transform_frame_to_world(self.tasks[target.name].frame).np)

        velocity = solve_ik(self.configuration, self.tasks.values(), self.dt, solver=self.solver)
        self.configuration.integrate_inplace(velocity, self.dt)

        if self.viz is not None:
            self.viz.display(self.configuration.q)
            
        return self.configuration.q
    

    def get_current_frame(self, name: str) -> npt.NDArray:
        transform =  self.configuration.get_transform_frame_to_world(self.tasks[name].frame).np

        return transform


def main1():
    model_path = Path('simulation/assets/urdf')
    urdf_dir = model_path / 'RobotBimanualV5' /'urdf' / 'Simplify_Robot_plate_gripper.urdf'

    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_dir, 
                                                                   package_dirs=model_path,
                                                                   root_joint=None)
    robot = RobotWrapper(model, 
                         collision_model=collision_model, 
                         visual_model=visual_model)
    viz = start_meshcat_visualizer(robot)


    ee_task = FrameTask('link5', 
                        position_cost=10.0,
                        orientation_cost=1.0,)

    posture_task = PostureTask(cost=1e-3)

    tasks: List[Union[FrameTask, PostureTask]] = [ee_task, posture_task]

    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    for task in tasks:
        task.set_target_from_configuration(configuration)
    viz.display(configuration.q)

    viewer = viz.viewer
    meshcat_shapes.frame(viewer['ee_target'], opacity=0.5)
    meshcat_shapes.frame(viewer['ee'], opacity=1.0)

    solver = qpsolvers.available_solvers[0]
    if 'osqp' in qpsolvers.available_solvers:
        solver = 'osqp'

    rate = RateLimiter(frequency=100.0)
    dt = rate.period
    t = 0.0
    while True:
        ee_target = ee_task.transform_target_to_world
        ee_target.translation[1] = 0.1 + 0.1 * np.sin(t / 4)
        ee_target.translation[2] = 0.6

        viewer['ee_target'].set_transform(ee_target.np)
        viewer['ee'].set_transform(configuration.get_transform_frame_to_world(ee_task.frame).np)

        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        viz.display(configuration.q)
        rate.sleep()
        t += dt

def main2():
    import torch
    from imitation.utils.transforms import matrix_to_axis_angle


    urdf_dir = URDF_DIR
    arm = RobotParams.LEFT

    ik_solver = OptBasedIK(urdf_dir, 100, visualize=True)

    ik_solver.setup_elbow_wrist_tracking(RobotParams.LEFT, 
                                         elbow_position_cost=10.0,
                                         elbow_orientation_cost=1.0,
                                         wrist_position_cost=10.0,
                                         wrist_orientation_cost=1.0)
    ik_solver.setup_elbow_wrist_tracking(RobotParams.RIGHT, 
                                         elbow_position_cost=10.0,
                                         elbow_orientation_cost=1.0,
                                         wrist_position_cost=10.0,
                                         wrist_orientation_cost=1.0)
    
    ik_solver.setup_posture(cost=1.0)
    
    ik_solver.set_target_from_configuration()

    ik_solver.set_viewer()


    t = 0.0
    while True:
        left_elbow_frame = ik_solver.get_current_frame('left_elbow')
        left_elbow_translation = left_elbow_frame[:3, 3].copy()
        left_elbow_translation[1] = 0.1 + 0.1 * np.sin(t / 4)
        left_elbow_translation[2] = 0.6
        left_elbow_target = IKtarget('left_elbow', translation=left_elbow_translation)

        left_wrist_frame = ik_solver.get_current_frame('left_wrist')
        left_wrist_translation = left_wrist_frame[:3, 3].copy()
        left_wrist_translation[1] = 0.1 + 0.2 * np.sin(t / 4)
        left_wrist_translation[2] = 0.6
        left_wrist_target = IKtarget('left_wrist', translation=left_wrist_translation)

        right_elbow_frame = ik_solver.get_current_frame('right_elbow')
        right_elbow_translation = right_elbow_frame[:3, 3].copy()
        right_elbow_translation[1] = - 0.1 - 0.1 * np.sin(t / 4)
        right_elbow_translation[2] = 0.6
        right_elbow_target = IKtarget('right_elbow', translation=right_elbow_translation)

        right_wrist_frame = ik_solver.get_current_frame('right_wrist')
        right_wrist_translation = right_wrist_frame[:3, 3].copy()
        right_wrist_translation[1] = - 0.1 - 0.2 * np.sin(t / 4)
        right_wrist_translation[2] = 0.6
        right_wrist_target = IKtarget('right_wrist', translation=right_wrist_translation)

        q = ik_solver.ik([left_elbow_target, left_wrist_target, right_elbow_target, right_wrist_target])

        ik_solver.rate.sleep()
        t += ik_solver.dt



if __name__=='__main__':
    main2()    