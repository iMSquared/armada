import os
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Union
import numpy as np
import numpy.typing as npt
import pybullet as p
from pybullet_utils.bullet_client import BulletClient
from simulation.bullet.robot import Robot
import trimesh


@dataclass(frozen=True)
class RobotCapture:
    joint_position: npt.ArrayLike        
    robot_base: npt.ArrayLike

@contextmanager
def dream(bc: BulletClient, robot: Robot):
    try:
        cap = capture(robot)
        yield cap
    finally:
        set_from_capture(bc, robot, cap)


def capture(robot: Robot) -> RobotCapture:
    
    # Parse robot arm
    joint_position, _, _ = robot.getJointStates(robot.arm)
    joint_position = tuple(joint_position)

    capture = RobotCapture(joint_position=joint_position,
                           robot_base=robot.pose)
    
    return capture


def set_from_capture(bc: BulletClient, robot: Robot, capture: RobotCapture):
    """
    - Reset robot joint configuration.

    Args:
        bc (BulletClient)
        robot_uidr (int)
        capture (RobotCapture)
    """

    # Reset robot base
    robot.pose = capture.robot_base

    # Reset robot joints
    robot.setJointStates('all', capture.joint_position)
