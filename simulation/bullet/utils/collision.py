#!/usr/bin/env python3

import numpy as np
import logging

from collections import defaultdict
from typing import Tuple, Iterable, Optional, Dict, List, Union
import numpy.typing as npt
from dataclasses import dataclass
import pybullet as p
from pybullet_utils.bullet_client import BulletClient

from simulation.bullet.utils.common import (
    get_relative_transform,
    get_link_pose,
    get_name_from_index
)
from control import DefaultControllerValues as RobotParams


@dataclass(frozen=True)
class LinkPair:
    body_id_a: int
    link_id_a: int
    body_id_b: int
    link_id_b: int


def is_allowed(allowlist: Iterable[LinkPair], contact: List):
    """Checks whether a given contact is allowed."""
    for C in allowlist:
        if contact[1] != C.body_id_a:
            continue
        if contact[2] != C.body_id_b:
            continue

        if (C.link_id_a is not None):
            if C.link_id_a != contact[3]:
                continue

        if (C.link_id_b is not None):
            if C.link_id_b != contact[4]:
                continue
        # allowed
        return True
    # not allowed
    return False


class SetRobotState:
    def __init__(self, bc: BulletClient,
                 robot_id: int,
                 attachlist: Iterable[LinkPair]):
        self.bc = bc
        self.robot_id = robot_id

        # Lookup relative transforms
        # of attachments.
        self.attach_xfms = {
            C: get_relative_transform(bc,
                                      C.body_id_a,
                                      C.link_id_a,
                                      C.link_id_b,
                                      C.body_id_b,
                                      inertial=False)
            for C in attachlist}

    def __call__(self, 
                 arm: str, 
                 value: Union[npt.ArrayLike, Tuple[npt.ArrayLike]], 
                 debug: bool=False):

        # Set kinematic states.
        self._update_robot(arm, value, debug)
        self._update_attached_bodies()


    def _update_robot(self, 
                      arm: str, 
                      value: Union[npt.ArrayLike, Tuple[npt.ArrayLike]], 
                      debug:bool = False):
        raise NotImplementedError

            
    def _update_attached_bodies(self):
        bc = self.bc
        # Update transforms of attached bodies.
        for C, xfm in self.attach_xfms.items():
            # NOTE(ycho): as long as we're consistent
            # about `inertial` keyword, we're fine.
            pose = get_link_pose(bc, C.body_id_a, C.link_id_a,
                                 inertial=False)
            pose = bc.multiplyTransforms(pose[0], pose[1],
                                         xfm[0], xfm[1])
            bc.resetBasePositionAndOrientation(
                C.body_id_b, pose[0], pose[1])


class SetRobotJointState(SetRobotState):
    def __init__(self, bc: BulletClient,
                 robot_id: int,
                 joint_ids: Iterable[int],
                 attachlist: Iterable[LinkPair]):
        
        super().__init__(bc=bc, robot_id=robot_id, attachlist=attachlist)
        self.joint_ids = np.array(joint_ids)


    def _update_robot(self, arm: str, q: npt.ArrayLike, debug: bool=False):
        indices = np.array(range(RobotParams.DOF)) if arm == 'left' else np.array(range(RobotParams.DOF, RobotParams.DOF*2))
        for i, v in zip(self.joint_ids[indices], q):
            self.bc.resetJointState(self.robot_id, i, v)
        


class SetRobotSyncedJointState(SetRobotJointState):
    def __init__(self, bc: BulletClient,
                 robot_id: int,
                 joint_ids: Iterable[int],
                 attachlist: Iterable[LinkPair]):
        
        super().__init__(bc=bc, robot_id=robot_id, joint_ids=joint_ids, attachlist=attachlist)


    def _update_robot(self, main_arm: str, qs: Tuple[npt.ArrayLike], debug: bool=False):
        left_indices = np.array(range(RobotParams.DOF))
        right_indices = np.array(range(RobotParams.DOF, RobotParams.DOF*2))

        (main_indices, sub_indices) = (left_indices, right_indices) if main_arm == 'left' else (right_indices, left_indices)

        main_q, sub_q = qs

        for i, v in zip(self.joint_ids[main_indices], main_q):
            self.bc.resetJointState(self.robot_id, i, v)

        for i, v in zip(self.joint_ids[sub_indices], sub_q):
            self.bc.resetJointState(self.robot_id, i, v)

            
class SetRobotPose(SetRobotState):
    def __init__(self, bc: BulletClient,
                 robot_id: int,
                 attachlist: Iterable[LinkPair]):
        super().__init__(bc=bc, robot_id=robot_id, attachlist=attachlist)


    def _update_robot(self, arm: str, value: npt.ArrayLike, debug: bool=False):
        # Set robot pose.
        if debug:
            x, y, _ = tuple(value[0])
            self.bc.addUserDebugPoints([(x, y, 0.005)], [(0, 0, 1)], pointSize=3)
        self.bc.resetBasePositionAndOrientation(self.robot_id, value[0], value[1])


class Collision:
    """General collision class.
    """

    def __init__(
            self, bc: BulletClient, robot_id: int,
            allowlist: Iterable[LinkPair] = [],
            attachlist: Iterable[LinkPair] = [],
            touchlist: Iterable[LinkPair] = [],
            tol: Optional[Dict[int, float]] = None,
            touch_tol=0.002,
            return_contact: bool=False):
        self.bc = bc
        self.robot_id = robot_id
        self.attachlist = attachlist
        self.tol = tol
        self.touch_tol = touch_tol
        self.set_state = SetRobotState(bc, robot_id, attachlist)
        self.return_contact = return_contact

        # Split by `body_id_a` for convenience.
        self.allowlist = defaultdict(list)
        for C in allowlist:
            self.allowlist[C.body_id_a].append(C)

        self.touchlist = defaultdict(list)
        for C in touchlist:
            self.touchlist[C.body_id_a].append(C)

        # Lookup relative transforms
        # of attachments.
        self.attach_xfms = {
            C: get_relative_transform(bc,
                                      C.body_id_a,
                                      C.link_id_a,
                                      C.link_id_b,
                                      C.body_id_b)
            for C in attachlist}
        
    def _convert(self, value: npt.ArrayLike) -> npt.ArrayLike:
        return value

    def __call__(self, value: npt.ArrayLike, arm: str=None, object_uid: int=None, debug: bool=False) -> bool:
        bc = self.bc
        robot_id: int = self.robot_id
        value = np.asarray(value)

        # Set robot state.
        value = self._convert(value)
        self.set_state(arm, value, debug)

        # Perform collision detection.
        bc.performCollisionDetection()

        # Check collisions.
        # We primarily check the robot and the attached bodies.
        bodies = [robot_id] + [C.body_id_b for C in self.attachlist]

        # Configure tolerances.
        if self.tol is None:
            tol = {}
        else:
            tol = self.tol
        for body_id in bodies:
            tol.setdefault(body_id, -5e-5)
        touch_tol = self.touch_tol


        for body in bodies:
            if object_uid is None:
                contacts = bc.getContactPoints(bodyA=body)
            else:
                contacts = bc.getContactPoints(bodyA=body, bodyB=object_uid)

            filtered_contacts = []
            allowlist = self.allowlist.get(body, [])
            touchlist = self.touchlist.get(body, [])
            for contact in contacts:
                if contact[8] >= tol[body]:
                    continue
                if is_allowed(allowlist, contact):
                    continue
                if is_allowed(touchlist, contact) and contact[8] >= touch_tol:
                    continue
                filtered_contacts.append(contact)
            contacts = filtered_contacts

            if len(contacts) > 0:
                msg = ''
                # In case of contact, optionally output debug messages.
                if debug:
                    for pt in contacts:
                        try:
                            names_a = get_name_from_index(
                                pt[1], bc.sim_id, [pt[3]], link=True)
                            names_b = get_name_from_index(
                                pt[2], bc.sim_id, [pt[4]], link=True)
                            msg += F'names_a = {names_a}, names_b = {names_b}\n'
                        except p.error:
                            msg += F'{pt[1], pt[2], pt[3], pt[4]}\n'
                            continue
                    #logging.debug(msg)
                    print(msg)
                return (True, contacts) if self.return_contact else True
        return (False, contacts) if self.return_contact else False


class ContactBasedCollision(Collision):
    """General contact-based collision checker.

    see pb.getContactPoints()
    """

    def __init__(
            self, bc: BulletClient, robot_id: int, joint_ids: Iterable[int],
            allowlist: Iterable[LinkPair],
            attachlist: Iterable[LinkPair],
            touchlist: Iterable[LinkPair] = [],
            joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            tol: Optional[Dict[int, float]] = None,
            touch_tol=0.002,
            return_contact: bool=False):

        self.joint_ids = joint_ids
        self.joint_limits = np.asarray(joint_limits)

        super().__init__(bc=bc, robot_id=robot_id,
                         allowlist=allowlist,
                         attachlist=attachlist,
                         touchlist=touchlist,
                         tol=tol, touch_tol=touch_tol,
                         return_contact=return_contact)

        self.set_state = SetRobotJointState(bc, robot_id, joint_ids, attachlist)


    def __call__(self, q: npt.NDArray, arm: str=None, object_uid: int=None, debug: bool = False) -> bool:

        joint_limits = self.joint_limits
        q = np.array(q)

        # Check if exceeding joint limits.
        # NOTE(ssh): joint_to_check ignores fixed joints
        # joint_to_check = np.where(joint_limits[0][indices] <= joint_limits[1][indices])
        joint_to_check = np.array([0,1,2,3,4,5]) #if arm == 'left' else np.array([6,7,8,9,10,11])
        if (q < joint_limits[0, joint_to_check]).any():
            logging.debug(F'Hit lower jlim: {q} < {joint_limits[0, joint_to_check]}')
            return True, []
        if (q >= joint_limits[1, joint_to_check]).any():
            logging.debug(F'Hit upper jlim: {q} >= {joint_limits[1,joint_to_check]}')
            return True, []
        
        return super().__call__(q, arm=arm, object_uid=object_uid, debug=debug)


class NavigationCollision(Collision):
    '''
        Collsion function for Navigation
    '''
    def __init__(self, bc: BulletClient, 
                 robot_id: int, 
                 allowlist: Iterable[LinkPair] = [], 
                 attachlist: Iterable[LinkPair] = [], 
                 touchlist: Iterable[LinkPair] = [], 
                 tol: Union[Dict[int, float], None] = None, 
                 touch_tol=0.002):
        super().__init__(bc, robot_id, allowlist, attachlist, touchlist, tol, touch_tol)
        self.set_state = SetRobotPose(self.bc, self.robot_id, attachlist=attachlist)
    
    def _convert(self, value: npt.ArrayLike):

        x, y, theta = tuple(value)
        z = self.bc.getBasePositionAndOrientation(self.robot_id)[0][2]
        orn = self.bc.getQuaternionFromEuler((0, 0, theta))        
        
        pose = ((x,y,z),orn)

        return pose
    

class BimanualCollision(ContactBasedCollision):
    """General contact-based collision checker.

    see pb.getContactPoints()
    """

    def __init__(
            self, 
            bc: BulletClient, 
            robot_id: int,
            main_arm: str, 
            joint_ids: Iterable[int],
            allowlist: Iterable[LinkPair],
            attachlist: Iterable[LinkPair],
            touchlist: Iterable[LinkPair] = [],
            joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            tol: Optional[Dict[int, float]] = None,
            touch_tol=0.002,
            return_contact: bool=False):


        super().__init__(bc=bc, 
                         robot_id=robot_id,
                         joint_ids=joint_ids,
                         allowlist=allowlist,
                         attachlist=attachlist,
                         touchlist=touchlist,
                         joint_limits=joint_limits,
                         tol=tol, touch_tol=touch_tol,
                         return_contact=return_contact)
        
        self.main_arm = main_arm
        self.sub_arm = 'right' if self.main_arm == 'left' else 'left'

        self.set_state = SetRobotSyncedJointState(bc, robot_id, joint_ids, attachlist)


    def __call__(self, qs: npt.NDArray, object_uid: int=None, debug: bool = False) -> bool:

        joint_limits = self.joint_limits
        

        qs = qs.reshape(-1, RobotParams.DOF)

        # Check if exceeding joint limits.
        for q, arm in zip(qs, (self.main_arm, self.sub_arm)):
            joint_to_check = np.array(range(RobotParams.DOF)) if arm == 'left' else np.array(range(RobotParams.DOF, RobotParams.DOF*2))
            if (q < joint_limits[0, joint_to_check]).any():
                logging.debug(F'Hit lower jlim: {q} < {joint_limits[0, joint_to_check]}')
                return True, []
            if (q >= joint_limits[1, joint_to_check]).any():
                logging.debug(F'Hit upper jlim: {q} >= {joint_limits[1,joint_to_check]}')
                return True, []
        
        return Collision.__call__(self, arm=self.main_arm, value=qs, object_uid=object_uid, debug=debug)