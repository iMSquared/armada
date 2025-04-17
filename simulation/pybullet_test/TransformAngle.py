import torch
from scipy.spatial.transform import Rotation as R
import numpy as np
import math

class TransformAngle:
    def __init__(self):
        None

    def quat2euler(self, q):
        q = R.from_quat(q)
        e = q.as_euler('xyz', degrees=True)
        return e

    def quat2axisangle(self, q):
        q = R.from_quat(q)
        a = q.as_rotvec()
        return a

    def axisangle2quat(self, a):
        a = R.from_rotvec(a)
        q = a.as_quat()
        return q

    def axisangle2euler(self, a):
        a = R.from_rotvec(a)
        e = a.as_euler('xyz', degrees=True)
        return e

    def euler2quat(self, e):
        e = R.from_euler('xyz', e, degrees=True)
        q = e.as_quat()
        return q

    def euler2axisangle(self, e):
        e = R.from_euler('xyz', e, degrees=True)
        a = e.as_rotvec()
        return a