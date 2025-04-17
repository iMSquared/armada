#!/usr/bin/env python3
import sys
from pathlib import Path
CONTROL_BASE_DIR = Path(__file__).parent
if str(CONTROL_BASE_DIR) not in sys.path:
    sys.path.append(str(CONTROL_BASE_DIR))

import math
import numpy as np
import numpy.typing as npt
from typing import List, Dict, Tuple

from utils.dynamixel_client import *
import time

KP = 800
KI = 0
KD = 0
CURRENT_LIMIT = 500
L_ID = 2
R_ID = 1
L_INIT_POS = 76
R_INIT_POS = 142
L_RANGE = np.array([66, 126])
R_RANGE = np.array([132, 192])
USB_DIR = '/dev/ttyGripper'

class GripperControl:
    def __init__(self,
                 id: int=1, 
                 p_gain: float=KP,
                 i_gain: float=KI,
                 d_gain: float=KD,
                 current_limit: float=CURRENT_LIMIT,
                 init_pos: float=L_INIT_POS,
                 range: List[float]=L_RANGE,
                 **kwargs):
        
        self.kP = p_gain
        self.kI = i_gain
        self.kD = d_gain
        self.current_limit = current_limit
        self.prev_pos = self.pos = self.curr_pos = np.array([self.deg2rad(init_pos)])
        self.angle_limit = range
           
        #You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        self.motors = motors = [id]
        try:
            self.dxl_client = DynamixelClient(motors, USB_DIR, 115200)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', 115200)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(motors, 'COM8', 115200)
                self.dxl_client.connect()

        #Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        self.dxl_client.sync_write(motors, np.ones(len(motors))*5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        # self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2) # Pgain stiffness     
        # self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2) # Igain
        # self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2) # Dgain damping
        #Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        # self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.current_limit, 102, 2)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

        time.sleep(2)
        print(f'gripper position = {self.rad2deg(self.read_pos())}')
        print(f'gripper velocity = {self.read_vel()}')
        print(f'gripper currnet = {self.read_cur()}')

    def set_desired_pos(self, pose: float):
        self.curr_pos = np.array([pose])
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
        
    def rad2deg(self, radian):
        return radian * (180 / math.pi)

    def deg2rad(self, degree):
        return degree * (math.pi / 180)
        
    def move(self, desired_pos: float, delay: float=0.005):
        current_pos = float(self.rad2deg(self.read_pos())[0])

        if desired_pos > self.angle_limit[1] or desired_pos < self.angle_limit[0]:
            print(f'Joint limit {self.angle_limit}')
            return 0
        else:
            position_difference = desired_pos - current_pos
            steps: int = max(5, min(10, int(position_difference / 1)))
            step_size = position_difference / steps

            # print(f"[Gripper] steps={steps} delay={delay}")

            for i in range(steps):
                if self.read_cur() > self.current_limit:
                    current_pos += 0
                else:
                    current_pos += step_size
                self.set_desired_pos(self.deg2rad(current_pos))
                time.sleep(delay)

            return True
                
    #read position
    def read_pos(self):
        return self.dxl_client.read_pos()
    #read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()
    #read current
    def read_cur(self):
        return self.dxl_client.read_cur()

#init the node
def main(**kwargs):

    gripper_control = GripperControl()

    time.sleep(3)
    print('Set complete')

    while True:
        user_input = int(input("Enter a value (Enter 0 to stop): "))
        gripper_control.move(user_input)
        

if __name__ == "__main__":
    main()
