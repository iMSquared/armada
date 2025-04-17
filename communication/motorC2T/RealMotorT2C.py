import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import csv
import os, sys
from pathlib import Path
import pickle

BASEDIR = Path(__file__).parent.parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))

AK70_10_csv_file = "communication/motorC2T/dynamometer_AK70.pkl"
RMD_joint5_csv_file =  "communication/motorC2T/dynamometer_RMD_joint5.pkl"
RMD_joint6_csv_file =  "communication/motorC2T/dynamometer_RMD_joint6.pkl"

class RealMotorT2C:
    def __init__(self, profile_dir: Path = Path(__file__).parent.parent.parent) -> None:
        with open(profile_dir / AK70_10_csv_file, 'rb') as f:
            AK70_data = pickle.load(f)
        with open(profile_dir / RMD_joint5_csv_file, 'rb') as f:
            RMD_joint5_data = pickle.load(f)
        with open(profile_dir / RMD_joint6_csv_file, 'rb') as f:
            RMD_joint6_data = pickle.load(f)

        self.AK70_x_values = np.array(AK70_data['torque'])
        self.AK70_y_values = np.array(AK70_data['current'])
        self.RMD_joint5_x_values = np.array(RMD_joint5_data['torque'])
        self.RMD_joint5_y_values = np.array(RMD_joint5_data['current'])
        self.RMD_joint6_x_values = np.array(RMD_joint6_data['torque'])
        self.RMD_joint6_y_values = np.array(RMD_joint6_data['current'])

        self.AK70_torque_to_current = interp1d(
            self.AK70_x_values, self.AK70_y_values, kind='linear', fill_value='extrapolate'
        )
        self.AK70_current_to_torque = interp1d(
            self.AK70_y_values, self.AK70_x_values, kind='linear', fill_value='extrapolate'
        )

        self.RMD_joint5_torque_to_current = interp1d(
            self.RMD_joint5_x_values, self.RMD_joint5_y_values, kind='linear', fill_value='extrapolate'
        )
        self.RMD_joint5_current_to_torque = interp1d(
            self.RMD_joint5_y_values, self.RMD_joint5_x_values, kind='linear', fill_value='extrapolate'
        )

        self.RMD_joint6_torque_to_current = interp1d(
            self.RMD_joint6_x_values, self.RMD_joint6_y_values, kind='linear', fill_value='extrapolate'
        )
        self.RMD_joint6_current_to_torque = interp1d(
            self.RMD_joint6_y_values, self.RMD_joint6_x_values, kind='linear', fill_value='extrapolate'
        )

        self.T2Cepsilon_AK70 = 0.21555556
        self.C2Tepsilon_AK70 = self.AK70_torque_to_current(self.T2Cepsilon_AK70)

        self.T2Cepsilon_RMD_joint5 = 0.014
        self.C2Tepsilon_RMD_joint5 = self.RMD_joint5_torque_to_current(self.T2Cepsilon_RMD_joint5)

        self.T2Cepsilon_RMD_joint6 = 0.0012
        self.C2Tepsilon_RMD_joint6 = self.RMD_joint6_torque_to_current(self.T2Cepsilon_RMD_joint6)

        # Precompute slopes for the epsilon linear functions
        y_epsilon_AK70 = self.AK70_torque_to_current(self.T2Cepsilon_AK70)
        self.slope_AK70_T2C = y_epsilon_AK70 / self.T2Cepsilon_AK70
        self.slope_AK70_C2T = self.T2Cepsilon_AK70 / y_epsilon_AK70

        y_epsilon_RMD_joint5 = self.RMD_joint5_torque_to_current(self.T2Cepsilon_RMD_joint5)
        self.slope_RMD_joint5_T2C = y_epsilon_RMD_joint5 / self.T2Cepsilon_RMD_joint5
        self.slope_RMD_joint5_C2T = self.T2Cepsilon_RMD_joint5 / y_epsilon_RMD_joint5

        y_epsilon_RMD_joint6 = self.RMD_joint6_torque_to_current(self.T2Cepsilon_RMD_joint6)
        self.slope_RMD_joint6_T2C = y_epsilon_RMD_joint6 / self.T2Cepsilon_RMD_joint6
        self.slope_RMD_joint6_C2T = self.T2Cepsilon_RMD_joint6 / y_epsilon_RMD_joint6

    def AK70_Torque2Current(self, torque):
        torque = np.asarray(torque)
        abs_torque = np.abs(torque)
        sign_torque = np.sign(torque)

        current = np.empty_like(torque)

        mask = abs_torque > self.T2Cepsilon_AK70
        current[mask] = self.AK70_torque_to_current(abs_torque[mask]) * sign_torque[mask]
        current[~mask] = self.slope_AK70_T2C * torque[~mask]

        return current

    def AK70_Current2Torque(self, current):
        current = np.asarray(current)
        abs_current = np.abs(current)
        sign_current = np.sign(current)

        torque = np.empty_like(current)

        mask = abs_current > self.C2Tepsilon_AK70
        torque[mask] = self.AK70_current_to_torque(abs_current[mask]) * sign_current[mask]
        torque[~mask] = self.slope_AK70_C2T * current[~mask]

        return torque

    def RMD_Joint5_Torque2Current(self, torque):
        torque = np.asarray(torque)
        abs_torque = np.abs(torque)
        sign_torque = np.sign(torque)

        current = np.empty_like(torque)

        mask = abs_torque > self.T2Cepsilon_RMD_joint5
        current[mask] = self.RMD_joint5_torque_to_current(abs_torque[mask]) * sign_torque[mask]
        current[~mask] = self.slope_RMD_joint5_T2C * torque[~mask]

        return current

    def RMD_Joint5_Current2Torque(self, current):
        current = np.asarray(current)
        abs_current = np.abs(current)
        sign_current = np.sign(current)

        torque = np.empty_like(current)

        mask = abs_current > self.C2Tepsilon_RMD_joint5
        torque[mask] = self.RMD_joint5_current_to_torque(abs_current[mask]) * sign_current[mask]
        torque[~mask] = self.slope_RMD_joint5_C2T * current[~mask]

        return torque
    
    def RMD_Joint6_Torque2Current(self, torque):
        torque = np.asarray(torque)
        abs_torque = np.abs(torque)
        sign_torque = np.sign(torque)

        current = np.empty_like(torque)

        mask = abs_torque > self.T2Cepsilon_RMD_joint6
        current[mask] = self.RMD_joint6_torque_to_current(abs_torque[mask]) * sign_torque[mask]
        current[~mask] = self.slope_RMD_joint6_T2C * torque[~mask]

        return current

    def RMD_Joint6_Current2Torque(self, current):
        current = np.asarray(current)
        abs_current = np.abs(current)
        sign_current = np.sign(current)

        torque = np.empty_like(current)

        mask = abs_current > self.C2Tepsilon_RMD_joint6
        torque[mask] = self.RMD_joint6_current_to_torque(abs_current[mask]) * sign_current[mask]
        torque[~mask] = self.slope_RMD_joint6_C2T * current[~mask]

        return torque
    
    def RMD_Torque2Current(self, joint_idx, torque):
        if joint_idx == 4:
            return self.RMD_Joint5_Torque2Current(torque)
        elif joint_idx == 5:
            return self.RMD_Joint6_Torque2Current(torque)
        else:
            print("RMD joint idx error")
            raise NotImplementedError
        
    def RMD_Current2Torque(self, joint_idx, current):
        if joint_idx == 4:
            return self.RMD_Joint5_Current2Torque(current)
        elif joint_idx == 5:
            return self.RMD_Joint6_Current2Torque(current)
        else:
            print("RMD joint idx error")
            raise NotImplementedError
    
    def convertTorque2Current(self, torques: np.ndarray) -> np.ndarray:
        """
        Converts a torque array into a current array.
        The first 4 torques correspond to AK70 motors, and the rest correspond to RMD motors.

        Parameters:
            torques (np.ndarray): A numpy array of torques.

        Returns:
            np.ndarray: A numpy array of currents.
        """
        torques = np.asarray(torques)
        currents = np.empty_like(torques)
        currents[:4] = self.AK70_Torque2Current(torques[:4])
        currents[4] = self.RMD_Joint5_Torque2Current(torques[4])
        currents[5] = self.RMD_Joint6_Torque2Current(torques[5])
        return currents

    def convertCurrent2Torque(self, currents: np.ndarray) -> np.ndarray:
        """
        Converts a current array into a torque array.
        The first 4 currents correspond to AK70 motors, and the rest correspond to RMD motors.

        Parameters:
            currents (np.ndarray): A numpy array of currents.

        Returns:
            np.ndarray: A numpy array of torques.
        """
        currents = np.asarray(currents)
        torques = np.empty_like(currents)
        torques[:4] = self.AK70_Current2Torque(currents[:4])
        torques[4] = self.RMD_Joint5_Current2Torque(currents[4])
        torques[5] = self.RMD_Joint6_Current2Torque(currents[5])
        return torques
    
    def PlottingGraph(self):
        plt.xticks(range(0, 27, 3))
        plt.yticks(range(0, 30, 3))
        plt.plot(self.RMD_joint5_x_values, self.RMD_joint5_y_values)
        plt.plot(self.RMD_joint6_x_values, self.RMD_joint6_y_values)
        plt.grid(True)
        plt.show()
