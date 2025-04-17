import numpy as np
import math

from OrientationCal import OrientationCal


class TrajectoryGenerator:
    def __init__(self, contorl_frequency, n_joint):
        self.contorl_frequency = contorl_frequency
        self.time = 0.
        self.period = 0.
        self.current_position = []
        self.target_position = []

        self.NumberOfJoint = n_joint
        self.ref_var = []
        self.ref_vardot = []
        self.ref_varddot = []

        self.isEnd = False

        self.calori = OrientationCal()

    def ResetTime(self):
        self.time = 0

    def SetSinusoidalTrajectory(self, target_position, current_position, period):
        self.ref_var.clear()
        self.ref_vardot.clear()
        self.ref_varddot.clear()

        for i in range(self.NumberOfJoint):
            self.ref_var.append(0)
            self.ref_vardot.append(0)
            self.ref_varddot.append(0)


        self.current_position = current_position
        self.target_position = target_position
        self.period = period

        self.time += self.contorl_frequency
            
        for i in range(self.NumberOfJoint):
            self.ref_var[i] = (self.target_position[i]-self.current_position[i])*0.5*(1-math.cos(1.0*math.pi*self.time/self.period))+self.current_position[i]
            self.ref_vardot[i] = (self.target_position[i]-self.current_position[i])*0.5*(math.pi/self.period)*(math.sin(1.0*math.pi*self.time/self.period))
            self.ref_varddot[i] = (self.target_position[i]-self.current_position[i])*0.5*math.pow((math.pi/self.period),2)*(math.cos(1.0*math.pi*self.time/self.period))

    def SetSinusoidalPath(self, target_position, current_position, period):
        self.ref_var.clear()
        self.ref_vardot.clear()
        self.ref_varddot.clear()

        for i in range(6):
            self.ref_var.append(0)
            self.ref_vardot.append(0)
            self.ref_varddot.append(0)


        self.current_position = current_position
        self.target_position = target_position
        self.period = period

        self.time += self.contorl_frequency
            
        for i in range(6):
            self.ref_var[i] = (self.target_position[i]-self.current_position[i])*0.5*(1-math.cos(1.0*math.pi*self.time/self.period))+self.current_position[i]
            self.ref_vardot[i] = (self.target_position[i]-self.current_position[i])*0.5*(math.pi/self.period)*(math.sin(1.0*math.pi*self.time/self.period))
            self.ref_varddot[i] = (self.target_position[i]-self.current_position[i])*0.5*math.pow((math.pi/self.period),2)*(math.cos(1.0*math.pi*self.time/self.period))

    def SetSinusoidalPath_X(self, target_position, current_position, period):
        self.ref_var.clear()
        self.ref_vardot.clear()
        self.ref_varddot.clear()

        for i in range(3):
            self.ref_var.append(0)
            self.ref_vardot.append(0)
            self.ref_varddot.append(0)


        self.current_position = current_position
        self.target_position = target_position
        self.period = period

        self.time += self.contorl_frequency
            
        for i in range(3):
            self.ref_var[i] = (self.target_position[i]-self.current_position[i])*0.5*(1-math.cos(1.0*math.pi*self.time/self.period))+self.current_position[i]
            self.ref_vardot[i] = (self.target_position[i]-self.current_position[i])*0.5*(math.pi/self.period)*(math.sin(1.0*math.pi*self.time/self.period))
            self.ref_varddot[i] = (self.target_position[i]-self.current_position[i])*0.5*math.pow((math.pi/self.period),2)*(math.cos(1.0*math.pi*self.time/self.period))

    def SetSinusoidalPath_Ori(self, target_position, current_position, period):
        self.ref_var.clear()

        for i in range(4):
            self.ref_var.append(0)

        self.current_position = current_position
        self.target_position = target_position
        self.period = period

        self.time += self.contorl_frequency
            
        for i in range(4):
            self.ref_var[i] = (self.target_position[i]-self.current_position[i])*0.5*(1-math.cos(1.0*math.pi*self.time/self.period))+self.current_position[i]



    def Getvar(self):
        return self.ref_var, self.ref_vardot, self.ref_varddot
    
    def GetFinish(self):
        if self.time > self.period:
            return 1
        else:
            return 0