"""
RMD class implementation


Run command line for can interface
sudo ip link set can0 type can bitrate 1000000
sudo ip link set up can0


"<l", "<Hl" meaning

< : little-endian byte order
l : signed long integer (typically 32 bits or 4 bytes, in C).
H : unsigned short integer (usually 16 bits or 2 bytes, in C). 
h : signed short integer (usually 16 bits, or 2 bytes, in C)


"""

#Python library to be intalled with ù
#requires python 3.9 >
#pip install python-can
from typing import List
import can
import struct
import time
from pathlib import Path
import sys
import numpy as np

from communication.motorC2T.RealMotorT2C import RealMotorT2C


class RMD:


    #init the Motor
    def __init__(self,can_id,canbus):
        self.nodeID = 0x140+can_id
        self.ffpdID = 0x400+can_id
        self.bus = canbus
        #init used variables
        self.encoderPosition = 0
        self.multiTurn = 0
        self.singleTurn = 0

        self.PidPosKp  = 0
        self.PidPosKi  = 0
        self.PidVelKp  = 0
        self.PidVelKi  = 0
        self.PidTrqKp  = 0
        self.PidTrqKi  = 0
        self.acceleration  = 0

        self.actualTorque   = 0
        self.actualVelocity = 0
        self.actualPosition = 0

        #clear all can messages in queue
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)

    def info(self):
        print("######### INFO ########")
        print("\tPID:")
        stringa = "\t\tPosKp/i:"+str(self.PidPosKp)+":"+str(self.PidPosKi)+"\n"
        stringa += "\t\tVelKp/i:"+str(self.PidVelKp)+":"+str(self.PidVelKi)+"\n"
        stringa += "\t\tTrqKp/i:"+str(self.PidTrqKp)+":"+str(self.PidTrqKi)+"\n"
        print(stringa)
        print("\tAcceleration:")
        print("\t\t",self.acceleration)

    def print(self):
        stringa = "#POSITION "+ str(self.nodeID) +"     RAW:" + str(self.encoderPosition) +"\t"
        stringa += "M" + "{:.2f}".format(self.multiTurn) +"\t"
        stringa += "S" + "{:.2f}".format(self.singleTurn)
        print(stringa)

    def get_current_theta(self):
        stringa = "#POSITION     RAW:" + str(self.encoderPosition) +"\t"
        stringa += "M" + "{:.2f}".format(self.multiTurn) +"\t"
        stringa += "S" + "{:.2f}".format(self.singleTurn)
        print(stringa)

    def encoderInfo(self):

        stringa = "#POSITION\n\tRAW:" + str(self.encoderPosition)
        stringa += "\n\t" + str(self.encoderOriginalPosition)
        stringa += "\n\t" + str(self.encoderOffset)
        print(stringa)


      #write and receive RAW data
    def write_receive(self,data,ff_pd=False, log=False):
        if ff_pd: communication_id = self.ffpdID
        else: communication_id = self.nodeID
        msg = can.Message(arbitration_id=communication_id,
                      data=data,
                      is_extended_id=False)
        #try to send the message on the bus
        try:
            self.bus.send(msg)
        except can.CanError:
            if log:
                print("Message NOT sent")
            return (False, [])

        #read the response, no timeout on this action without arguments in the recv function
        try:
            msg = self.bus.recv(1.0)
        except:
            print("Message NOT received")
            return (False, [22,0,0,0,0,0,0,0])
        if msg is None:
            if log:
                print("Message is None")
            return (False, [22,0,0,0,0,0,0,0])
        return (True, msg.data)
    
      #write RAW data
    def write_only(self,data,ff_pd=False):
        if ff_pd: communication_id = self.ffpdID
        else: communication_id = self.nodeID
        msg = can.Message(arbitration_id=communication_id,
                      data=data,
                      is_extended_id=False)
        #try to send the message on the bus
        try:
            self.bus.send(msg)
        except can.CanError:
            print("Message NOT sent")
            return False

        return True

    #Calibrate offset to specific angle
    def Fn19(self):
        #Our offset is the actual original postion, shifted of 180degrees
        #180degres for this encoder are 0x8000 in hex
        #we can add the value and mod with 0x1000
        """
        We run in a stupid problem from the driver
        the offset is a unsigned value, and the output of the
        RAW after offset is always a unsigned 16bit value
        The problem raises when the multiturn is taking place
        this is because the internal system works like this:
        it "subtracts" the offset to the RAW sensor data
        but then it's used in the signed positioning value
        and it goes instead of 0 to 360
        """
        data = [0x19,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
        ret = self.write_receive(data)
        if (ret[0]) and (ret[1][0] == 0x19):
            pass
        else:
            print("ERRORE",data)
            for el in ret[1]:
                print(el)

    #read PIDs
    def Fn30(self):
        data = [0x30,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
        ret = self.write_receive(data)
        if (ret[0]) and (ret[1][0] == 0x30):
            print("Decode Pids")
            self.PidPosKp  = struct.unpack("B",ret[1][2:3])[0]
            self.PidPosKi  = struct.unpack("B",ret[1][3:4])[0]
            self.PidVelKp  = struct.unpack("B",ret[1][4:5])[0]
            self.PidVelKi  = struct.unpack("B",ret[1][5:6])[0]
            self.PidTrqKp  = struct.unpack("B",ret[1][6:7])[0]
            self.PidTrqKi  = struct.unpack("B",ret[1][7:8])[0]
        else:
            print("ERRORE",data)
            for el in ret[1]:
                print(el)

    #write PIDs RAM
    def Fn31(self):
        data = [0x31,0x00]
        data2 = struct.pack("BBBBBB",self.PidPosKp,self.PidPosKi,self.PidVelKp,self.PidVelKi,self.PidTrqKp,self.PidTrqKi)
        for el in data2:
            data.append(el)
        ret = self.write_receive(data)
        if (ret[0]) and (ret[1][0] == 0x31):
            self.PidPosKp  = struct.unpack("B",ret[1][2:3])[0]
            self.PidPosKi  = struct.unpack("B",ret[1][3:4])[0]
            self.PidVelKp  = struct.unpack("B",ret[1][4:5])[0]
            self.PidVelKi  = struct.unpack("B",ret[1][5:6])[0]
            self.PidTrqKp  = struct.unpack("B",ret[1][6:7])[0]
            self.PidTrqKi  = struct.unpack("B",ret[1][7:8])[0]
        else:
            print("ERRORE",data)
            for el in ret[1]:
                print(el)

    #read Acceleration
    def Fn33(self, log=False):
        data = [0x33,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
        ret = self.write_receive(data)
        if (ret[0]) and (ret[1][0] == 0x33):
            self.acceleration  = struct.unpack("<l",ret[1][4:8])[0]
        else:
            if log:
                print("ERRORE", data)
                for el in ret[1]:
                    print(el)

    #write Acceleration RAM
    def Fn34(self, log=False):
        data = [0x34,0x00,0x00,0x00]
        data2 = struct.pack("<l",self.acceleration)
        for el in data2:
            data.append(el)
        ret = self.write_receive(data)
        if (ret[0]) and (ret[1][0] == 0x34):
            self.acceleration  = struct.unpack("<l",ret[1][4:8])[0]
        else:
            if log:
                print("ERRORE", data)
                for el in ret[1]:
                    print(el)

    # Set current encoder position to ROM as motor zero
    def Fn64(self, log=False):
        data = [0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
        ret = self.write_receive(data)
        if (ret[0]) and (ret[1][0] == 0x64):
            if log:
                print("Zero set completed. Motor reset required.")
            self.Fn76(log=log)
        else:
            if log:
                print("ERROR from Fn64",data)
                for el in ret[1]:
                    print(el)

    # Motor reset
    def Fn76(self, log=False):
        data = [0x76,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
        reset = self.write_only(data)
        if log:
            if reset: 
                print("Motor reset completed.")
            else: 
                print("Motor reset FAILED!")

    #Motor OFF
    def Fn80(self, log=False):
        data = [0x80,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
        ret = self.write_receive(data)
        if (ret[0]) and (ret[1][0] == 0x80):
            if log:
                print("Motor OFF")
        else:
            if log:
                print("ERRORE",data)
                for el in ret[1]:
                    print(el)

    #Motor STOP
    def Fn81(self, log=False):
        data = [0x81,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
        ret = self.write_receive(data)
        if (ret[0]) and (ret[1][0] == 0x81):
            if log:
                print("Motor STOP")
        else:
            if log:
                print("ERRORE",data)
                for el in ret[1]:
                    print(el)

    #read internal encoder position and off set
    def Fn90(self, log=False):
        data = [0x90,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
        ret = self.write_receive(data)
        if (ret[0]) and (ret[1][0] == 0x90):
            self.encoderPosition  = struct.unpack("<H",ret[1][2:4])[0]
            self.encoderOriginalPosition = struct.unpack("<H",ret[1][4:6])[0]
            self.encoderOffset = struct.unpack("<H",ret[1][6:8])[0]
            #print(self.encoderPosition, self.encoderOriginalPosition, self.encoderOffset)
        else:
            if log:
                print("ERRORE",data)
                for el in ret[1]:
                    print(el)

    #Calibrate offset to specific angle
    def Fn91(self,angle = 180, log=False):
        #Our offset is the actual original postion, shifted of 180degrees
        #180degres for this encoder are 0x8000 in hex
        #we can add the value and mod with 0x1000
        """
        We run in a stupid problem from the driver
        the offset is a unsigned value, and the output of the
        RAW after offset is always a unsigned 16bit value
        The problem raises when the multiturn is taking place
        this is because the internal system works like this:
        it "subtracts" the offset to the RAW sensor data
        but then it's used in the signed positioning value
        and it goes instead of 0 to 360
        """
        value = self.encoderOriginalPosition + 0x8000
        value = value%0x10000
        data = [0x91,0x00,0x00,0x00,0x00,0x00]
        data2 = struct.pack("<H",value)
        for el in data2:
            data.append(el)
        ret = self.write_receive(data)
        if (ret[0]) and (ret[1][0] == 0x91):
            self.encoderOffset = struct.unpack("<H",ret[1][6:8])[0]
            #print(self.encoderPosition, self.encoderOriginalPosition, self.encoderOffset)
        else:
            if log:
                print("ERRORE",data)
                for el in ret[1]:
                    print(el)

    #read multi run angle
    def Fn92(self, log=False):
        data = [0x92,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
        ret = self.write_receive(data)
        if (ret[0]) and (ret[1][0] == 0x92):
            data = []
            for el in ret[1][1:]:
                data.append(el)
            data.append(data[-1])
            self.multiTurn  = struct.unpack("<q",bytes(data))[0]
            #print("multiBeforeChange",self.nodeID,self.multiTurn)
            #print(data)
            #self.multiTurn  += 18000
            self.multiTurn  /= 100
            if log:
                print("Actual angle: ", self.multiTurn)
        else:
            if log:
                print("ERRORE",data)
                for el in ret[1]:
                    print(el)

    #read single turn angle
    def Fn94(self, log=False):
        data = [0x94,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
        ret = self.write_receive(data)
        if (ret[0]) and (ret[1][0] == 0x94):
            self.singleTurn = struct.unpack("<H",ret[1][6:8])[0]/100
            if log:
                print("FN94",self.singleTurn)
        else:
            if log:
                print("ERRORE",data)
                for el in ret[1]:
                    print(el)

    # Current Control Cmd
    def FnA1(self,desiredCurrentx100, return_value=False, log=False):
        data = [0xA1,0x00,0x00,0x00]
        self.desiredTorque = desiredCurrentx100

        #print("GO TO",desiredPosition,self.desiredPosition)
        data2 = struct.pack("<h",self.desiredTorque)
        #print("SPACCEHTTA",struct.unpack("<Hl",data2))
        for el in data2:
            data.append(el)
        data.append(0x00)
        data.append(0x00)

        if log:
            print(f"--------- CONTROL || desired current: {self.desiredTorque*0.01}A")
            print(f"--------- Binary  || input: {RMD.bin2str(data)}")
            

        ret = self.write_receive(data)
        if (ret[0]) and (ret[1][0] == 0xA1):
            # We can read and update the response info of the motor
            self.actualTorque  = struct.unpack("<h",ret[1][2:4])[0]
            self.actualVelocity  = struct.unpack("<h",ret[1][4:6])[0]
            self.actualPosition  = struct.unpack("<h",ret[1][6:8])[0]
            if log:
                print(f"--------- CONTROL || current torque: {0.01*self.actualTorque}A, velocity: {self.actualVelocity}dps, position: {self.actualPosition}degree ---------")
                # print(f"--------- Binary  || output: {RMD.bin2str(ret[1])}")
            if return_value:
                return self.actualPosition, self.actualVelocity, 0.01*self.actualTorque

        else:
            if log:
                print("ERRORE",data)
                for el in ret[1]:
                    print(el)
            if return_value: return None

    # Speed Control Cmd
    def FnA2(self,desiredSpeed,log=False):
        data = [0xA2,0x00,0x00,0x00]
        self.desiredSpeed = desiredSpeed

        #print("GO TO",desiredPosition,self.desiredPosition)
        data2 = struct.pack("<l",self.desiredSpeed)
        #print("SPACCEHTTA",struct.unpack("<Hl",data2))
        for el in data2:
            data.append(el)

        ret = self.write_receive(data)
        if (ret[0]) and (ret[1][0] == 0xA2):
            # We can read and update the response info of the motor
            self.actualTorque  = struct.unpack("<h",ret[1][2:4])[0]
            self.actualVelocity  = struct.unpack("<h",ret[1][4:6])[0]
            self.actualPosition  = struct.unpack("<h",ret[1][6:8])[0]
            if log:
                print(f"--------- CONTROL || current torque: {0.01*self.actualTorque}A, velocity: {self.actualVelocity}dps, position: {self.actualPosition}degree ---------")

        else:
            if log:
                print("ERRORE",data)
                for el in ret[1]:
                    print(el)

    # Absolute Position Control Cmd
    def FnA4(self,desiredPosition,maxSpeed,log=False):
        data = [0xA4,0x00]
        #self.desiredPosition = desiredPosition +18000
        self.desiredPosition = desiredPosition
        self.maxSpeed = maxSpeed
        #print("GO TO",desiredPosition,self.desiredPosition)
        data2 = struct.pack("<Hl",maxSpeed,self.desiredPosition)
        #print("SPACCEHTTA",struct.unpack("<Hl",data2))
        for el in data2:
            data.append(el)
        ret = self.write_receive(data)
        if (ret[0]) and (ret[1][0] == 0xA4):
            # We can read and update the response info of the motor
            self.actualTorque  = struct.unpack("<h",ret[1][2:4])[0]
            self.actualVelocity  = struct.unpack("<h",ret[1][4:6])[0]
            self.actualPosition  = struct.unpack("<h",ret[1][6:8])[0]
            if log:
                print(f"--------- CONTROL || current torque: {0.01*self.actualTorque}A, velocity: {self.actualVelocity}dps, position: {self.actualPosition}degree ---------")
        else:
            if log:
                print("ERROR from FnA4",data)
                for el in ret[1]:
                    print(el)


    # Motion Mode Control command
    

    # CAUTION: After change can ID, you should reset the communication
    def change_can_ID(self):
        data = [0x79,0x00,0x00,0x00,0x00,0x00,0x00,0x07]
        ret = self.write_receive(data)
        print(ret)

    def check_can_ID(self):
        data = [0x79,0x00,0x01,0x00,0x00,0x00,0x00,0x00]
        ret = self.write_receive(data)
        print(ret)
        

    #read multi run angle (Return the actual motor output angle)
    def get_actual_angle(self, log=False):
        data = [0x92,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
        ret = self.write_receive(data)
        if (ret[0]) and (ret[1][0] == 0x92):
            data = []
            for el in ret[1][1:]:
                data.append(el)
            data.append(data[-1])
            self.multiTurn  = struct.unpack("<q",bytes(data))[0]
            self.multiTurn  /= 100
            angle = struct.unpack("<l",ret[1][4:8])[0] / 100.
        else:
            if log:
                print(f"ERROR from get_actual_angle | Response: {RMD.bin2str(data)}")
            angle = None
        return angle

    def set_zero(self, log=False):
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)
        if log:
            print(f"angle before set zero: {self.get_actual_angle()}")

        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)

        self.Fn64(log=log)

        #clear all can messages in queue
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)

        if log:
            print(f"angle after set zero: {self.get_actual_angle()}")
        msg = self.bus.recv(0.2)
        msg = self.bus.recv(0.2)

    def pos_control_degree(self,pos,speed,log=False):
        if log: self.print()
        self.FnA4((int)(pos*100),speed,log=log)
        if log:
            for _ in range(10):
                print(f"current angle: {self.get_actual_angle()}")
                time.sleep(0.1)

        return self.actualPosition*np.pi/180, self.actualVelocity*np.pi/180, 0.01*self.actualTorque

    def vel_control_degree(self,speed, log=False):
        if log: self.print()
        self.FnA2(speed*100, log=log)
        if log:
            for _ in range(10):
                print(f"current angle: {self.get_actual_angle()}")
                time.sleep(0.1)

    def cur_control(self, current, return_value=False, log=False):
        return self.FnA1(int(current*100), return_value=return_value, log=log)
    
    def torque_control_RMD(self, joint_idx, t2c:RealMotorT2C, des_tau, offset=None, log=False):
        des_cur = t2c.RMD_Torque2Current(joint_idx, des_tau)
        cur_pos, cur_vel, cur_cur = self.cur_control(des_cur, return_value=True, log=log)
        cur_tau = t2c.RMD_Current2Torque(joint_idx, cur_cur)
        cur_pos_rad = cur_pos* np.pi / 180. 

        if offset is not None:
            cur_pos_rad = cur_pos_rad + offset

        return cur_pos_rad, cur_vel* np.pi / 180., cur_tau

    def current_control_RMD(self, des_cur, offset=None, log=False):

        # start_time = time.time()

        cur_pos, cur_vel, cur_cur = self.cur_control(des_cur, return_value=True, log=log)
        cur_pos_rad = cur_pos* np.pi / 180. 

        # dur = time.time()-start_time
        # print(f'RMD  duration: {dur*1000} ms')

        if offset is not None:
            cur_pos_rad = cur_pos_rad + offset

        return cur_pos_rad, cur_vel* np.pi / 180., cur_cur

    def motor_stop(self, log=False):
        self.Fn81(log=log)

    def motor_shutdown(self, log=False):
        self.Fn80(log=log)

    def generate_data_ff_pd_RMD(self, p_des, v_des, kp, kd, t_ff, log=False):
        # Convert the signals to their respective hex values
        p_des_signal = (int)((p_des + 12.5)/25.*65535.)  # 16-bit
        v_des_signal = (int)((v_des + 45.)/90.*4095.)  # 12-bit
        kp_signal = (int)(kp/500.*4095.)  # 12-bit
        kd_signal = (int)(kd/5.*4095.)  # 12-bit
        t_ff_signal = (int)((t_ff + 24.)/48.*4095.)  # 12-bit
        
        # Create the CAN message by combining the signals
        # data = [
        #     p_des_signal & 0xFF, (p_des_signal >> 8) & 0xFF,         # p_des 16-bit
        #     v_des_signal & 0xFF, ((v_des_signal >> 8) & 0xF) << 4 | (kp_signal & 0xF),  # v_des 12-bit and kp 12-bit
        #     (kp_signal >> 4) & 0xFF,
        #     kd_signal & 0xFF, ((kd_signal >> 8) & 0xF) | (t_ff_signal & 0xF),  # kd 12-bit and t_ff 12-bit
        #     (t_ff_signal >> 4) & 0xFF
        # ]

        data = [
            (p_des_signal >> 8) & 0xFF, p_des_signal & 0xFF,          # p_des 16-bit
            (v_des_signal >> 4) & 0xFF, ((v_des_signal & 0xF) << 4) | ((kp_signal >> 8) & 0xF),  # v_des 12-bit and kp 12-bit
            kp_signal & 0xFF,
            (kd_signal >> 4) & 0xFF, ((kd_signal & 0xF) << 4) | ((t_ff_signal >> 8) & 0xF),  # kd 12-bit and t_ff 12-bit
            t_ff_signal & 0xFF
        ]

        if log:
            print(f"Signal | p_des: {p_des_signal}, v_des: {v_des_signal}, kp: {kp_signal}, kd: {kd_signal}, t_ff: {t_ff_signal}")
            print(f"data: {RMD.bin2str(data)}")
        
        return data

    def decode_msg_ff_pd_RMD(self, can_message: List, log=False):
        # Extract bytes and combine into the original signals
        p_des_signal = (can_message[1] << 8) | can_message[2]  # 16-bit
        v_des_signal = (can_message[3] << 4) | (can_message[4] >> 4)  # upper 12 bits
        t_ff_signal = ((can_message[4] << 8) & 0xF00) | can_message[5] # lower 4 bits + 8 bits

        # Convert signals back to their respective values
        p_des = (p_des_signal / 65535.0) * 25.0 - 12.5
        v_des = (v_des_signal / 4095.0) * 90.0 - 45.0
        t_ff = (t_ff_signal / 4095.0) * 48.0 - 24.0

        if log:
            print(f"Decoded Message| p_des={p_des}, v_des={v_des}, t_ff={t_ff}")
        
        return p_des, v_des, t_ff
    
    def ff_pd_RMD(self, p_des: float, v_des:float, kp:float, kd:float, t_ff:float, log=False):
        data = self.generate_data_ff_pd_RMD(p_des, v_des, kp, kd, t_ff, log=log)
        ret = self.write_receive(data, ff_pd=True, log=log)
        if ret[0]:
            if log:
                print(f"Response from CAN: {ret}")
            return self.decode_msg_ff_pd_RMD(ret[1])
        else:
            if log:
                print("ERROR from get_actual_angle",data)
                for el in ret[1]:
                    print(el)
            return None
        

    @staticmethod
    def bin2str(data: List):
        data_str = ""
        for i in data:
            data_str += f"{i:>08b}, "

        return data_str
            