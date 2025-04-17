import motor.RMD as RMD
import can
import time
import numpy as np

# bus = can.interface.Bus(bustype="socketcan", channel="can0", bitrate=1000000)

# motor = RMD.RMD(0x141, bus)

bus0 = can.interface.Bus(bustype="socketcan", channel="can0", bitrate=1000000)
bus1 = can.interface.Bus(bustype="socketcan", channel="can1", bitrate=1000000)

joint5 = RMD.RMD(0x141, bus0)
joint6 = RMD.RMD(0x141, bus1)


# motor.change_can_ID()
# motor.check_can_ID()

# motor.Fn30()  # read PID
# motor.Fn33()  # read acceleration

# motor.info() # 'info' prints Pid and Acceleration
# motor.Fn90() # read internal encoder position and off set
# motor.Fn92() # read multi run angle





##### Position control #####

time.sleep(0.1)
joint5.set_zero()
joint6.set_zero()
time.sleep(0.1)

N = 100
for i in range(N+1):
    target_theta = 45*np.sin(2.*np.pi*i/N)
    joint6.pos_control_degree(target_theta, 100)
    print(f"current angle: {joint6.get_actual_angle()}")
    time.sleep(0.03)

N = 100
for i in range(N+1):
    target_theta = 45*np.sin(2.*np.pi*i/N)
    joint5.pos_control_degree(target_theta, 100)
    print(f"current angle: {joint5.get_actual_angle()}")
    time.sleep(0.03)

joint5.motor_stop()
joint6.motor_stop()

joint5.motor_shutdown()
joint6.motor_shutdown()

##### Velocity control #####

# time.sleep(1)
# motor.set_zero()

# time.sleep(1)
# motor.vel_control(100)

# time.sleep(1)
# motor.motor_stop()


##### Torque control #####

# time.sleep(1)
# motor.set_zero()

# time.sleep(1)
# motor.trq_control(0.2) # CAUTION! be careful to use torque bigger than 0.2A! unit: A (current)

# time.sleep(0.3)
# motor.motor_stop()






# Specify desired PID
# motor_E.PidPosKp = 30
# motor_E.PidPosKi = 0
# motor_E.PidVelKp = 30
# motor_E.PidVelKi = 5
# motor_E.PidTrqKp = 30
# motor_E.PidTrqKp = 5
# motor_E.Fn31()  # write PID to Ram

# Specify desired accelration
# motor_E.acceleration = a
# motor_E.Fn34()  # write acceleration to Ram