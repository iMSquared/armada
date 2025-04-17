import sys
from pathlib import Path
BASEDIR = Path(__file__).parent.parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.append(str(BASEDIR))

from typing import Tuple, List, Dict, Callable
import numpy.typing as npt
import numpy as np
import time

from simulation.bullet.manipulation import *

URDF_NAME = 'simulation/assets/urdf/RobotBimanualV5/urdf/Simplify_Robot_plate_gripper.urdf'        # 'LinkageURDF3'
IP = '[::]'
PORT = '50051'
FPS = 200.0
JOINT_POS_LOWER_LIMIT = np.concatenate((RobotParams.L_JOINT_LIMIT_MIN, RobotParams.GRIPPER_SIM_MIN, RobotParams.R_JOINT_LIMIT_MIN, RobotParams.GRIPPER_SIM_MIN))
JOINT_POS_UPPER_LIMIT = np.concatenate((RobotParams.L_JOINT_LIMIT_MAX, RobotParams.GRIPPER_SIM_MAX, RobotParams.R_JOINT_LIMIT_MAX, RobotParams.GRIPPER_SIM_MAX))
JOINT_VEL_UPPER_LIMIT = RobotParams.JOINT_VEL_UPPER_LIMIT
JOINT_ID_OFFSET = RobotParams.DOF_ID_OFFSET
PALETTE = {'left': (0,0,1), 'right': (1,0,0)}

def add_joint_debug_params(manip: Manipulation):
    debugparams = []
    for i in range(RobotParams.DOF):
        debugparams.append(manip.bc.addUserDebugParameter(f"theta_{i+1}",RobotParams.L_JOINT_LIMIT_MIN[i],RobotParams.L_JOINT_LIMIT_MAX[i],0))

    return debugparams

def read_joint_debug_params(manip: Manipulation, debugparams: List):
    q = []
    for param in debugparams:
        q.append(manip.bc.readUserDebugParameter(param))

    return q

def main():
    start_pos = [0, 0, 0]
    start_orn = [0, 0, 0]

    ### Bimanual
    manip = Bimanipulation(start_pos, 
                           start_orn, 
                           main_arm=LEFT,
                           robot_name=URDF_NAME, 
                           control_freq=FPS,
                           resolution=0.01, 
                           joint_min=JOINT_POS_LOWER_LIMIT, 
                           joint_max=JOINT_POS_UPPER_LIMIT, 
                           joint_vel_upper_limit=JOINT_VEL_UPPER_LIMIT,
                           joint_id_offset=JOINT_ID_OFFSET,
                           debug=True,
                           scene_on=False)
    
    link_id_dict = {}
    for _id in range(manip.robot.bc.getNumJoints(manip.robot.uid)):
        joint_info = manip.bc.getJointInfo(manip.robot.uid, _id)
        link_name = joint_info[12].decode('UTF-8')
        link_id_dict[link_name] = _id
        print(link_name, _id)

    # link1 0
    # link2 1
    # link3 2
    # link4 3
    # link5 4
    # link6 5
    # tool1 6
    # link7 7
    # link8 8
    # link9 9
    # link10 10
    # link11 11
    # link12 12
    # tool2 13

    
    # T_O = ((0.3, 0, 0.5515), manip.bc.getQuaternionFromEuler((0, 0, 0)))

    # left_handle_pos = (0.3, 0.045, 0.5515)
    
    # box size: 26cm x 31.5cm x 27.5cm

    # cube_uid = manip.bc.loadURDF('simulation/assets/urdf/Cube/im2_cube.urdf', *T_O, useFixedBase=1)

    boxShape = (0.13, 0.1575, 0.1375) # size: (0.315, 0.26, 0.275)
    height_from_bottom = 0.73

    # T_O = ((0.25, 0.3, height_from_bottom+boxShape[2]/2.), manip.bc.getQuaternionFromEuler((0, 0, np.pi/4.)))
    # target_l_ee_pos = T_O[0] + 0.2/np.sqrt(2.)*np.array([-1., 1., 0.])
    # target_r_ee_pos = T_O[0] + 0.2/np.sqrt(2.)*np.array([1., -1., 0.])
    # target_l_q = manip.bc.calculateInverseKinematics(manip.robot.uid, 6, target_l_ee_pos, manip.bc.getQuaternionFromEuler((0, 0, -np.pi/4.)), maxNumIterations=1000)
    # target_r_q = manip.bc.calculateInverseKinematics(manip.robot.uid, 13, target_r_ee_pos, manip.bc.getQuaternionFromEuler((0, 0, -np.pi/4.)), maxNumIterations=1000)

    # T_O = ((0.25, -0.3, 0.5), manip.bc.getQuaternionFromEuler((0, 0, -np.pi/4.)))
    # target_l_ee_pos = T_O[0] + 0.2/np.sqrt(2.)*np.array([1., 1., 0.])
    # target_r_ee_pos = T_O[0] + 0.2/np.sqrt(2.)*np.array([-1., -1., 0.])
    # target_l_q = manip.bc.calculateInverseKinematics(manip.robot.uid, 6, target_l_ee_pos, manip.bc.getQuaternionFromEuler((0, 0, -3.*np.pi/4.)), maxNumIterations=1000)
    # target_r_q = manip.bc.calculateInverseKinematics(manip.robot.uid, 13, target_r_ee_pos, manip.bc.getQuaternionFromEuler((0, 0, -3.*np.pi/4.)), maxNumIterations=1000)

    # 0.3, 0.2, 0.86875, 0.000, 0.000, 0.383, 0.924
    T_O = ((0.3, 0.2, 0.86875), (0.000, 0.000, 0.383, 0.924))
    target_l_ee_pos = T_O[0] + 0.2/np.sqrt(2.)*np.array([-1., 1., 0.])
    target_r_ee_pos = T_O[0] + 0.2/np.sqrt(2.)*np.array([1., -1., 0.])
    target_l_q = manip.bc.calculateInverseKinematics(manip.robot.uid, 6, target_l_ee_pos, manip.bc.getQuaternionFromEuler((0, 0, -1*np.pi/4.)), maxNumIterations=1000)
    target_r_q = manip.bc.calculateInverseKinematics(manip.robot.uid, 13, target_r_ee_pos, manip.bc.getQuaternionFromEuler((0, 0, -1*np.pi/4.)), maxNumIterations=1000)
    

    manip.robot.setJointStates('left', target_l_q[:6])
    manip.robot.setJointStates('right', target_r_q[6:12])

    tableShape = (0.15, 0.20, height_from_bottom/2.)
    tablePose = (0.25, 0.3, height_from_bottom/2.)

    tableColor = (np.array([100, 100, 170, 255]) / 255.0).tolist()
    tableVisualShapeId = manip.bc.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=tableShape,
        rgbaColor=tableColor
    )
    tableCollisionShapeId = manip.bc.createCollisionShape(
        shapeType=p.GEOM_BOX, 
        halfExtents=tableShape
    )
    table_uid = manip.bc.createMultiBody(
        baseMass=10,
        baseCollisionShapeIndex=tableCollisionShapeId,
        baseVisualShapeIndex=tableVisualShapeId,
        basePosition=tablePose,
        baseOrientation=(0, 0, 0, 1)
    )

    boxColor = (np.array([170, 170, 170, 255]) / 255.0).tolist()
    boxVisualShapeId = manip.bc.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=boxShape,
        rgbaColor=boxColor
    )
    boxCollisionShapeId = manip.bc.createCollisionShape(
        shapeType=p.GEOM_BOX, 
        halfExtents=boxShape
    )
    box_uid = manip.bc.createMultiBody(
        baseMass=10,
        baseCollisionShapeIndex=boxCollisionShapeId,
        baseVisualShapeIndex=boxVisualShapeId,
        basePosition=T_O[0],
        baseOrientation=T_O[1]
    )
    print('left: (' + ', '.join(('%.3f' % f) for f in target_l_q[:6]) + ')')
    print('right: (' + ', '.join(('%.3f' % f) for f in target_r_q[6:12]) + ')')
    print(f'box init: {T_O[0]:}, table init: {tablePose}')
    print('quat: (' + ', '.join(('%.3f' % f) for f in T_O[1]) + ')')

    left_init_q = np.zeros(6)
    left_init_q[3] = 0.75

    right_init_q = np.zeros(6)
    right_init_q[3] = -0.75

    manip.robot.setJointStates('left', left_init_q)
    manip.robot.setJointStates('right', right_init_q)

    # left: [0.   0.   0.   0.75 0.   0.  ]
    # right: [ 0.    0.    0.   -0.75  0.    0.  ]
    print(f'left init: {left_init_q}, right init: {right_init_q}')
    
    l_ee_pose = manip.robot.get_ee_pose('all')[0]
    r_ee_pose = manip.robot.get_ee_pose('all')[1]


    manip.bc.addUserDebugPoints([l_ee_pose[0]], [[0,0,1]], 5)
    manip.bc.addUserDebugPoints([T_O[0]], [[1,0,1]], 5)

    # debugparams = add_joint_debug_params(manip)
    manip.bc.setRealTimeSimulation(True)

    target_l_ee_pos = np.array(l_ee_pose[0])
    target_l_ee_pos[1] -= 0.1

    target_l_q = manip.bc.calculateInverseKinematics(manip.robot.uid, 6, target_l_ee_pos, l_ee_pose[1])

    target_r_ee_pos = np.array(r_ee_pose[0])
    target_r_ee_pos[1] += 0.1

    target_r_q = manip.bc.calculateInverseKinematics(manip.robot.uid, 13, target_r_ee_pos, r_ee_pose[1])

    manip.robot.setJointStates('left', target_l_q[:6])
    manip.robot.setJointStates('right', target_r_q[6:12])

    # left: (-0.061392462522546, 0.1996183719652857, 0.0019825183187181274, 0.7484120476250624, 0.10403442053944419, 0.17980477468156866)
    # right: (0.06139212279180246, -0.19961835381111762, -0.0019835618589031873, -0.748409403304776, -0.10403468735720031, -0.17980442656237097)
    print(f'left grasp: {target_l_q[:6]}, right grasp: {target_r_q[6:12]}')

    target_l_ee_pos[2] = 0.737
    target_l_quat = (0.10567404727611206, -0.10566852275817577, -0.6991676301883463, 0.6991650620041041)

    target_l_q = manip.bc.calculateInverseKinematics(manip.robot.uid, 6, target_l_ee_pos, target_l_quat)

    target_r_ee_pos[2] = 0.737
    target_r_quat = target_l_quat

    target_r_q = manip.bc.calculateInverseKinematics(manip.robot.uid, 13, target_r_ee_pos, target_r_quat)

    manip.robot.setJointStates('left', target_l_q[:6])
    manip.robot.setJointStates('right', target_r_q[6:12])

    # left: (-0.18154423529267152, 0.29810494014474526, 0.07350331500380758, -0.3892091975412944, 0.33506407665406224, 0.07963879156139092)
    # right: (0.18153433557115561, -0.29809915934305586, -0.07350571697933644, 0.3892128456334144, -0.335055957767464, -0.07963156664887275)
    print(f'left lift: {target_l_q[:6]}, right lift: {target_r_q[6:12]}')
    
    target_l_ee_pos[0] += 0.3
    target_l_quat = (0., 0., -0.7071080798499331, 0.7071054825016957)

    target_l_q = manip.bc.calculateInverseKinematics(manip.robot.uid, 6, target_l_ee_pos, target_l_quat)

    target_r_ee_pos[0] += 0.3
    target_r_quat = target_l_quat

    target_r_q = manip.bc.calculateInverseKinematics(manip.robot.uid, 13, target_r_ee_pos, target_r_quat)

    manip.robot.setJointStates('left', target_l_q[:6])
    manip.robot.setJointStates('right', target_r_q[6:12])

    # left: (-0.24597135700196462, 0.0190495895028373, -0.8070755824495057, 0.7882453220066473, 0.02233438259532105, 0.2469707098842873)
    # right: (0.2459769377748551, -0.019047598601272725, 0.807073730994891, -0.7882432257128936, -0.022334779300551, -0.2469711221578937)
    print(f'far left: {target_l_q[:6]}, far right: {target_r_q[6:12]}')

    try:
        counter = 0
        while True:
            # st = time.time()
            # q = read_joint_debug_params(manip, manip.robot.debugparams)
            # manip.robot.setJointStates('left', q)

            left_ee_pos = manip.robot.get_ee_pose('left')[0]

            manip.bc.addUserDebugLine(left_handle_pos, left_ee_pos, lineColorRGB=(0,0,0), lineWidth=3, lifeTime=0.1)
            if counter % 10 == 0:
                print(f'Distance is {np.linalg.norm(np.array(left_handle_pos)-np.array(left_ee_pos))}')
                counter = 0
            else:
                counter += 1

            dur = time.time() - st
            if 0.01 - dur > 0:
                time.sleep(0.01 - dur)
    except Exception as e:
        manip.close()


if __name__ == '__main__':
    main()

    
