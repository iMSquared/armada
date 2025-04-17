import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import pyrealsense2 as rs
import numpy as np
from dt_apriltags import Detector
from PIL import Image
from scipy.spatial.transform import Rotation as R

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def matrix_from_R_T(R, T):
    T = np.reshape(T, (3))
    matrix = np.zeros((4, 4))
    matrix[:3, :3] = R
    matrix[:3, 3] = T
    matrix[3, 3] = 1.
    return matrix

def inverse_matrix_from_R_T(R, T):
    R_t = np.transpose(R)
    T_new = -1.*np.matmul(R_t, T)
    return matrix_from_R_T(R_t, T_new)

def get_camera_pose(rgb, cam_param):

    gray = rgb2gray(rgb)
    at_detector = Detector(families='tag36h11')
    tags = at_detector.detect(gray.astype(np.uint8), True, cam_param, 0.1591)
    tags_dict = {tag.tag_id: (tag.pose_R, tag.pose_t) for tag in tags}
    if len(tags_dict)!=1: return None, None, False
    r_cam_to_tag = tags_dict[0][0]
    t_cam_to_tag = tags_dict[0][1]
    print("original april tag R & T: ", r_cam_to_tag, t_cam_to_tag)
    r_cam_to_tag =  R.from_matrix(r_cam_to_tag) # * R.from_euler('y', np.pi)
    mat_cam = inverse_matrix_from_R_T(r_cam_to_tag.as_matrix(), t_cam_to_tag)

    offset_H = np.eye(4)
    offset_H[:3,:3] = R.from_euler('x', np.pi).as_matrix()
    mat_cam = offset_H@mat_cam
    Rm_ = mat_cam[:3,:3]
    mat_cam[:3,:3] = (R.from_matrix(Rm_) * R.from_euler('x', np.pi)).as_matrix()

    pos = mat_cam[:3, 3]
    rot = mat_cam[:3, :3]
    r = R.from_matrix(rot)

    modified_rot = r
    modified_pos = pos
    quat = modified_rot.as_quat()

    return modified_pos, quat, True

def camera_init(serial_no = '234322307454'):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_no)
    # config_1.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming from both cameras
    cfg = pipeline.start(config)

    profile_1 = cfg.get_stream(rs.stream.color) # Fetch stream profile for depth stream

    # intr_1 = profile_1.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
    # print("intrinsic: ", intr_1) # intrinsic:  [ 640x480  p[317.557 253.814]  f[608.916 608.687]
    # camera_param_1 = (608.916, 608.687, 317.557, 253.814)
    # cam_intrinsic = np.array([640., 480., 608.916, 608.687, 317.557, 253.814])

    align_to = rs.stream.color
    align = rs.align(align_to)

    time.sleep(0.1)
    get_img(pipeline, align)
    time.sleep(0.1)

    return pipeline, align

def get_img(pipeline, align):
    
    frames_1 = pipeline.wait_for_frames()
    frames_1 = align.process(frames_1)
    color_frame_1 = frames_1.get_color_frame()
    color_image_1 = np.asanyarray(color_frame_1.get_data())

    rgb_image = color_image_1[:, :, ::-1]

    return rgb_image