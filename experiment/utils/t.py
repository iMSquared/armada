import cv2
import numpy as np

# Define video properties
output_path = '/home/user/backup_240213/personal/hardware/sim2real-robot-arm/keypoint_vis/output_video_fp_with_robot_2.mp4'   # Output video file
frame_rate = 24                    # Frames per second
frame_size = (320, 240)            # Size of each frame (width, height)

# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, frame_size)

# Loop through each image
for i in range(1100):
    # Read the image
    image_path = f'/home/user/backup_240213/personal/hardware/sim2real-robot-arm/keypoint_vis/image_{i}.png'
    img = cv2.imread(image_path)

    # Resize image to match video size if necessary
    if img.shape[1] != frame_size[0] or img.shape[0] != frame_size[1]:
        img = cv2.resize(img, frame_size)

    # Write the frame to the video
    video_writer.write(img)

# Release the video writer
video_writer.release()
print("Video created successfully.")