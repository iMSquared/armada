import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import cv2

# with open(f'real_policy_execute_{current_time}.pkl', 'wb') as f:
#         pickle.dump({'time':np.array(time_list), 'joint_pos':np.array(joint_pos_list), \
#                     'joint_vel':np.array(joint_vel_list), 'joint_tau':np.array(joint_tau_list), \
#                     'des_pos': np.array(des_pos_list), 't_policy_list':np.array(t_policy_list), \
#                     'obs': obs_list, 'imgs': img_list, 'des_tau_list':np.array(des_tau_list), \
#                     'kps':kps, 'kds':kds, 'time_checker':np.array(time_checker)}, f)



joint_pos_lower_limit = np.array([-1.7453, -0.6981, -1.7453, -0.7854, -2.9671, -0.6109])
joint_pos_upper_limit = np.array([1.1345, 0.8727, 0.8727, 1.3090, 2.9671, 3.7525])
joint_vel_upper_limit = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.1750, 2.1750])
joint_tau_upper_limit = np.array([4., 4., 2., 2., 1., 1.])

file_path = '/home/user/workspace/perception/experiment/result/real_policy_data_0257PM_January072025.pkl'  # Replace with your file path

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def extract_kps_kds(observation):
    """
    Extract kp and kd from the observation list of tensors.
    
    Parameters:
    observation (list of tensors): List of observation tensors.

    Returns:
    np.ndarray: kp values (N_timestep, 6).
    np.ndarray: kd values (N_timestep, 6).
    """
    kps = []
    kds = []
    for obs in observation:
        obs_np = obs.clone().detach().cpu().numpy()
        kps.append(obs_np[0, 51:57])  # kp values
        kds.append(obs_np[0, 57:63])  # kd values
    return np.array(kps), np.array(kds)

def visualize_motor_data(time_steps, kp, kd, des_pos, joint_pos, residual, pos_limits):
    num_motors = kp.shape[1]
    
    fig, axes = plt.subplots(num_motors, 4, figsize=(20, 15), sharex=True)

    action_length = kp[:-1, 0].shape[0]
    timestep_action = 10*np.array(list(range(action_length)))
    
    
    for i in range(num_motors):
        axes[i, 0].plot(time_steps[timestep_action], kp[:-1, i])  # kp of previous timestep
        axes[i, 0].set_title(f'Joint {i+1} Kp Over Time')
        axes[i, 0].set_xlabel('Time Step')
        axes[i, 0].set_ylabel('Kp Value')
        axes[i, 0].grid(True)

        axes[i, 1].plot(time_steps[timestep_action], kd[:-1, i])  # kd of previous timestep
        axes[i, 1].set_title(f'Joint {i+1} Kd Over Time')
        axes[i, 1].set_xlabel('Time Step')
        axes[i, 1].set_ylabel('Kd Value')
        axes[i, 1].grid(True)

        axes[i, 2].plot(time_steps[timestep_action], residual[:-1, i])
        axes[i, 2].set_xlabel('Time Step')
        axes[i, 2].set_ylabel('residual')
        axes[i, 2].grid(True)
        axes[i, 2].legend()

        axes[i, 3].plot(time_steps, des_pos[:, i], label='Desired Position')
        axes[i, 3].plot(time_steps, joint_pos[:, i], label='Joint Position')
        axes[i, 3].set_xlabel('Time Step')
        axes[i, 3].set_ylabel('Position')
        axes[i, 3].grid(True)
        axes[i, 3].legend()

        

    handles, labels = axes[0, 3].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    print(1)


# Example usage
# file_path = 'real_policy_execute_<timestamp>.pkl'  # Replace with your file path
data = load_data(file_path)

import os
import cv2

# Load your data (assuming this is already done in your code)
obs = data['obs']
imgs = data['imgs']

#####

# import numpy as np

# # Example data: Replace this with your actual sequence of keypoints
# keypoints = obs[:, 12:28]

# cur_idx = 71
# curr_keypoints = keypoints[cur_idx].reshape(8, 2)
# prev_keypoints = keypoints[cur_idx-1].reshape(8, 2)

# keyp_diff = np.max(np.linalg.norm((curr_keypoints-prev_keypoints), axis=1))
# print(keyp_diff)

# # # Step 1: Reshape keypoints to (N, 8, 2) for easier processing
# keypoints_reshaped = keypoints.reshape(-1, 8, 2)

# # Step 2: Compute the Euclidean distances for each of the 8 keypoints
# differences = np.linalg.norm(np.diff(keypoints_reshaped, axis=0), axis=2)

# # Step 3: Define the difference as the largest distance among the 8 keypoints
# max_differences = np.max(differences, axis=1)  # Shape (N-1,)

# # Step 4: Determine the threshold for anomalies
# # mu = np.mean(max_differences)
# # sigma = np.std(max_differences)
# # threshold = mu + 3 * sigma

# threshold = 60.

# # Step 5: Detect anomalies
# anomalies = np.where(max_differences > threshold)[0]

# # Step 6: Replace anomalies with the previous frame's keypoints
# for anomaly_idx in anomalies:
#     keypoints[anomaly_idx + 1] = keypoints[anomaly_idx]

# print(f"Threshold: {threshold}")
# print(f"Anomalies detected at indices: {anomalies}")




#####


# Create a directory to save keypoint visualizations if it doesn't exist
os.makedirs('keypoint_vis', exist_ok=True)

# Set up video writer
height, width, _ = imgs[0].shape  # Get dimensions from the first image
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
fps = 20  # Frames per second
out = cv2.VideoWriter('keypoint_vis/output_video_241028_dark.mp4', fourcc, fps, (width, height))

# Loop through each observation and corresponding image
for obs_idx in range(obs.shape[0]):
    obser = obs[obs_idx]
    image = imgs[obs_idx]
    cur_keypoints = obser[12:28]
    goal_keypoints = obser[28:44]

    # Draw current keypoints (red)
    for idx in range(8):
        image = cv2.circle(image, (int(cur_keypoints[2*idx]), int(cur_keypoints[2*idx+1])), 3, (255 * (1.-idx / 8.), 0, 255 * idx / 8.), -1)

    # Draw goal keypoints (blue)
    for idx in range(8):
        image = cv2.circle(image, (int(goal_keypoints[2*idx]), int(goal_keypoints[2*idx+1])), 3, (255 * (1.-idx / 8.), 0, 255 * idx / 8.), -1)

    # Save each frame as an image (optional, you can comment this out if not needed)
    cv2.imwrite(f'keypoint_vis/image_{obs_idx}.png', image)

    # Write the frame to the video
    out.write(image)

    print(f"Processed frame {obs_idx}")

# Release the video writer
out.release()

print("Video saved as 'keypoint_vis/output_video.mp4'")

# Convert to DataFrame for easier handling
df = pd.DataFrame(data['act'])

# Define the max values for the graphs
max_values = [-0.25, -0.3, -0.45, -0.40, -0.30, -0.30]+[0.1]*6+[0.0]*6
min_values = [0.25, 0.3, 0.45, 0.40, 0.30, 0.30]+[1.8, 2.2, 1.6, 1.6, 0.7, 0.7] + [0.06, 0.08, 0.1, 0.1, 0.06, 0.06]

# Plot for [:, :3] with axis ±0.6
plt.figure(figsize=(10, 5))
for i in range(6):
    plt.plot(df.index, df.iloc[:, i], label=f'Column {i+1}')
    plt.axhline(y=max_values[i], color='r', linestyle='--')
    plt.axhline(y=min_values[i], color='r', linestyle='--')
    plt.legend()
    plt.title(f'Graph for joint {i} target residual')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.grid(True)
    plt.show()

# Plots for kp values [:, 6:12] with specific max values
for i in range(6, 12):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df.iloc[:, i], label=f'Column {i-5}')
    plt.axhline(y=max_values[i], color='r', linestyle='--')
    plt.axhline(y=min_values[i], color='r', linestyle='--')
    plt.legend()
    plt.title(f'Graph for KP of joint {i}')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.grid(True)
    plt.show()

print(1)

# Extract kps and kds from observation
# kps, kds = extract_kps_kds(data['obs'])
obs = data['obs']
res = obs[:, 45:51]
kps = obs[:, 51:57]
kds = obs[:, 57:]

# Extract other relevant data
time_steps = data['time']
des_pos = data['des_pos']
joint_pos = data['joint_pos']
des_tau = data['des_tau_list']
joint_tau = data['joint_tau']

# Ensure time_steps, des_pos, and joint_pos have the same length
assert len(time_steps) == des_pos.shape[0] == joint_pos.shape[0], "Time steps and data lengths do not match"

# Define joint position limits
joint_pos_lower_limit = np.array([-1.7453, -0.6981, -1.7453, -0.7854, -2.9671, -0.6109])
joint_pos_upper_limit = np.array([1.1345, 0.8727, 0.8727, 1.3090, 2.9671, 3.7525])
pos_limits = {'lower': joint_pos_lower_limit, 'upper': joint_pos_upper_limit}

# Visualize the data
visualize_motor_data(time_steps, kps, kds, des_pos, joint_pos, res, pos_limits)