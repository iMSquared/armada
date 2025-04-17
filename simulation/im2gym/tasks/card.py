from simulation.im2gym.tasks.domain import Domain
from simulation.im2gym import CuboidalObject
from typing import List, Dict
from isaacgym import gymapi
import numpy as np
import torch
from simulation.utils.torch_jit_utils import quat_diff_rad


class Card(Domain):
    def __init__(self, cfg, sim_device, graphics_device_id, headless, use_state=False, gym=None):
        super().__init__(cfg, sim_device, graphics_device_id, headless, use_state, gym=gym)

    def _set_table_dimension(self, cfg):
        table_params = cfg["env"]["geometry"]["table"]
        table_dims = gymapi.Vec3(table_params["width"], table_params["length"], table_params["height"])
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(table_params["x"], table_params["y"], table_params["z"])
        self.table_dims = table_dims
        self.table_pose = table_pose

    def _get_table_prim_names(self) -> List[str]:
        prim_names = ["table"]
        return prim_names

    def _set_object_dimension(self, object_dims) -> CuboidalObject:
        return CuboidalObject((object_dims["width"], object_dims["length"], object_dims["height"]))

    def _create_table(self, env_ptr, env_index: int, actor_indices: Dict[str, List[int]]):

        # param6: bitwise filter for elements in the same collisionGroup to mask off collision
        # originally 1
        table_handle = self.gym.create_actor(env_ptr, self.asset_handles["table"], self.table_pose, "table", env_index, 0, 1)
        table_color = gymapi.Vec3(0.54, 0.57, 0.59)
        self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, table_color)
        table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
        actor_indices["table"].append(table_idx)

    def _create_object(self, env_ptr, env_index: int, actor_indices: Dict[str, List[int]]):
        object_handle = self.gym.create_actor(env_ptr, self.asset_handles["object"], gymapi.Transform(), "object", env_index, 0, 2)
        object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
        actor_indices["object"].append(object_idx)

    def _max_dist_btw_obj_and_goal(self) -> torch.Tensor:
        return np.sqrt(self.table_dims.x ** 2 + self.table_dims.y ** 2) * torch.ones(1, dtype=torch.float32, device=self.device)

    def _define_table_asset(self):
        """ Define Gym asset for table. This function returns nothing.
        """
        # define table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        table_asset_options.thickness = 0.001
        # load table asset
        table_asset = self.gym.create_box(self.sim, self.table_dims.x, self.table_dims.y, self.table_dims.z, table_asset_options)
        # set table properties
        table_props = self.gym.get_asset_rigid_shape_properties(table_asset)
        # sensor_pose = gymapi.Transform()
        # _=self.gym.create_asset_force_sensor(table_asset, 0, sensor_pose)
        # iterate over each mesh
        for p in table_props:
            p.friction = 0.5
            # p.restitution = 0.5
        self.gym.set_asset_rigid_shape_properties(table_asset, table_props)

        self.asset_handles["table"] = table_asset

    def _define_object_asset(self):
        """ Define Gym asset for object.
        """
        # define object asset
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = False
        object_asset_options.thickness = 0.001
        object_asset_options.density = self.cfg["env"]["geometry"]["object"]["density"]
        object_asset = self.gym.load_asset(self.sim, self._assets_dir, 'urdf/Panda/Coloredcard.urdf', object_asset_options)
        object_props = self.gym.get_asset_rigid_shape_properties(object_asset)
        # sensor_pose = gymapi.Transform()
        # _=self.gym.create_asset_force_sensor(object_asset, 0, sensor_pose)
        for p in object_props:
            p.friction = 0.5
            # p.restitution = 0.8
        self.gym.set_asset_rigid_shape_properties(object_asset, object_props)

        return object_asset

    def _check_failure(self) -> torch.Tensor:
        failed_envs = torch.le(self._object_state_history[0][:, 2], (0.8 * self.table_dims.z))
        return failed_envs

    def _check_success(self) -> torch.Tensor:
        delta = self._object_state_history[0][:, 0:3] - self._object_goal_poses_buf[:, 0:3]
        dist = torch.norm(delta, p=2, dim=-1)
        goal_position_reached = torch.le(dist, self.cfg["env"]["reward_terms"]["object_dist"]["th"])
        quat_a = self._object_state_history[0][:, 3:7]
        quat_b = self._object_goal_poses_buf[:, 3:7]
        angles = quat_diff_rad(quat_a, quat_b)
        goal_rotation_reached = torch.le(torch.abs(angles), self.cfg["env"]["reward_terms"]["object_rot"]["th"])
        goal_reached = torch.logical_and(goal_rotation_reached, goal_position_reached)
        return goal_reached
