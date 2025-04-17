from simulation.im2gym.tasks.domain import Domain
from simulation.im2gym import CuboidalObject
from typing import List, Dict
from isaacgym import gymapi
import numpy as np
import torch
from simulation.utils.torch_jit_utils import quat_diff_rad


class Bump(Domain):
    def __init__(self, cfg, sim_device, graphics_device_id, headless, use_state=False, gym=None, record=False):
        super().__init__(cfg, sim_device, graphics_device_id, headless, use_state, gym=gym, record=record)

    def _set_table_dimension(self, cfg):
        table_params = self.cfg["env"]["geometry"]["table"]
        # table_dims = gymapi.Vec3(0.4, 0.5, 0.6)
        table_dims = gymapi.Vec3(table_params["width"], table_params["length"], table_params["height"])
        table_pose = gymapi.Transform()
        # table_pose.p = gymapi.Vec3(0.5, 0.0, 0.2)
        table_pose.p = gymapi.Vec3(table_params["x"], table_params["y"], table_params["z"])
        self.table_dims = table_dims
        self.table_pose = table_pose
        self.bump_dims = self.cfg["env"]["geometry"]["obstacle"]

        self.use_bolt = cfg["env"]["geometry"]["bolt"]
        if self.use_bolt:
            bolt_pose = gymapi.Transform()
            bolt_pose.p = gymapi.Vec3(0.3+0.053, 0.25-0.02, 0.4)
            bolt_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.5*np.pi, 0.0)
            self.bolt_1_pose = bolt_pose

            bolt_pose = gymapi.Transform()
            bolt_pose.p = gymapi.Vec3(0.7-0.053, -0.25+0.02, 0.4)
            bolt_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.5*np.pi, 0.0)
            self.bolt_2_pose = bolt_pose

    def _get_table_prim_names(self) -> List[str]:
        prim_names = ["table", "bump"]
        return prim_names

    def _set_object_dimension(self, object_dims) -> CuboidalObject:
        return CuboidalObject((object_dims["width"], object_dims["length"], object_dims["height"]))

    def _create_table(self, env_ptr, env_index: int, actor_indices: Dict[str, List[int]]):
        table_handle = self.gym.create_actor(env_ptr, self.asset_handles["table"], self.table_pose, "table", env_index, 1, 1)
        table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
        actor_indices["table"].append(table_idx)
        bump_pose = gymapi.Transform()
        bump_pose.p = gymapi.Vec3(self.bump_dims["x"],self.bump_dims["y"],self.bump_dims["z"])
        bump_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        bump_handle = self.gym.create_actor(env_ptr, self.asset_handles["bump"], bump_pose, "bump", env_index, 1, 1)
        bump_idx = self.gym.get_actor_index(env_ptr, bump_handle, gymapi.DOMAIN_SIM)
        actor_indices["bump"].append(bump_idx)

        if self.use_bolt:
            bolt_color = gymapi.Vec3(0.54, 0.57, 0.59)
            bolt_1_handle = self.gym.create_actor(env_ptr, self.asset_handles["bolt_1"], self.bolt_1_pose, "bolt_1", env_index, 1, 1)
            bolt_2_handle = self.gym.create_actor(env_ptr, self.asset_handles["bolt_2"], self.bolt_2_pose, "bolt_2", env_index, 1, 1)
            self.gym.set_rigid_body_color(env_ptr, bolt_1_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, bolt_color)
            self.gym.set_rigid_body_color(env_ptr, bolt_2_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, bolt_color)
            bolt_1_idx = self.gym.get_actor_index(env_ptr, bolt_1_handle, gymapi.DOMAIN_SIM)
            bolt_2_idx = self.gym.get_actor_index(env_ptr, bolt_2_handle, gymapi.DOMAIN_SIM)
            actor_indices["bolt_1"].append(bolt_1_idx)
            actor_indices["bolt_2"].append(bolt_2_idx)

    def _create_object(self, env_ptr, env_index: int, actor_indices: Dict[str, List[int]]):
        object_handle = self.gym.create_actor(env_ptr, self.asset_handles["object"], gymapi.Transform(), "object", env_index, 0, 2)
        object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
        actor_indices["object"].append(object_idx)

    def _max_dist_btw_obj_and_goal(self) -> torch.Tensor:
        return np.sqrt(self.table_dims.x ** 2 + self.table_dims.y ** 2) * torch.ones(1, dtype=torch.float32, device=self.device)

    def _define_table_asset(self):
        """ Define Gym asset for table.
        """
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        table_asset_options.thickness = 0.001
        table_asset = self.gym.create_box(self.sim, self.table_dims.x, self.table_dims.y, self.table_dims.z, table_asset_options)
        table_props = self.gym.get_asset_rigid_shape_properties(table_asset)
        for p in table_props:
            p.friction = self.cfg['env']['properties']['table']['friction']
            p.restitution = self.cfg['env']['properties']['table']['restitution']
        self.gym.set_asset_rigid_shape_properties(table_asset, table_props)
        self.asset_handles["table"] = table_asset

        bump_options = gymapi.AssetOptions()
        bump_options.disable_gravity = True
        bump_options.fix_base_link = True
        bump_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        bump_options.thickness = 0.001
        bump_asset = self.gym.create_box(self.sim, self.bump_dims["width"], self.bump_dims["length"], self.bump_dims["height"], bump_options)
        bump_props = self.gym.get_asset_rigid_shape_properties(bump_asset)
        for p in bump_props:
            p.friction = self.cfg['env']['properties']['bump']['friction']
            p.restitution = self.cfg['env']['properties']['bump']['restitution']
        self.gym.set_asset_rigid_shape_properties(bump_asset, bump_props)
        self.asset_handles["bump"] = bump_asset

        if self.use_bolt:
            bolt_asset_options = gymapi.AssetOptions()
            bolt_asset_options.disable_gravity = True
            bolt_asset_options.fix_base_link = True
            bolt_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            bolt_asset_options.thickness = 0.001
            
            bolt_asset = self.gym.create_capsule(self.sim,  0.0055, 0.008, bolt_asset_options)
            bolt_props = self.gym.get_asset_rigid_shape_properties(bolt_asset)

            for p in bolt_props:
                p.friction = self.cfg['env']['properties']['bolt']['friction']
                p.restitution = self.cfg['env']['properties']['bolt']['restitution']
            
            self.gym.set_asset_rigid_shape_properties(bolt_asset, bolt_props)

            self.asset_handles["bolt_1"] = bolt_asset
            self.asset_handles["bolt_2"] = bolt_asset

    def _define_object_asset(self):
        """ Define Gym asset for object.
        """
        # define object asset
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = False
        object_asset_options.thickness = 0.001
        object_asset_options.flip_visual_attachments = True
        obj_density = self.cfg["env"]["geometry"]["object"]["density"]
        object_asset_options.density = obj_density
        # object_asset_options.linear_damping = 0.1
        object_asset = self.gym.load_asset(self.sim, self._assets_dir, 'urdf/Cube/im2_cube.urdf')
        object_props = self.gym.get_asset_rigid_shape_properties(object_asset)
        for p in object_props:
            p.friction = self.cfg['env']['properties']['object']['friction']
            p.restitution = self.cfg['env']['properties']['object']['restitution']
        self.gym.set_asset_rigid_shape_properties(object_asset, object_props)
        
        return object_asset

    def _check_failure(self) -> torch.Tensor:
        failed_reset = torch.le(self._object_state_history[0][:, 2], (0.8 * self.table_dims.z))
        return failed_reset

    def _check_success(self) -> torch.Tensor:

        # TODO: implement position_only option here using self.cfg["env"]["reward_terms"]["position_only"]

        delta = self._object_state_history[0][:, 0:3] - self._object_goal_poses_buf[:, 0:3]
        dist = torch.norm(delta, p=2, dim=-1)
        goal_position_reached = torch.le(dist, self.cfg["env"]["reward_terms"]["object_dist"]["th"])

        if self.cfg["env"]["reward_terms"]["position_only"]: return goal_position_reached

        quat_a = self._object_state_history[0][:, 3:7]
        quat_b = self._object_goal_poses_buf[:, 3:7]
        angles = quat_diff_rad(quat_a, quat_b)
        goal_rotation_reached = torch.le(torch.abs(angles), self.cfg["env"]["reward_terms"]["object_rot"]["th"])
        goal_reached = torch.logical_and(goal_rotation_reached, goal_position_reached)
        return goal_reached
