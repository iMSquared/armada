from simulation.im2gym.tasks.domain import Domain, gen_keypoints, compute_projected_points
from isaacgym.torch_utils import quat_from_euler_xyz, normalize
from torch.distributions.bernoulli import Bernoulli
from torch.distributions import Uniform
from simulation.im2gym.algos.utils import *
from typing import Tuple, Dict, List
import numpy as np
import torch


def generatorBuilder(name, **kwargs):
    if name=="Card":
        return cardGenerator(**kwargs)
    elif name=="Bump":
        return bumpGenerator(**kwargs)
    elif name=="Snatching":
        return SnatchingGenerator(**kwargs)
    elif name=="Throw":
        return ThrowGenerator(**kwargs)
    elif name=="Throw_left":
        return ThrowLeftGenerator(**kwargs)
    else:
        raise Exception(f"no available genrator for {name}")



class sampleGenerator:

    def __init__(self, writer, map, IK_query_size, device) -> None:
        """
            writer: tensorboard writer 
            map: mapping object
        """
        self.writer=writer
        self.map=map
        self.IK_query_size=IK_query_size
        self.device=device

    def set_env(self, env: Domain):
        self.env = env

    def sample(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Need task specific implementaion")

class cardGenerator(sampleGenerator):

    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs["writer"], kwargs["map"], kwargs["IK_query_size"], kwargs["device"])
        geometry=kwargs.pop("geometry")
        self.constant = geometry["constant"]

        xmin=geometry["xmin"]
        xmax=geometry["xmax"]

        ymin=geometry["ymin"]
        ymax=geometry["ymax"]

        self.table_x=geometry["table"]["x"]
        self.table_y=geometry["table"]["y"]
        self.table_z=geometry["table"]["z"]

        cardDims=geometry["object"]
        cardDims=[cardDims["width"], cardDims["length"], cardDims["height"]]
        cardLength=np.sqrt(cardDims[0]**2+cardDims[1]**2)/2
        self.card_height=cardDims[2]
        table_dims=geometry["table"]
        self.table_dims=[table_dims["width"], table_dims["length"], table_dims["height"]]

        self.xsampler = Uniform((cardLength +xmin), (xmax - cardLength))
        self.ysampler = Uniform((cardLength +ymin), (ymax - cardLength))
        

    def sample(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.constant = False
        # size = int(size/2)

        if self.constant: return self.constant_sample(size)
        # x_O = 0.5*torch.ones((size, 1), dtype=torch.float, device=self.device)*self.table_dims[0] - self.table_dims[0] / 2 + self.table_x 
        # y_O = 0.5*torch.ones((size, 1), dtype=torch.float, device=self.device)*self.table_dims[1] - self.table_dims[1] / 2 + self.table_y
        # x_O = self.xsampler.sample((size, 1)).to(self.device)
        x_O = 0.12 + 0.36 * torch.rand((size, 1), device=self.device)
        y_O = 0.02 + 0.38 * torch.rand((size, 1), device=self.device)
        # x_G = torch.rand((size, 1), dtype=torch.float, device=self.device)*self.table_dims[0] - self.table_dims[0] / 2 + self.table_x 
        x_G = 0.12 + 0.36 * torch.rand((size, 1), device=self.device)
        y_G = 0.02 + 0.38  * torch.rand((size, 1), device=self.device)
        z = (self.table_z + self.table_dims[2]/2. + self.card_height / 2) * torch.ones((size, 1), dtype=torch.float, device=self.device)
        roll = torch.zeros((size * 2), dtype=torch.float, device=self.device)
        pitch = torch.zeros((size * 2), dtype=torch.float, device=self.device)
        yaw = 2 * np.pi * torch.rand((size * 2), dtype=torch.float, device=self.device)
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        T_O = torch.cat([x_O, y_O, z, quat[:size]], -1)
        T_G = torch.cat([x_G, y_G, z, quat[size:]], -1)
        return T_O, T_G
    
    def constant_sample(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_O = 0.5*torch.ones((size, 1), dtype=torch.float, device=self.device)*self.table_dims[0] - self.table_dims[0] / 2 + self.table_x 
        y_O = 0.5*torch.ones((size, 1), dtype=torch.float, device=self.device)*self.table_dims[1] - self.table_dims[1] / 2 + self.table_y
        x_G = 0.15*torch.ones((size, 1), dtype=torch.float, device=self.device)*self.table_dims[0] - self.table_dims[0] / 2 + self.table_x 
        y_G = 0.1*torch.ones((size, 1), dtype=torch.float, device=self.device)*self.table_dims[1] - self.table_dims[1] / 2 + self.table_y
        z = (self.table_z + self.table_dims[2]/2. + self.card_height / 2) * torch.ones((size, 1), dtype=torch.float, device=self.device)
        roll = torch.zeros((size * 2), dtype=torch.float, device=self.device)
        pitch = torch.zeros((size * 2), dtype=torch.float, device=self.device)
        yaw = torch.zeros((size * 2), dtype=torch.float, device=self.device) # 2 * np.pi * torch.rand((size * 2), dtype=torch.float, device=self.device)
        yaw[size:] = 2 * np.pi*torch.ones((size * 1), dtype=torch.float, device=self.device) # 2 * np.pi * torch.rand((size * 2), dtype=torch.float, device=self.device)
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        T_O = torch.cat([x_O, y_O, z, quat[:size]], -1)
        T_G = torch.cat([x_G, y_G, z, quat[size:]], -1)
        return T_O, T_G

class bumpGenerator(sampleGenerator):

    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs["writer"], kwargs["map"], kwargs["IK_query_size"], kwargs["device"])

        geometry=kwargs.pop("geometry")
        xmin=geometry["xmin"]
        xmax=geometry["xmax"]

        ymin=geometry["ymin"]
        ymax=geometry["ymax"]

        bump=geometry["obstacle"]
        self.bumpDims=[bump["width"], bump["length"], bump["height"]]
        self.bump_pose = [bump["x"], bump["y"], bump["z"]]

        boxDims=geometry["object"]
        self.boxDims=[boxDims["width"], boxDims["length"], boxDims["height"]]
        self.boxLength=np.sqrt(self.boxDims[0]**2+self.boxDims[1]**2)/2
        table_dims=geometry["table"]
        self.table_pose_ = [table_dims["x"], table_dims["y"], table_dims["z"]]
        self.table_dims=[table_dims["width"], table_dims["length"], table_dims["height"]/2 + table_dims["z"]]
        self.safety_margin = 0.01
        self.both_side = geometry["both_side"]
        if self.both_side:
            self.left_right_sampler = Bernoulli(0.5)

        self.xsampler = Uniform((self.boxLength + xmin), (xmax - self.boxLength))
        self.ysampler = Uniform((self.bumpDims[1] / 2 + self.boxLength), (self.table_dims[1] / 2 - self.boxLength))

        self.Gysampler = Uniform(
            low=0.,
            high=(self.bump_pose[1] + self.bumpDims[1]/2.)
        )

        init_goal: Dict[str, List[float]] = geometry.get('init_goal', None)
        self.constant = init_goal is not None
        if self.constant:
            self.T_O = init_goal.get('T_O', None)
            self.T_G = init_goal.get('T_G', None)

    def sample(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:

        x_O = self.xsampler.sample((size, 1)).to(self.device)
        y_O = self.Gysampler.sample((size, 1)).to(self.device)
        onbump = (y_O >= (self.bump_pose[1] - self.bumpDims[1]/2.))
        z_O = torch.zeros_like(x_O)
        z_O[onbump] = self.bump_pose[2] + self.bumpDims[2] / 2 + self.boxDims[2] / 2
        z_O[~onbump] = self.table_dims[2] + self.boxDims[2] / 2
        y_O[~onbump] -= (self.safety_margin + self.boxLength)

        # Define the invalid range and the total valid range
        max_y = self.bump_pose[1] + self.bumpDims[1]/2.
        invalid_low = 0.1
        invalid_high = 0.2
        gap = invalid_high - invalid_low
        valid_range = max_y - gap

        # Sample y_G avoiding the invalid range
        y_G = torch.rand(size, 1).to(self.device) * valid_range
        y_G[y_G >= invalid_low] += gap

        x_G = self.xsampler.sample((size, 1)).to(self.device)
        onbump = (y_G >= (self.bump_pose[1] - self.bumpDims[1]/2.))
        z_G = torch.zeros_like(z_O)
        z_G[onbump] = self.bump_pose[2] + self.bumpDims[2] / 2 + self.boxDims[2] / 2
        z_G[~onbump] = self.table_dims[2] + self.boxDims[2] / 2
        y_G[~onbump] -= (self.safety_margin + self.boxLength)

        if self.both_side:
            left_right = self.left_right_sampler.sample((size,))
            right = (left_right < 0.5)
            y_O[right] = 2*self.bump_pose[1] - y_O[right]

            left_right = self.left_right_sampler.sample((size,))
            right = (left_right > 0.5)
            y_G[right] = 2*self.bump_pose[1] - y_G[right]

        roll = (np.pi / 2) * torch.randint(0, 4, (size * 2,), device=self.device)
        pitch = (np.pi / 2) * torch.randint(0, 4, (size * 2,), device=self.device)
        yaw = (2 * np.pi) * torch.rand((size * 2), dtype=torch.float, device=self.device)
        quat = quat_from_euler_xyz(roll, pitch, yaw)

        quat[:] = torch.tensor([0., 0., 0., 1], device=self.device)[None, :]

        # y_G = 0.55*torch.ones((size, 1), dtype=torch.float, device=self.device)*self.table_dims[1] - self.table_dims[1] / 2 + self.table_pose_[1]
        # z_G = (self.table_dims[2] + self.boxDims[2] / 2) * torch.ones((size, 1), dtype=torch.float, device=self.device)

        T_O = torch.cat([x_O, y_O, z_O, quat[:size]], -1)
        # T_G = torch.cat([x[size:], y_G, z_G, quat[size:]], -1)
        T_G = torch.cat([x_G, y_G, z_G, quat[size:]], -1)

        if self.constant:
            fixed_T_O, fixed_T_G = self.constant_sample(size)
            T_O = fixed_T_O if fixed_T_O is not None else T_O
            T_G = fixed_T_G if fixed_T_G is not None else T_G

        return T_O, T_G
    
    def constant_sample(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        T_O, T_G = None, None
        if self.T_G is not None:
            T_G = torch.tensor(self.T_G, device=self.device).repeat((size, 1))
        if self.T_O is not None:
            T_O = torch.tensor(self.T_O, device=self.device).repeat((size, 1))

        return T_O, T_G



class SnatchingGenerator(sampleGenerator):

    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs["writer"], kwargs["map"], kwargs["IK_query_size"], kwargs["device"])

        geometry=kwargs.pop("geometry")
        xmin=geometry["xmin"]
        xmax=geometry["xmax"]
        ymin=geometry["ymin"]
        ymax=geometry["ymax"]

        basket=geometry["basket"]
        self.basket_pose = [basket["x"], basket["y"], basket["z"]]

        boxDims=geometry["object"]
        self.boxDims=[boxDims["width"], boxDims["length"], boxDims["height"]]
        self.boxLength=np.sqrt(self.boxDims[0]**2+self.boxDims[1]**2)/2
        table_dims=geometry["table"]
        self.table_pose_ = [table_dims["x"], table_dims["y"], table_dims["z"]]
        self.table_dims=[table_dims["width"], table_dims["length"], table_dims["height"]/2 + table_dims["z"]]
        self.safety_margin = 0.01

        self.xsampler = Uniform((self.boxLength +xmin), (xmax - self.boxLength))
        self.ysampler = Uniform((self.boxLength +ymin), (ymax - self.boxLength))
        

    def sample(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = 0.5*torch.ones((size, 1), dtype=torch.float, device=self.device)*self.table_dims[0] - self.table_dims[0] / 2 + self.table_pose_[0]
        x_G = 0.9*torch.ones((size, 1), dtype=torch.float, device=self.device)*self.table_dims[0] - self.table_dims[0] / 2 + self.table_pose_[0]
        y_O = 0.5*torch.ones((size, 1), dtype=torch.float, device=self.device)*self.table_dims[1] - self.table_dims[1] / 2 + self.table_pose_[1]
        z_O = (self.table_dims[2] + self.boxDims[2] / 2) * torch.ones((size, 1), dtype=torch.float, device=self.device)

        roll = (np.pi / 2) * torch.randint(0, 4, (size * 2,), device=self.device)*0
        pitch = (np.pi / 2) * torch.randint(0, 4, (size * 2,), device=self.device)*0
        yaw = (2 * np.pi) * torch.rand((size * 2), dtype=torch.float, device=self.device)*0
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        y_G = 1.46*torch.ones((size, 1), dtype=torch.float, device=self.device)*self.table_dims[1] - self.table_dims[1] / 2 + self.table_pose_[1]

        z_G = (self.boxDims[2] / 2) * torch.ones((size, 1), dtype=torch.float, device=self.device) * 0
        T_O = torch.cat([x[:size], y_O, z_O, quat[:size]], -1)
        T_G = torch.cat([x_G, y_G, z_G, quat[size:]], -1)
        return T_O, T_G
    


class ThrowGenerator(sampleGenerator):

    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs["writer"], kwargs["map"], kwargs["IK_query_size"], kwargs["device"])

        geometry:Dict = kwargs.pop("geometry")
        xmin, xmax = geometry["xmin"], geometry["xmax"]
        ymin, ymax = geometry["ymin"], geometry["ymax"]
        zmin, zmax = geometry["zmin"], geometry["zmax"]


        boxDims=geometry["object"]
        self.boxDims=[boxDims["length"], boxDims["width"], boxDims["height"]]

        self.safety_margin = 0.01

        init_goal: Dict[str, List[float]] = geometry.get('init_goal', None)
        self.constant = init_goal is not None or init_goal['T_O'] is not None

        if self.constant:
            self.T_O = init_goal.get('T_O', None)
        
        else:
            self.xsampler = Uniform((xmin + self.boxDims[0]), (xmax - self.boxDims[0]))
            self.ysampler = Uniform((ymin + self.boxDims[1]), (ymax - self.boxDims[1]))
            self.zsampler = Uniform((zmin + self.boxDims[2]), (zmax - self.boxDims[2]))


    def sample(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        T_G is dummy data for throwing, because we want to throw as far as we can.
        TODO: we should make the robot start with the cube on the hand.
        '''

        if self.constant:
            x_O, y_O, z_O = self.T_O[:3]
            x_O = x_O*torch.ones((size, 1)).to(self.device)
            y_O = y_O*torch.ones((size, 1)).to(self.device)
            z_O = z_O*torch.ones((size, 1)).to(self.device)
        else:
            x_O = self.xsampler.sample((size, 1)).to(self.device)
            y_O = self.ysampler.sample((size, 1)).to(self.device)
            z_O = self.zsampler.sample((size, 1)).to(self.device)

        roll = (np.pi / 2) * torch.randint(0, 4, (size * 2,), device=self.device)
        pitch = (np.pi / 2) * torch.randint(0, 4, (size * 2,), device=self.device)
        yaw = (2 * np.pi) * torch.rand((size * 2), dtype=torch.float, device=self.device)
        quat = quat_from_euler_xyz(roll, pitch, yaw)

        # TODO: hotfixed quat randomized error.
        quat[:, :3] = 0.
        quat[:, 3] = 1.

        T_O = torch.cat([x_O, y_O, z_O, quat[:size]], -1)
        T_G = torch.cat([torch.ones_like(x_O)*100, torch.ones_like(y_O)*100, torch.ones_like(z_O)*100, quat[:size]], -1)

        return T_O, T_G

class ThrowLeftGenerator(sampleGenerator):

    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs["writer"], kwargs["map"], kwargs["IK_query_size"], kwargs["device"])

        geometry:Dict = kwargs.pop("geometry")
        xmin, xmax = geometry["xmin"], geometry["xmax"]
        ymin, ymax = geometry["ymin"], geometry["ymax"]
        zmin, zmax = geometry["zmin"], geometry["zmax"]


        boxDims=geometry["object"]
        self.boxDims=[boxDims["length"], boxDims["width"], boxDims["height"]]

        self.safety_margin = 0.01

        init_goal: Dict[str, List[float]] = geometry.get('init_goal', None)
        self.constant = init_goal is not None or init_goal['T_O'] is not None

        if self.constant:
            self.T_O = init_goal.get('T_O', None)
            self.T_G = init_goal.get('T_G', None)
        
        else:
            self.xsampler = Uniform((xmin + self.boxDims[0]), (xmax - self.boxDims[0]))
            self.ysampler = Uniform((ymin + self.boxDims[1]), (ymax - self.boxDims[1]))
            self.zsampler = Uniform((zmin + self.boxDims[2]), (zmax - self.boxDims[2]))


    def sample(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        T_G is dummy data for throwing, because we want to throw as far as we can.
        TODO: we should make the robot start with the cube on the hand.
        '''

        if self.constant:
            T_O = (torch.tensor(self.T_O)[None, :]*torch.ones((size, 7))).to(self.device)
            T_G = (torch.tensor(self.T_G)[None, :]*torch.ones((size, 7))).to(self.device)
        else:
            x_O = self.xsampler.sample((size, 1)).to(self.device)
            y_O = self.ysampler.sample((size, 1)).to(self.device)
            z_O = self.zsampler.sample((size, 1)).to(self.device)
            x_G = self.xsampler.sample((size, 1)).to(self.device)
            y_G = self.ysampler.sample((size, 1)).to(self.device)
            z_G = self.zsampler.sample((size, 1)).to(self.device)

            roll = (np.pi / 2) * torch.randint(0, 4, (size * 2,), device=self.device)
            pitch = (np.pi / 2) * torch.randint(0, 4, (size * 2,), device=self.device)
            yaw = (2 * np.pi) * torch.rand((size * 2), dtype=torch.float, device=self.device)
            quat = quat_from_euler_xyz(roll, pitch, yaw)

            # TODO: hotfixed quat randomized error.
            quat[:, :3] = 0.
            quat[:, 3] = 1.

            T_O = torch.cat([x_O, y_O, z_O, quat[:size]], -1)
            T_G = torch.cat([torch.ones_like(x_O)*100, torch.ones_like(y_O)*100, torch.ones_like(z_O)*100, quat[:size]], -1)

        return T_O, T_G