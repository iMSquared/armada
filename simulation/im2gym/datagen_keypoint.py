from pathlib import Path
BASEDIR = Path(__file__).parent.parent.parent
import sys
if str(BASEDIR) not in sys.path:
    sys.path.append(str(BASEDIR))

import isaacgym
from isaacgym.torch_utils import normalize
from torch.multiprocessing import Process, Event
import torch
from simulation.utils.generator import generatorBuilder, sampleGenerator
from simulation.utils.torch_jit_utils import unscale_transform
from simulation.utils.reformat import omegaconf_to_dict
from simulation.utils.misc import set_seed
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import hydra
from simulation.im2gym.algos.policy import Model
from simulation.im2gym.algos.utils import load_checkpoint
from simulation.im2gym.tasks import immgym_task_map
from simulation.im2gym.tasks.domain import Domain
from typing import Tuple, Dict, List, Optional
from pathlib import Path
from tqdm import tqdm
import numpy as np
import shutil
import cv2
from simulation.im2gym.algos.set_initial_position import PreCalculatedIK, setFixedInitPosition


def imageWriter(
    run, done, numIteration: int, numEnvs: int, 
    images: torch.Tensor, segmentation: torch.Tensor, dataPath: Path, 
    startIndex: int, numImages: int
):
    startNumber = startIndex
    for _ in range(numIteration):
        run.wait()
        run.clear()
        for j in range(startIndex, (startIndex + numImages)):
            fileNumber = startNumber + j - startIndex + 1
            cv2.imwrite(
                str(dataPath.joinpath('rgb', f'{fileNumber}.png')), 
                cv2.cvtColor(images[j].numpy(), cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(
                str(dataPath.joinpath('segmentation', f'{fileNumber}.png')), 
                segmentation[j].numpy()
            )
        startNumber += numEnvs
        done.set()


class DataGenerator:
    def __init__(
        self, cfg, env: Domain, model: Model, 
        datasetPath: str, size: int = int(1e6), validation: bool = False
    ):
        self.env = env
        self.model = model
        self.size = size
        env_list = ["Card", "HiddenCard", "Flip_chamfer", "Hole", "Reorientation", "Bookshelf", "Bump", "Hole_wide"]
        env_dict = dict(zip(env_list, range(len(env_list))))
        self.cfg = cfg
        self.rl_device = cfg["rl_device"]
        config = self.cfg["train"]["params"]["config"]
        self.numEnvs: int = config["num_actors"]
        self.IK_query_size = 22000
        self.generator = generatorBuilder(config["name"], writer=None,
            map=None, IK_query_size=self.IK_query_size, device=self.rl_device, geometry=self.cfg["task"]["env"]["geometry"])
        
        base_dir = Path(datasetPath)
        if validation: self.save_dir = base_dir.joinpath('validation')
        else: self.save_dir = base_dir.joinpath('train')
        if self.save_dir.exists(): shutil.rmtree(self.save_dir) # Remove existing contents in the directory
        self._make_directory()

        

        self.keypoints = torch.zeros((self.size, 16), dtype=torch.float) # 2D keypoints
        self.image_height = int(self.cfg["task"]["env"]["camera"]["size"][0])
        self.image_width = int(self.cfg["task"]["env"]["camera"]["size"][1])
        self.cameraImages = torch.zeros(
            (self.numEnvs, self.image_height, self.image_width, 3), # N * H * W * C
            dtype=torch.uint8
        )
        self.cameraImages.share_memory_()
        self.segmentation = torch.zeros(
            (self.numEnvs, self.image_height, self.image_width),
            dtype=torch.uint8
        )
        self.segmentation.share_memory_()

        self.numIteration = self.size // self.numEnvs
        self.numWorkers = 20
        self.runs = list()
        for _ in range(self.numWorkers): self.runs.append(Event())
        self.dones = list()
        for _ in range(self.numWorkers): self.dones.append(Event())
        startIndices, jobAllocation = self._allocate_jobs()
        self.writers = list()
        for i in range(self.numWorkers):
            writer = Process(
                target=imageWriter,
                args=(
                    self.runs[i], self.dones[i], self.numIteration, self.numEnvs,
                    self.cameraImages, self.segmentation, self.save_dir, 
                    startIndices.item(i), jobAllocation.item(i)
                )
            )
            writer.start()
            self.writers.append(writer)
        self.sP = setFixedInitPosition([0.4, 0., 0., 0.05, -0.3, 0.])
        # self.sP = PreCalculatedIK()

    def _make_directory(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.joinpath('rgb').mkdir(exist_ok=True)
        self.save_dir.joinpath('segmentation').mkdir(exist_ok=True)

    def _allocate_jobs(self) -> Tuple[np.ndarray, np.ndarray]:
        jobAllocation = np.full(self.numWorkers, (self.numEnvs / self.numWorkers), dtype=np.int32)
        remainingJobs = np.zeros_like(jobAllocation)
        remainingJobs[:(self.numEnvs % self.numWorkers)] = 1
        jobAllocation += remainingJobs
        startIndices = np.roll(np.cumsum(jobAllocation), 1)
        startIndices[0] = 0
        return startIndices, jobAllocation

    def loadPolicies(self, checkpoint):
        state = load_checkpoint(checkpoint)
        post_state = state.pop('post')
        self.model.policy.load_state_dict(post_state)

    def generate(self):
        self.model.policy.eval()
        print('Generating and writing images...')
        progressBar = tqdm(total=self.size)
        steps = 0
        resetCount = 0
        self._make_initial_contact()
        observation, image, segmentation = self._reset_env()
        while steps < self.size:
            self._write_images(image, segmentation)
            action = self.model.step(observation)[0]
            observation = unscale_transform(observation, self.env._observations_scale.low, self.env._observations_scale.high)
            self.keypoints[steps:(steps + self.numEnvs)] = observation[:, 12:28].cpu()
            observation, image, segmentation, reset = self._step_env(action)
            steps += self.numEnvs
            progressBar.update(self.numEnvs)
            resetCount += torch.count_nonzero(reset)
            if resetCount > self.numEnvs:
                self._make_initial_contact()
                resetCount = 0
        progressBar.close()

    def _make_initial_contact(self):
        num_samples = self.numEnvs * 2
        T_O, T_G, q_R = self._get_fixed_initial_joint_position(num_samples)
        self.env.push_data(T_O, T_G, q_R)

    def _get_fixed_initial_joint_position(self, num_samples: int) -> Tuple[torch.Tensor, ...]:
        T_O, T_G = self.generator.sample(num_samples)
        q_R = torch.zeros((num_samples, 6), dtype=torch.float, device=self.rl_device)

        n_env = self.cfg['num_envs'] #24576 #128
        if n_env == '':
            n_env = 24576

        EE_position = T_O[:n_env, :3].clone().detach()
        EE_position[:, 2] = 0.52
        joint_pose = self.sP.fixed_init_pos(n_env)
        # joint_pose = self.sP.pre_calculated_IK(EE_position)
        
        q_R[:] = torch.concatenate((joint_pose, joint_pose), 0)

        return T_O, T_G, q_R

    def _stack_variable(self, data: Dict[str, List[torch.Tensor]], max_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        stacked_data: Dict[str, torch.Tensor] = dict()
        for key, tensor_list in data.items():
            stacked_data[key] = torch.cat(tensor_list, dim=0)
        if max_size is not None:
            for key in data.keys():
                stacked_data[key] = stacked_data[key][:max_size]
        return stacked_data

    def _step_env(self, action: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        obs_dict, _, reset, _ = self.env.step(action)
        observation = torch.clone(obs_dict["obs"])
        image = self.env.camera_image
        segmentation = self.env.segmentation
        reset = torch.clone(reset)
        return observation, image, segmentation, reset

    def _reset_env(self) -> Tuple[torch.Tensor, ...]:
        obs_dict = self.env.reset()
        observation = torch.clone(obs_dict["obs"])
        image = self.env.camera_image
        segmentation = self.env.segmentation
        return observation, image, segmentation

    def _write_images(self, image: torch.Tensor, segmentation: torch.Tensor):
        self.cameraImages[:] = image.cpu()
        self.segmentation[:] = segmentation.to(device='cpu', dtype=torch.uint8).squeeze()
        self._run_workers()
        self._wait_workers()

    def _run_workers(self):
        for i in range(self.numWorkers): self.runs[i].set()
        
    def _wait_workers(self):
        for i in range(self.numWorkers):
            self.dones[i].wait()
            self.dones[i].clear()

    def post_process(self):
        u_coordinate = torch.arange(0, 16, step=2)
        v_coordinate = torch.arange(1, 16, step=2)
        u_coord_range_test = torch.logical_and(
            (self.keypoints[:, u_coordinate] >= 0), 
            (self.keypoints[:, u_coordinate] <= self.image_width)
        )
        v_coord_range_test = torch.logical_and(
            (self.keypoints[:, v_coordinate] >= 0),
            (self.keypoints[:, v_coordinate] <= self.image_height)
        )
        num_of_valid_keypoints = (
            torch.sum(u_coord_range_test.to(dtype=torch.int), dim=-1) + 
            torch.sum(v_coord_range_test.to(dtype=torch.int), dim=-1)
        )
        image_validity = (num_of_valid_keypoints > 4).nonzero().squeeze()
        np.save(str(self.save_dir.joinpath("processed_data_8_points.npy")), self.keypoints.reshape((-1, 8, 2)).numpy())
        np.save(str(self.save_dir.joinpath("trainable_images_8_points.npy")), image_validity.numpy())


## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)

# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
# num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)

@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    env: Domain = immgym_task_map[cfg.task_name](
        omegaconf_to_dict(cfg.task), cfg["sim_device"], cfg["graphics_device_id"], cfg["headless"], cfg["use_states"]
    )
    env_cfg = {}
    env_cfg["num_obs"] = env.num_obs # 69
    env_cfg["num_states"] = env.num_states # 115
    # env_cfg["num_student_obs"] = env.num_student_obs
    env_cfg["num_acts"] = env.num_acts

    generator = DataGenerator(
        cfg, env, Model(cfg, env_cfg), '/home/user/backup_240213/personal/hardware/v3_left_bump', int(cfg["dataset_size"]), cfg["validation"]
    )
    generator.loadPolicies('/home/user/backup_240213/personal/hardware/sim2real-robot-arm/trains/Bump/joint/24-09-05-16-35-seed-42/Bump_ep_1100_rew_1031.3053.pth')
    generator.generate()
    generator.post_process()

if __name__ == "__main__":
    launch_rlg_hydra()
