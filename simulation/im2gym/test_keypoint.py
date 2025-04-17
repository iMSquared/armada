import isaacgym
from isaacgym import gymapi
from isaacgym.torch_utils import normalize

from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import hydra

from simulation.utils.generator import generatorBuilder, sampleGenerator
from simulation.utils.misc import set_np_formatting, set_seed
# from simulation.im2gym.algos.mapping import MapActionToJointPosition
from utils.reformat import omegaconf_to_dict
from simulation.im2gym.algos.policy import PreContactModel, Model as PostModel
from simulation.im2gym.algos.utils import load_checkpoint
from simulation.im2gym.tasks import immgym_task_map
from simulation.im2gym.tasks.domain import Domain

from typing import Tuple, Dict, List, Optional
from kornia.geometry.transform import Resize
from pathlib import Path
import torch
import cv2

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

import torch.nn.functional as F
from train_keypoint import Keypoint_Detect


class Tester:
    def __init__(
        self, cfg, env: Domain, 
        preModel: PreContactModel, postModel: PostModel, detector_checkpoint,
        preCheckpoint, dataSize: int, testSize: int, video_path
    ):
        self.cfg = cfg
        self.env = env
        self.preModel = preModel
        self.postModel = postModel
        self.dataSize = dataSize
        self.testSize = testSize
        self.videoPath = video_path

        self.height = int(self.cfg["task"]["env"]["camera"]["size"][0])
        self.width = int(self.cfg["task"]["env"]["camera"]["size"][1])
        self.record_video = True
        self.resize = Resize((240, 240))
        self.seed = self.cfg["seed"]
        
        config = self.cfg["transfer"]["params"]["config"]
        self.rl_device = cfg["rl_device"]
        self.nactors = config["num_actors"]
        self.reward_shaper = config["reward_shaper"]["scale_value"]

        env_list = ["Card", "HiddenCard", "Flip", "Hole", "Reorientation", "Bookshelf", "Bump"]
        env_dict = dict(zip(env_list, range(len(env_list))))

        self.sampler: sampleGenerator = generatorBuilder(
            config["name"], model_pre=None, writer=None, map=None, IK_query_size=None, 
            device=self.rl_device, geometry=self.cfg["task"]["env"]["geometry"]
        )
        self.IK_query_size = 100
        self.mapping = MapActionToJointPosition(
            env_dict["Card"], self.dataSize, 2, analytical=config["joint"]["analytical_ik"]
        )
        self.keypoint_detector = Keypoint_Detect(3, 8).to(self.rl_device)
        state = torch.load(detector_checkpoint)
        self.keypoint_detector.load_state_dict(state['model_state_dict'])
        self.keypoint_detector.eval()
        self._load_pre(preCheckpoint)

    def _load_pre(self, checkpoint):
        state = load_checkpoint(checkpoint)
        preState = state.pop('pre')
        self.preModel.policy.load_state_dict(preState)
        self.preModel.policy.eval()

    def load_post(self, checkpoint):
        state = load_checkpoint(checkpoint)
        postState = state.pop('post')
        self.postModel.policy.load_state_dict(postState)
        self.postModel.policy.eval()

    def test_epoch(self, epoch: int, summaryPath: Path) -> float:
        self.record_video = True
        self.video = cv2.VideoWriter(
            str(summaryPath.joinpath(f'epoch_{epoch}.mp4')),
            cv2.VideoWriter_fourcc(*'mp4v'), 20, (self.width, self.height)
        )
        success_rate = self.test()
        self.video.release()
        return success_rate

    def test(self) -> float:
        self.preModel.policy.eval()
        self.postModel.policy.eval()
        self.video = cv2.VideoWriter(
            self.videoPath,
            cv2.VideoWriter_fourcc(*'mp4v'), 20, (self.width, self.height)
        )
        self._make_initial_contact()
        self.env.extract = False
        state, observation = self._reset_env()
        reset_count = 0
        success_count = 0
        episode_count = 0
        episode_suc_count = 0
        while True:
            action, value, neglogp, mu, logstd = self.postModel.step(observation, state)
            state, observation, reward, reset = self._step_env(action)

            # reset_count += torch.count_nonzero(reset)
            # success_count += torch.count_nonzero(self.env.env_succeed)

            reset_count += reset[0]
            success_count += self.env._successes[0]

            env_reset_indices = reset.view(self.nactors, 1).all(dim=1).nonzero(as_tuple=False)
            env_suc_indices = self.env.env_succeed.view(self.nactors, 1).all(dim=1).nonzero(as_tuple=False)
            episode_count += torch.sum(self.env.progress_buf[env_reset_indices])
            episode_suc_count += torch.sum(self.env.progress_buf[env_suc_indices])
            if reset_count >= 10: break
        print(f"all tries {reset_count}, succeed {success_count} times and total episodes are {episode_count} and for success {episode_suc_count}")
        self.video.release()
        return success_count.item() / float(reset_count.item())

    def _make_initial_contact(self):
        numFeasible = 0
        keys = ['T_O', 'T_G', 'q_R']
        _data: Dict[str, List[torch.Tensor]] = {key: list() for key in keys}
        while numFeasible < self.dataSize:
            initialObjectPose, goalObjectPose = self.sampler.sample(self.IK_query_size)
            state = torch.cat([initialObjectPose, goalObjectPose], dim=-1)
            action, *_ = self.preModel.step(state)
            action[:, :4] = normalize(action[:, :4])
            result = self.mapping.convert(action, initialObjectPose, batchSize=self.IK_query_size)
            jointPosition, isFeasible = result[:, :9], (result[:, 9] > 0.5)
            _data['T_O'].append(initialObjectPose[isFeasible])
            _data['T_G'].append(goalObjectPose[isFeasible])
            _data['q_R'].append(jointPosition[isFeasible])
            numFeasible += torch.count_nonzero(isFeasible)
        teacherBuffer = self._stack_variable(_data, max_size=self.dataSize)
        self.env.push_data(teacherBuffer['T_O'], teacherBuffer['T_G'], teacherBuffer['q_R'])

    def _stack_variable(self, data: Dict[str, List[torch.Tensor]], max_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        stacked_data: Dict[str, torch.Tensor] = dict()
        for key, tensor_list in data.items():
            stacked_data[key] = torch.cat(tensor_list, dim=0)
        if max_size is not None:
            for key in data.keys():
                stacked_data[key] = stacked_data[key][:max_size]
        return stacked_data

    def _reset_env(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_dict = self.env.reset()
        observation = torch.clone(obs_dict["obs"])
        state = torch.clone(obs_dict["states"])
        points = self._get_points()
        observation[:, 18:34] = points.view(-1, 16)[:]
        return state, observation

    def _step_env(self, action) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_dict, reward, reset, _ = self.env.step(action)
        observation = torch.clone(obs_dict["obs"])
        state = torch.clone(obs_dict["states"])
        points = self._get_points()
        observation[:, 18:34] = points.view(-1, 16)[:]
        reward = torch.clone(reward) * self.reward_shaper # reward scaling
        reset = torch.clone(reset)
        return state, observation, reward, reset

    def _get_points(self) -> torch.Tensor:
        image = self.env.camera_image
        if self.record_video:
            image_numpy = image[0].to(device='cpu', non_blocking=True).numpy()
        image = image.permute((0, 3, 1, 2)).to(dtype=torch.float) / 255.0
        results, _ = self.keypoint_detector(image)
        results = F.softmax(results.reshape(-1, 240 * 320), dim=-1).reshape(-1, 8, 240, 320)
        points = (results.flatten(-2).argmax(-1))
        points = torch.stack([points%320, points//320],-1).to(torch.float)
        if self.record_video:
            for idx in range(8):
                image_numpy = cv2.circle(image_numpy, (int(points[0, idx, 0]), int(points[0, idx, 1])), 3, (0, 0, 255), -1)
            self.video.write(cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR))
        points[:, :, 0] = 2 * (points[:, :, 0] - 160) / 320.
        points[:, :, 1] = 2 * (points[:, :, 1] - 120) / 240.
        return points

    def close(self):
        self.mapping.close()


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

    # set numpy formatting for printing only
    set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    print("-------------------")
    rlg_config_dict = omegaconf_to_dict(cfg.task)
    
    env = immgym_task_map[cfg.task_name](rlg_config_dict, cfg["sim_device"], cfg["graphics_device_id"], cfg["headless"], cfg["use_states"])
    env_cfg = {}
    env_cfg["num_obs"] = env.num_obs
    env_cfg["num_states"] = env.num_states 
    env_cfg["num_acts"] = env.num_acts
    
    preModel = PreContactModel(cfg, env_cfg)
    Model = PostModel(cfg, env_cfg)

    num_episodes = cfg['num_episodes']
    videoDir = '../assets/test.mp4'
    tester = Tester(
        cfg, env, preModel, Model, cfg['detector_ckpt'], cfg['checkpoint_pre'], (2 * num_episodes), num_episodes, video_path=videoDir
    )
    
    tester.load_post(cfg['checkpoint'])
    success_rate = tester.test()
    print(f'Success rate: {success_rate}')
    tester.close()


if __name__ == "__main__":
    launch_rlg_hydra()
