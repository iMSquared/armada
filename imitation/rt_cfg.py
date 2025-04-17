#!/usr/bin/env python3

from typing import List, Tuple
from pathlib import Path
from datetime import date
import pickle
import numpy as np

from imitation.perception.pose.utils.path import get_latest_file, ensure_directory
from loguru import logger

def _get_index():
    today = date.today()
    return today.strftime('%Y-%m-%d')


class RuntimeConfig:
    # == where to save data ==
    root: str = 'imitation/perception/pose/runtime'

    # == tag ==
    tag_family: str = 'tag36h11'
    tag_size: float = 0.1655

    # == robot ip ==
    robot_ip: str = "kim-MS-7C82"

    def __init__(self):
        ensure_directory(self.root)

    @property
    def default_T_ec(self):
        return np.asarray([[-0.10237489, 0.94428195, -0.31281141, 0.0441859],
                           [-0.99314449, -0.11486116, -0.02170082, 0.06869562],
                           [-0.05642157, 0.30844532, 0.94956732, 0.02734694],
                           [0., 0., 0., 1.]],
                          dtype=np.float32)

    @property
    def cam_ids(self) -> List[str]:
        # return ['043322074870', '313522072186', '234322307454', '049222070927'] # D455 234322307454 extra D435 234322070242
        return ['234322307454']


    @property
    def calib_data_file(self) -> Path:
        return Path(self.root) / 'pq_list.pkl'

    @property
    def transforms_file(self) -> Path:
        return Path(self.root) / 'transforms.pkl'

    @property
    def extrinsics_file(self) -> Path:
        return Path(self.root) / 'extrinsics.pkl'

    @property
    def table_color_file(self) -> Path:
        return Path(self.root) / 'table_color.pkl'

    
    @property
    def extrinsics(self) -> Path:
        with open(self.extrinsics_file, 'rb') as fp:
            extrinsics = pickle.load(fp)
        return [extrinsics[dev_id] for dev_id in self.cam_ids]

    @property
    def new_task_file(self) -> Path:
        task_files = list(Path(self.root).glob('task*.pkl'))
        num_tasks = len(task_files)
        return Path(self.root) / F'task{num_tasks:04d}.pkl'
    
    def april_offset_file(self, object: str) -> Path:
        return Path(self.root) / F'april_tag_offset_{object}.pkl'

    @property
    def task_file(self) -> Path:
        return get_latest_file(Path(self.root), 'task*.pkl')
    
    @property
    def new_task_cfg_file(self) -> Path:
        tcfg_files = list(Path(self.root).glob('tcfg*.pkl'))
        num_tasks = len(tcfg_files)
        return Path(self.root) / F'tcfg{num_tasks:04d}.pkl'
    
    @property
    def task_cfg_file(self) -> Path:
        return get_latest_file(Path(self.root), 'tcfg*.pkl')


def main():
    cfg = RuntimeConfig()
    logger.info(cfg.calib_data_file)
    logger.info(cfg.transforms_file)
    logger.info(cfg.extrinsics_file)
    logger.info(cfg.new_task_file)
    

if __name__ == '__main__':
    main()
