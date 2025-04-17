from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.utils.data
import torch

from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

import os, sys
from pathlib import Path
BASEDIR = Path(__file__).parent.parent.parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))

from models.keypoint_detector import *

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('dataset_path', type=str)
    # parser.add_argument('texture_path', type=str)
    # parser.add_argument('--checkpoint', type=str)
    # parser.add_argument('--epoch', type=int)
    # args = parser.parse_args()
    detector = Keypoint_Detect(3, 8)
    optimizer = torch.optim.Adam(detector.parameters(), lr=2e-4, weight_decay=1e-8)
    # datasetPath = Path(args.dataset_path).expanduser()
    datasetPath = Path('/home/user/backup_240213/personal/hardware/v3_left_bump')
    trainingDatasetPath = datasetPath.joinpath('train')
    validationDatasetPath = datasetPath.joinpath('validation')
    currentTime = datetime.today()
    runName = f'{str(currentTime.year)[-2:]}-{currentTime.month:02d}-{currentTime.day:02d}-{currentTime.hour:02d}-{currentTime.minute:02d}'
    summaryPath = datasetPath.joinpath('result').joinpath(runName)
    summaryPath.mkdir(exist_ok=True)
    # texturePath = Path(args.texture_path).expanduser()
    texturePath = Path('/home/user/workspace/VIMABench/vima_bench/tasks/assets/textures')
    writer = SummaryWriter(summaryPath)
    runner = Runner(
        model=detector.to("cuda"), optimizer=optimizer, 
        trainDataset=DataLoader(
            KeypointDataset(trainingDatasetPath),
            batch_size=300,
            num_workers=8,
            pin_memory=True
        ),
        validationDataset=DataLoader(
            KeypointDataset(validationDatasetPath, limit_size=20000),
            batch_size=100,
            num_workers=8,
            pin_memory=True
        ),
        texturePath=texturePath,
        device="cuda", epochs=80, writer=writer, summaryPath=summaryPath
    )
    # if args.epoch is not None:
    #     runner.load_checkpoint(args.checkpoint, args.epoch)
    # runner.load_checkpoint('/home/user/backup_240213/personal/hardware/sim2real-robot-arm/simulation/imm-gym/bigger_dataset/result/24-07-08-20-06', 3)
    runner.train()
