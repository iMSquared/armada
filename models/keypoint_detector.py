from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
import torch

from typing import Dict, Tuple
from pathlib import Path
import numpy as np
import kornia
import cv2

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))
        return out3


class Down(nn.Module):
    """(convolution => [BN] => ReLU) with stride 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample((240, 320), mode='bilinear', align_corners=True)
            self.conv = nn.Conv2d(in_channels*2, out_channels, kernel_size=3, padding=1, bias=False)
        else:
            self.up = nn.ConvTranspose2d(in_channels*2, in_channels , kernel_size=3, stride=2)
            self.conv =  nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x1, x2):
        # input is CHW
        x = torch.cat([x1, x2], dim=1)
        x = self.up(x)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class KeypointEncoder(nn.Module):
    def __init__(self, n_channels: int):
        super(KeypointEncoder, self).__init__()
        self.down1 = Down(n_channels, 16)
        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.down2 = Down(16, 32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.down3 = Down(32, 64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.down4 = Down(64, 128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.res1 = ResidualBlock(128, 128)
        self.res2 = ResidualBlock(128, 128)
        self.res3 = ResidualBlock(128, 128)
        # Contrastive loss
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(34048, 256)

    def forward(self, x: torch.Tensor):
        x1 = self.down1(x)
        x2 = self.conv1(x1)
        x3 = self.down2(x2)
        x4 = self.conv2(x3)
        x5 = self.down3(x4)
        x6 = self.conv3(x5)
        x7 = self.down4(x6)
        x8 = self.conv4(x7)
        x = self.res1(x8)
        x = x + x8
        x = self.res2(x)
        x = x + x8
        x = self.res3(x)
        x = x + x8
        z = self.linear(self.flatten(x))
        return x, x1, x2, x3, x4, x5, x6, x7, z


""" Full assembly of the parts to form the complete network """
class Keypoint_Detect(nn.Module):
    def __init__(self, n_channels, n_points):
        super(Keypoint_Detect, self).__init__()
        self.n_channels = n_channels
        self.n_points = n_points

        self.keypoint_encoder = KeypointEncoder(n_channels)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.up1 = Up(128, 64, bilinear=False)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.up2 = Up(64, 32, bilinear=False)
        self.conv7 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.up3 = Up(32, 16, bilinear=False)
        self.conv8 = nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False)
        self.up4 = Up(16, n_points, bilinear=True)
        self.loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x, x1, x2, x3, x4, x5, x6, x7, z = self.keypoint_encoder(x)
        x = self.conv5(x)
        x = self.up1(x, x7)
        x = torch.cat([x, x6], dim=1)
        x = self.conv6(x)
        x = self.up2(x, x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv7(x)
        x = self.up3(x, x3)
        x = torch.cat([x, x2], dim=1)
        x = self.conv8(x)
        logits = self.up4(x, x1)
        return logits, z
    
    def calc_loss(self, heat_map: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return self.loss(F.log_softmax(heat_map.view(-1, (240 * 320)), dim=-1).view(-1, 8, 240, 320), gt)


class GenerateHeatmap():
    def __init__(self, height, width, num_parts):
        self.height = height
        self. width = width
        self.num_parts = num_parts
        self.sigma = 6
        size = 6 * self.sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * self.sigma + 1, 3 * self.sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

    def __call__(self, keypoints):
        hms = np.zeros(shape = (self.num_parts, self.height, self.width), dtype = np.float32)
        sigma = self.sigma
        for idx, pt in enumerate(keypoints):
            x, y = int(pt[0]), int(pt[1])
            if x < 0 or y < 0 or x >= self.width or y >= self.height:
                continue
            ul = int(x - 3 * sigma - 1), int(y - 3 * sigma - 1)
            br = int(x + 3 * sigma + 2), int(y + 3 * sigma + 2)

            c, d = max(0, -ul[0]), min(br[0], self.width) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], self.height) - ul[1]

            cc, dd = max(0, ul[0]), min(br[0], self.width)
            aa, bb = max(0, ul[1]), min(br[1], self.height)
            hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms


class Queue:
    def __init__(self, max_size: int, key_dim: int = 256, device: str = 'cpu'):
        self.max_size = max_size
        self._queue = F.normalize(
            torch.randn((max_size, key_dim), dtype=torch.float, device=device),
            dim=0
        )
        self.ptr = 0

    def enqueue_and_dequeue(self, key: torch.Tensor):
        batch_size = key.shape[0]
        start = self.ptr
        end = self.ptr + batch_size
        if end > self.max_size:
            self._queue[start:self.max_size, :] = key[:(self.max_size - start)]
            self._queue[:(end - self.max_size), :] = key[(self.max_size - start):]
            self.ptr = end - self.max_size
        else:
            self._queue[start:end, :] = key
            self.ptr = end

    @property
    def queue(self) -> torch.Tensor:
        return self._queue.detach().clone()


class KeypointDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: Path, limit_size=None):
        self.width = 320
        self.height = 240
        self.generateHeatmap = GenerateHeatmap(self.height, self.width, 8)
        self.rgb_file_list = self._get_file_list(data_path, 'rgb')
        self.seg_file_list = self._get_file_list(data_path, 'segmentation')
        self.data: np.ndarray = np.load(str(data_path.joinpath("processed_data_8_points.npy")))
        self.image_idx: np.ndarray = np.load(str(data_path.joinpath("trainable_images_8_points.npy")))
        if limit_size != None:
            self.image_idx = self.image_idx[:limit_size]

    def __len__(self):
        return self.image_idx.shape[0]
    
    def _get_file_list(self, data_path: Path, data_type: str) -> np.ndarray:
        files = data_path.joinpath(data_type).glob('*.png')
        file_list = sorted(files, key=lambda file: int(str(file.name)[:-4]))
        file_list = [str(file) for file in file_list]
        return np.array(file_list, dtype=np.dtype(f'U{len(file_list[-1])}'))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        bgr_image = cv2.imread(str(self.rgb_file_list[self.image_idx[index]]))
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        segmentation = cv2.imread(str(self.seg_file_list[self.image_idx[index]]), cv2.IMREAD_GRAYSCALE)
        keypoints = self.data[self.image_idx[index]]
        heatmap = self.generateHeatmap(keypoints)
        return {
            'image': torch.from_numpy(rgb_image),
            'heatmap': torch.from_numpy(heatmap),
            'segmentation': torch.from_numpy(segmentation)
        }


class Runner:
    def __init__(
        self, model: Keypoint_Detect, optimizer: torch.optim.Optimizer, 
        trainDataset: DataLoader, validationDataset: DataLoader, texturePath: Path, 
        device: str, epochs: int, writer: SummaryWriter, summaryPath: Path
    ):
        self.model = model
        self.key_encoder = KeypointEncoder(self.model.n_channels).to(device=device)
        self.key_size = 256
        self.train_key_queue = Queue(max_size=30000, key_dim=self.key_size, device=device)
        self.validation_key_queue = Queue(max_size=500, key_dim=self.key_size, device=device)
        self.contrastive_loss = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.trainDataset = trainDataset
        self.trainDatasetSize = len(trainDataset.dataset)
        self.validationDataset = validationDataset
        self.texturePath = texturePath
        self.device = device
        self.epochs = epochs
        self.starting_epoch = 1
        self.writer = writer
        self.summaryPath = summaryPath
        self.m = 0.999
        self.temperature = 0.07
        self.augmentation = kornia.augmentation.ColorJiggle(brightness=0.3, contrast=0, saturation=0.3, hue=0.07)
        self._get_textures()
        self._initialize_key_encoder()

    def _get_textures(self):
        import glob
        from PIL import Image
        self.textures = []
        files = glob.glob(str(self.texturePath) + "/*.jpg")
        for file in files:
            img = Image.open(file)
            if (img.size[1] < 320) or (img.size[0] < 240):
                width = max(img.size[1],320)
                height = max(img.size[0],240)
                img = img.resize((int(width),int(height)))
            self.textures.append(np.array(img))
        files = glob.glob(str(self.texturePath) + "/*/*")
        for file in files:
            img = Image.open(file)
            if (img.size[1] < 320) or (img.size[0] < 240):
                width = max(img.size[1], 320)
                height = max(img.size[0], 240)
                img = img.resize((int(width), int(height)))
            self.textures.append(np.array(img))
        self.num_textures = len(self.textures)

    def _random_choose_texture(self):
        self.texture_buf = np.zeros((self.num_textures, 240, 320, 3),dtype=np.uint8)
        for i, texture in enumerate(self.textures):
            if texture.shape[1] > 320:
                x = np.random.randint(0, texture.shape[1] - 320)
            else: x = 0
            if texture.shape[0] > 240:
                y = np.random.randint(0, texture.shape[0] - 240)
            else: y = 0
            self.texture_buf[i] = texture[y:(y + 240), x:(x + 320)]
        self.texture_buf = torch.tensor(self.texture_buf).permute(0, 3, 1, 2).to(self.device)

    def augment_image(self, image: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        num_image = image.shape[0]
        apply_texture = torch.rand(3, num_image, device="cuda")

        ## apply to table
        table_mask = (seg == 1)
        color = apply_texture[0] > 0.5
        table_color_mask = table_mask[color].float()
        random_color = (torch.rand(num_image, 3, 1, 1, device="cuda") * 255).to(torch.uint8)
        image[color] = (1 - table_color_mask) * image[color] + (table_color_mask) * random_color[color]

        table_texture_mask = table_mask[~color].float()
        random_texture = torch.randint(0,self.num_textures,(num_image,), device="cuda")[~color]
        image[~color] = (1 - table_texture_mask) * image[~color] + (table_texture_mask) * self.texture_buf[random_texture]

        ## apply to floor
        floor_mask = (seg == 4)
        color = apply_texture[1] > 0.5
        floor_color_mask = floor_mask[color].float()
        random_color = (torch.rand(num_image, 3, 1, 1, device="cuda") * 255).to(torch.uint8)
        image[color] = (1-floor_color_mask)*image[color]+(floor_color_mask)*random_color[color]

        floor_texture_mask = floor_mask[~color].float()
        random_texture = torch.randint(0, self.num_textures, (num_image,), device="cuda")[~color]
        image[~color] = (1 - floor_texture_mask) * image[~color] + (floor_texture_mask) * self.texture_buf[random_texture]

        ## apply to background
        back_mask = (seg == 0)
        color = apply_texture[2] > 0.5
        back_color_mask = back_mask[color].float()
        random_color = (torch.rand(num_image, 3, 1, 1, device="cuda") * 255).to(torch.uint8)
        image[color] = (1 - back_color_mask) * image[color] + (back_color_mask) * random_color[color]

        back_texture_mask = back_mask[~color].float()
        random_texture = torch.randint(0, self.num_textures, (num_image,), device="cuda")[~color]
        image[~color] = (1-back_texture_mask) * image[~color] + (back_texture_mask) * self.texture_buf[random_texture]

        ## apply to the robot
        robot_mask = (seg == 3)
        color = apply_texture[2] > 0.5
        back_color_mask = robot_mask[color].float()
        random_color = (torch.rand(num_image, 3, 1, 1, device="cuda") * 255).to(torch.uint8)
        image[color] = (1 - back_color_mask) * image[color] + (back_color_mask) * random_color[color]

        robot_texture_mask = robot_mask[~color].float()
        random_texture = torch.randint(0, self.num_textures, (num_image,), device="cuda")[~color]
        image[~color] = (1-robot_texture_mask) * image[~color] + (robot_texture_mask) * self.texture_buf[random_texture]
        
        image = self.augmentation(image / 255.)
        return image

    def _initialize_key_encoder(self):
        for param_q, param_k in zip(self.model.keypoint_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self):
        for epoch in range(self.starting_epoch, (self.epochs + 1)):
            print(f"Epoch {epoch}\n---------------------------------")
            trainLoss = self._train_epoch()
            validationLoss = self._validate_epoch()
            self._write_epoch_train_result(epoch, trainLoss, validationLoss)
            self._save_checkpoint(epoch)
            if epoch > 3:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 1e-4

    def _write_epoch_train_result(self, epoch: int, trainLoss: Tuple[float, float], validationLoss: Tuple[float, float]):
        trainKeypointLoss = trainLoss[0]
        trainContrastiveLoss = trainLoss[1]
        validationKeypointLoss = validationLoss[0]
        validationContrastiveLoss = validationLoss[1]
        self.writer.add_scalar('Loss/train', (trainKeypointLoss + trainContrastiveLoss), epoch)
        self.writer.add_scalar('Loss/keypoint_train', trainKeypointLoss, epoch)
        self.writer.add_scalar('Loss/contrastive_train', trainContrastiveLoss, epoch)
        self.writer.add_scalar('Loss/validation', (validationKeypointLoss + validationContrastiveLoss), epoch)
        self.writer.add_scalar('Loss/keypoint_validation', validationKeypointLoss, epoch)
        self.writer.add_scalar('Loss/contrastive_validation', validationContrastiveLoss, epoch)

    def _train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        self.key_encoder.train()
        keypointLoss = 0.0
        contrastiveLoss = 0.0
        data_count = 0

        batch: Dict[str, torch.Tensor]
        for batchIndex, batch in enumerate(self.trainDataset):
            self._random_choose_texture()
            image = batch['image'].permute([0, 3, 1, 2]).to(self.device).to(dtype=torch.float)
            segment = batch['segmentation'].unsqueeze(1).to(self.device).to(dtype=torch.float) 
            heatmap = batch['heatmap'].to(self.device).to(dtype=torch.float)
            batch_size = image.shape[0]
            batchLoss = 0.0

            result, batchContrastiveLoss, key = self._calculate_contrastive_loss(image, segment, self.train_key_queue)
            batchLoss += batchContrastiveLoss
            batchKeypointLoss = self.model.calc_loss(result, heatmap)
            batchLoss += batchKeypointLoss
            self.optimizer.zero_grad(set_to_none=True)
            batchLoss.backward()
            self.optimizer.step()

            # Momentum update
            for param_q, param_k in zip(self.model.keypoint_encoder.parameters(), self.key_encoder.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            self.train_key_queue.enqueue_and_dequeue(key)

            prev_batch_ratio = data_count / float(data_count + batch_size)
            keypointLoss = prev_batch_ratio * keypointLoss + (1 - prev_batch_ratio) * batchKeypointLoss.detach().item()
            contrastiveLoss = prev_batch_ratio * contrastiveLoss + (1 - prev_batch_ratio) * batchContrastiveLoss.detach().item()
            data_count += batch_size
            if (batchIndex + 1) % 10 == 0:
                print(
                    f'batch loss: {batchLoss.item():>7.2f}, '
                    f'keypoint loss: {keypointLoss:.2f}, '
                    f'contrastive loss: {contrastiveLoss:.2f} '
                    f'[{data_count:>7d}/{self.trainDatasetSize:>7d}]'
                )
        return keypointLoss, contrastiveLoss
    
    def _validate_epoch(self) -> Tuple[float, float]:
        self.model.eval()
        self.key_encoder.eval()
        keypointLoss = 0.0
        contrastiveLoss = 0.0
        data_count = 0

        batch: Dict[str, torch.Tensor]
        with torch.no_grad():
            for batch in self.validationDataset:
                self._random_choose_texture()
                image = batch['image'].permute([0, 3, 1, 2]).to(self.device).to(dtype=torch.float)
                segment = batch['segmentation'].unsqueeze(1).to(self.device).to(dtype=torch.float) 
                heatmap = batch['heatmap'].to(self.device).to(dtype=torch.float)
                batch_size = image.shape[0]

                result, batchContrastiveLoss, key = self._calculate_contrastive_loss(image, segment, self.validation_key_queue)
                batchKeypointLoss = self.model.calc_loss(result, heatmap)
                self.validation_key_queue.enqueue_and_dequeue(key)

                prev_batch_ratio = data_count / float(data_count + batch_size)
                keypointLoss = prev_batch_ratio * keypointLoss + (1 - prev_batch_ratio) * batchKeypointLoss.detach().item()
                contrastiveLoss = prev_batch_ratio * contrastiveLoss + (1 - prev_batch_ratio) * batchContrastiveLoss.detach().item()
                data_count += batch_size
        
        print(
            f'Keypoint loss (validation): {keypointLoss:.2f}, '
            f'Contrastive loss (validation): {contrastiveLoss:.2f}'
        )
        return keypointLoss, contrastiveLoss

    def _calculate_contrastive_loss(self, image: torch.Tensor, segment: torch.Tensor, key_queue: Queue) -> Tuple[torch.Tensor, ...]:
        result: torch.Tensor
        query: torch.Tensor
        image_q = self.augment_image(image, segment)
        image_k = self.augment_image(image, segment)        
        result, query = self.model(image_q)
        key: torch.Tensor = self.key_encoder(image_k)[-1]
        key = key.detach()
        # query: (N, C), key: (N, C), queue: (K, C) where N: batch size, C: key size and K: queue size
        l_pos = torch.bmm(query.view(-1, 1, self.key_size), key.view(-1, self.key_size, 1)).squeeze(dim=-1)
        l_neg = torch.mm(query, key_queue.queue.transpose(0, 1))
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
        batchContrastiveLoss: torch.Tensor = self.contrastive_loss(logits, labels)
        return result, batchContrastiveLoss, key

    def _save_checkpoint(self, epoch: int):
        checkpointPath = str(self.summaryPath) + f'/epoch_{epoch}_8points.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, checkpointPath)
        
    def load_checkpoint(self, checkpoint_path: str, epoch: int):
        checkpoint_path += f'/epoch_{epoch}_8points.pth'
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state['model_state_dict'])
        self._initialize_key_encoder()
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.starting_epoch = epoch + 1

