import torch
import numpy as np
import time


def swap_and_flatten(array):
    s = array.shape
    return array.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

class AverageMeter(torch.nn.Module):
    def __init__(self, in_shape, max_size):
        super(AverageMeter, self).__init__()
        self.max_size = max_size
        self.current_size = 0
        self.register_buffer("mean", torch.zeros((in_shape,1), dtype = torch.float32))

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0)
        size = np.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = ((self.mean * old_size + new_mean * size) / size_sum)

    def clear(self):
        self.current_size = 0
        self.mean.fill_(0)

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean.squeeze(0).cpu().numpy()

def safe_filesystem_op(func, *args, **kwargs):
    """
    This is to prevent spurious crashes related to saving checkpoints or restoring from checkpoints in a Network
    Filesystem environment (i.e. NGC cloud or SLURM)
    """
    num_attempts = 5
    for attempt in range(num_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            print(f'Exception {exc} when trying to execute {func} with args:{args} and kwargs:{kwargs}...')
            wait_sec = 2 ** attempt
            print(f'Waiting {wait_sec} before trying again...')
            time.sleep(wait_sec)

    raise RuntimeError(f'Could not execute {func}, give up after {num_attempts} attempts...')

def safe_save(state, filename):
    return safe_filesystem_op(torch.save, state, filename)

def safe_load(filename):
    return safe_filesystem_op(torch.load, filename)

def save_checkpoint(filename, state):
    print("=> saving checkpoint '{}'".format(filename + '.pth'))
    safe_save(state, filename + '.pth')

def load_checkpoint(filename):
    print("=> loading checkpoint '{}'".format(filename))
    state = safe_load(filename)
    return state