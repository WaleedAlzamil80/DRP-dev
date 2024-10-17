import torch
from torch import nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification
from Dataset import CustomDataset, load_data
from train.train import get_train

import warnings
warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda. is_available() else 'cpu'

