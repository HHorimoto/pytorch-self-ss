import torch
import torch.nn as nn
import torchsummary
import torch.optim as optim

import yaml
import matplotlib.pyplot as plt

from src.data.dataset import create_dataset
from src.utils.seeds import fix_seed
from src.visualization.visualize import plot
from src.models.models import CNN, SimCLR
from src.models.loss import NT_Xent
from src.models.optim import LARS
from src.models.coachs import CoachSimCLR

def main():

    with open('config.yaml') as file:
        config_file = yaml.safe_load(file)
    print(config_file)

    ROOT = config_file['config']['root']
    NUM_EPOCH = config_file['config']['num_epoch']
    WARMUP_EPOCH = config_file['config']['warmup_epoch']
    BATCH_SIZE = config_file['config']['batch_size']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    unsup_loader = create_dataset(root=ROOT, download=True, batch_size=BATCH_SIZE)

    encoder = CNN(widen_factor=1).to(device) 
    net = SimCLR(encoder).to(device) 

    LR = 0.3 * (BATCH_SIZE / 256)
    criterion = NT_Xent(device, BATCH_SIZE, 0.5)

    optimizer = LARS(net.parameters(), lr=LR, weight_decay=1e-5, momentum=0.9, eta=0.001, 
                     weight_decay_filter=False, lars_adaptation_filter=False)

    coach = CoachSimCLR(net, unsup_loader, criterion, optimizer, LR, device, NUM_EPOCH, WARMUP_EPOCH)
    coach.train()

if __name__ == "__main__":
    fix_seed()
    main()