import torch
import torch.nn as nn
import torchsummary
import torch.optim as optim

import yaml
import matplotlib.pyplot as plt

from src.data.dataset import create_supdataset
from src.utils.seeds import fix_seed
from src.visualization.visualize import plot
from src.models.models import CNN
from src.models.coachs import Coach

def main():

    with open('./config/config_sup.yaml') as file:
        config_file = yaml.safe_load(file)
    print(config_file)

    ROOT = config_file['config']['root']
    NUM_EPOCH = config_file['config']['num_epoch']
    LR = config_file['config']['lr']
    BATCH_SIZE = config_file['config']['batch_size']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader = create_supdataset(root=ROOT, download=True, batch_size=BATCH_SIZE)

    net = CNN(widen_factor=1).to(device) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

    coach = Coach(net, train_loader, test_loader, criterion, optimizer, LR, device, NUM_EPOCH)
    coach.train_test()

    plot({"train": coach.train_loss, "test": coach.test_loss}, "loss")
    plot({"train": coach.train_acc, "test": coach.test_acc}, "acc")

if __name__ == "__main__":
    fix_seed()
    main()