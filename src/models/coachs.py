import torch
import torch.nn as nn
import numpy as np
import time

from src.models.optim import adjust_learning_rate

class CoachSimCLR:
    def __init__(self, net, unsup_loader, criterion, optimizer, lr, device, num_epoch, warmup_epoch):
        self.net = net
        self.unsup_loader = unsup_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.device = device
        self.num_epoch = num_epoch
        self.warmup_epoch = warmup_epoch

        # store
        self.train_loss = []

    def _train_epoch(self):
        self.net.train()
        dataloader = self.unsup_loader
        batch_loss = []

        for (X_i, X_j), _ in dataloader:
            X_i, X_j = X_i.to(self.device), X_j.to(self.device)

            z_i, z_j = self.net(X_i, X_j)
            loss = self.criterion(z_i, z_j)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_loss.append(loss.item())
        
        epoch_loss = np.mean(batch_loss)

        return epoch_loss

    def train(self):
        start = time.time()
        for epoch in range(self.num_epoch):
            adjust_learning_rate(self.optimizer, self.lr, epoch, self.num_epoch, self.warmup_epoch)

            train_epoch_loss = self._train_epoch()

            print("epoch: ", epoch+1, "/", self.num_epoch)
            print("[train] loss: ", train_epoch_loss, ", time: ", time.time()-start)

            self.train_loss.append(train_epoch_loss)