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

class Coach:
    def __init__(self, net, train_loader, test_loader, criterion, optimizer, lr, device, num_epoch):
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.device = device
        self.num_epoch = num_epoch

        # store
        self.train_loss, self.train_acc = [], []
        self.test_loss, self.test_acc = [], []
    
    def _train_epoch(self):
        self.net.train()
        dataloader = self.train_loader
        batch_loss, batch_correct = [], 0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            output = self.net(X)

            loss = self.criterion(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_loss.append(loss.item())
            pred = torch.argmax(output, dim=1)
            batch_correct += torch.sum(pred == y)
        
        epoch_loss = np.mean(batch_loss)
        epoch_acc = batch_correct.item() / len(dataloader.dataset)
        return epoch_loss, epoch_acc

    def _test_epoch(self):
        self.net.eval()
        dataloader = self.test_loader
        batch_loss, batch_correct = [], 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.net(X)
                loss = self.criterion(output, y)

                batch_loss.append(loss.item())
                pred = torch.argmax(output, dim=1)
                batch_correct += torch.sum(pred == y)

        epoch_loss = np.mean(batch_loss)
        epoch_acc = batch_correct.item() / len(dataloader.dataset)
        return epoch_loss, epoch_acc

    def train_test(self):
        start = time.time()
        for epoch in range(self.num_epoch):
            train_epoch_loss, train_epoch_acc = self._train_epoch()
            test_epoch_loss, test_epoch_acc = self._test_epoch()

            print("epoch: ", epoch+1, "/", self.num_epoch)
            print("[train] loss: ", train_epoch_loss, ", acc: ", train_epoch_acc, ", time: ", time.time()-start)
            print("[test] loss: ", test_epoch_loss, ", acc: ", test_epoch_acc)

            self.train_loss.append(train_epoch_loss)
            self.train_acc.append(train_epoch_acc)
            self.test_loss.append(test_epoch_loss)
            self.test_acc.append(test_epoch_acc)