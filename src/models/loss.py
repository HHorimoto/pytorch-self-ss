import torch
from torch import nn
import torch.nn.functional as F

class NT_Xent(nn.Module):
    def __init__(self, device, batch_size=128, temperature=1):
        super(NT_Xent, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size+i] = 0 
            mask[batch_size+i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        # -- #
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        # -- #
        sim_i_j = torch.diag(sim,  self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        # -- #
        labels = torch.zeros(N).to(positive_samples.to(self.device)).long()
        loss = self.criterion(logits, labels)
        loss = loss / N
        return loss