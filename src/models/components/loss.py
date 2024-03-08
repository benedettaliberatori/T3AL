from typing import Any
import torch
import torch.nn as nn

ce = nn.CrossEntropyLoss()
bce_cls = nn.BCEWithLogitsLoss()


class ByolLoss(nn.Module):
    def __init__(self, smoothing=0.2):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        pred = pred / pred.norm(dim=-1, keepdim=True)
        target = target / target.norm(dim=-1, keepdim=True)
        smoothed_target = (1 - self.smoothing) * target + self.smoothing / target.size(
            -1
        )
        loss = 2 - 2 * (pred * smoothed_target.detach()).sum(dim=0).mean()
        return loss


def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=0)
    return entropy
