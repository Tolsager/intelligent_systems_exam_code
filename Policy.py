import torch.nn.functional as F
from torch import nn


class Policy(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(in_dim, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, out_dim)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)