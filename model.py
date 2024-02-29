import torch.nn as nn
import torch.nn.functional as F
import torch
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(state_dim, 32)
        self.l2 = nn.Linear(32, 64)
        self.l3 = nn.Linear(64, 32)
        self.l4 = nn.Linear(32, action_dim)
        self.scaling_factor = torch.tensor(5e-6)
    def forward(self, s):

        s = F.relu(self.l1(s))
        s = F.relu(self.l2(s))
        s = F.relu(self.l3(s))
        a_prob = self.l4(s) * self.scaling_factor
        return a_prob
